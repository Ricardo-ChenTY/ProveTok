"""LLM Backend: 抽象接口 + HuggingFace 实现

支持的后端:
1. DummyLLM - 用于测试，不需要 GPU
2. HuggingFaceLLM - 本地 LLaMA 2-7B-Chat 等 HF 模型
3. 可扩展 API 后端（OpenAI / vLLM 等）

使用方式:
    llm = create_llm_backend("huggingface", model_name="meta-llama/Llama-2-7b-chat-hf")
    output = llm.generate(prompt, token_embeddings=embs, constrained_vocab=vocab)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any, Tuple
import torch
import torch.nn as nn


@dataclass
class LLMOutput:
    """LLM 生成结果"""
    text: str                                    # 原始生成文本
    slot_values: Dict[str, str] = field(default_factory=dict)  # 解析出的 slot values
    logprobs: Optional[List[float]] = None       # token-level log probs
    hidden_states: Optional[torch.Tensor] = None  # 最后一层隐藏状态 (用于训练)


class BaseLLMBackend(ABC):
    """LLM Backend 抽象基类

    所有后端必须实现:
    1. generate(): 给定 prompt + token embeddings 生成文本
    2. get_hidden_dim(): 返回隐藏层维度
    3. encode_prompt(): 将 prompt 编码为向量
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        token_embeddings: Optional[torch.Tensor] = None,
        constrained_vocab: Optional[Dict[str, Set[str]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> LLMOutput:
        """生成文本

        Args:
            prompt: 输入 prompt
            token_embeddings: (N, D) BET token embeddings，注入到 LLM
            constrained_vocab: slot -> 合法值集合，用于 constrained decoding
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling

        Returns:
            LLMOutput
        """
        pass

    @abstractmethod
    def get_hidden_dim(self) -> int:
        """返回 LLM 隐藏层维度"""
        pass

    @abstractmethod
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """将 prompt 编码为向量 (用于 cross-attention 等)

        Returns:
            (seq_len, hidden_dim) 编码结果
        """
        pass


class TokenProjector(nn.Module):
    """将 BET token embeddings 投影到 LLM embedding space

    根据 proposal: token embeddings 通过线性投影注入 LLM
    """

    def __init__(self, bet_dim: int = 32, llm_dim: int = 4096, num_layers: int = 2):
        super().__init__()
        if num_layers == 1:
            self.proj = nn.Linear(bet_dim, llm_dim)
        else:
            layers = [nn.Linear(bet_dim, llm_dim), nn.GELU()]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(llm_dim, llm_dim), nn.GELU()])
            layers.append(nn.Linear(llm_dim, llm_dim))
            self.proj = nn.Sequential(*layers)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: (N, bet_dim)
        Returns:
            (N, llm_dim)
        """
        return self.proj(token_embeddings)


# ---------------------------------------------------------------------------
# Dummy Backend (用于测试)
# ---------------------------------------------------------------------------

class DummyLLM(BaseLLMBackend):
    """不需要 GPU 的假 LLM，用于管线测试"""

    def __init__(self, hidden_dim: int = 4096):
        self._hidden_dim = hidden_dim

    def generate(
        self,
        prompt: str,
        token_embeddings: Optional[torch.Tensor] = None,
        constrained_vocab: Optional[Dict[str, Set[str]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> LLMOutput:
        # 根据 constrained vocab 生成假的 slot values
        slot_values: Dict[str, str] = {}
        if constrained_vocab:
            for slot, values in constrained_vocab.items():
                sorted_vals = sorted(values)
                slot_values[slot] = sorted_vals[0] if sorted_vals else "unspecified"

        # 构造假文本
        parts = []
        for slot, val in slot_values.items():
            parts.append(f"{slot}: {val}")
        text = "; ".join(parts) if parts else "No findings."

        return LLMOutput(
            text=text,
            slot_values=slot_values,
            logprobs=None,
            hidden_states=None,
        )

    def get_hidden_dim(self) -> int:
        return self._hidden_dim

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        # 返回假向量
        seq_len = min(len(prompt.split()), 128)
        return torch.randn(seq_len, self._hidden_dim)


# ---------------------------------------------------------------------------
# HuggingFace Backend (LLaMA 2-7B-Chat 等)
# ---------------------------------------------------------------------------

class HuggingFaceLLM(BaseLLMBackend):
    """HuggingFace Transformers 后端

    支持 LLaMA 2-7B-Chat、Mistral 等 causal LM。
    Token embeddings 通过 TokenProjector 投影后 prepend 到 prompt 前。

    用法:
        llm = HuggingFaceLLM(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            device="cuda",
            load_in_4bit=True,
        )
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cuda",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bet_dim: int = 32,
        torch_dtype: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.bet_dim = bet_dim
        self._torch_dtype = torch_dtype

        # 延迟加载（避免 import 时就占 GPU）
        self._model = None
        self._tokenizer = None
        self._projector: Optional[TokenProjector] = None

    def _ensure_loaded(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._torch_dtype, torch.float16)

        # 量化配置
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # 加载 tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 加载模型
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = self.device

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        self._model.eval()

        # 创建投影层
        llm_dim = self._model.config.hidden_size
        self._projector = TokenProjector(
            bet_dim=self.bet_dim,
            llm_dim=llm_dim,
        ).to(self.device)

    def get_hidden_dim(self) -> int:
        self._ensure_loaded()
        return self._model.config.hidden_size

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        self._ensure_loaded()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].squeeze(0)

    def _build_inputs_with_tokens(
        self,
        prompt: str,
        token_embeddings: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将 BET token embeddings prepend 到 prompt embeddings 前

        返回 (input_embeds, attention_mask)
        """
        self._ensure_loaded()

        # Tokenize prompt
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # 获取 prompt 的 word embeddings
        word_emb_layer = self._model.get_input_embeddings()
        prompt_embeds = word_emb_layer(input_ids)  # (1, seq_len, hidden_dim)

        if token_embeddings is None or self._projector is None:
            attn_mask = torch.ones(prompt_embeds.shape[:2], device=self.device)
            return prompt_embeds, attn_mask

        # 投影 BET embeddings
        token_embeddings = token_embeddings.to(self.device)
        projected = self._projector(token_embeddings)  # (N, hidden_dim)
        projected = projected.unsqueeze(0)  # (1, N, hidden_dim)

        # Prepend: [BET tokens] + [prompt tokens]
        combined = torch.cat([projected, prompt_embeds], dim=1)
        attn_mask = torch.ones(combined.shape[:2], device=self.device)

        return combined, attn_mask

    def generate(
        self,
        prompt: str,
        token_embeddings: Optional[torch.Tensor] = None,
        constrained_vocab: Optional[Dict[str, Set[str]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> LLMOutput:
        self._ensure_loaded()

        input_embeds, attn_mask = self._build_inputs_with_tokens(prompt, token_embeddings)

        with torch.no_grad():
            outputs = self._model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # Decode
        generated_ids = outputs.sequences[0]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 解析 slot values (简单的 key: value 格式)
        slot_values = self._parse_slot_values(text, constrained_vocab)

        return LLMOutput(
            text=text,
            slot_values=slot_values,
        )

    def _parse_slot_values(
        self,
        text: str,
        constrained_vocab: Optional[Dict[str, Set[str]]] = None,
    ) -> Dict[str, str]:
        """从生成文本中解析 slot values

        简单实现：在文本中查找 constrained vocab 中的值
        """
        slot_values: Dict[str, str] = {}
        if constrained_vocab is None:
            return slot_values

        text_lower = text.lower()
        for slot, valid_values in constrained_vocab.items():
            best_val = "unspecified"
            for val in valid_values:
                if val.lower() in text_lower:
                    best_val = val
                    break
            slot_values[slot] = best_val

        return slot_values


# ---------------------------------------------------------------------------
# Registry & Factory
# ---------------------------------------------------------------------------

_LLM_REGISTRY: Dict[str, type] = {
    "dummy": DummyLLM,
    "huggingface": HuggingFaceLLM,
}


def create_llm_backend(
    backend_type: str = "dummy",
    **kwargs,
) -> BaseLLMBackend:
    """创建 LLM Backend

    Args:
        backend_type: "dummy" | "huggingface"
        **kwargs: 传递给对应 Backend 构造函数

    Returns:
        BaseLLMBackend 实例
    """
    if backend_type not in _LLM_REGISTRY:
        available = ", ".join(_LLM_REGISTRY.keys())
        raise ValueError(f"Unknown LLM backend: {backend_type}. Available: {available}")

    return _LLM_REGISTRY[backend_type](**kwargs)


def list_available_llm_backends() -> List[str]:
    return list(_LLM_REGISTRY.keys())
