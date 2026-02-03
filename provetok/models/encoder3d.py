"""3D Encoder 模块: 支持多种预训练模型

根据 proposal，需要真实 3D encoder 输出 feature map，再对 cell 区域 pooling。

支持的 Encoder:
1. ToyEncoder3D - 简单 CNN（测试用）
2. ResNet3D - 3D ResNet（可用 MedicalNet 预训练）
3. SwinUNETR - MONAI 的 Swin Transformer（医学影像 SOTA）
4. CT2RepEncoder - CT2Rep 论文的 encoder（对齐 baseline）

使用方式:
    encoder = create_encoder("swin_unetr", pretrained=True)
    features = encoder(volume)  # (B, C, D', H', W')
    cell_emb = encoder.pool_region(features, cell_bounds)  # (emb_dim,)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder3D(ABC, nn.Module):
    """3D Encoder 基类

    所有 encoder 必须实现:
    1. forward(): 返回 feature map
    2. pool_region(): 对指定区域进行 pooling
    3. get_output_dim(): 返回输出特征维度
    """

    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.emb_dim = emb_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码 3D volume

        Args:
            x: (B, C, D, H, W) 输入 volume

        Returns:
            (B, C', D', H', W') feature map
        """
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """返回输出特征维度"""
        pass

    def pool_region(
        self,
        features: torch.Tensor,
        region_slices: Tuple[slice, slice, slice],
        pool_type: str = "avg",
    ) -> torch.Tensor:
        """对指定区域进行 pooling

        Args:
            features: (B, C, D, H, W) 或 (C, D, H, W) feature map
            region_slices: (slice_d, slice_h, slice_w) 区域切片
            pool_type: "avg" 或 "max"

        Returns:
            (B, C) 或 (C,) pooled features
        """
        squeeze = False
        if features.dim() == 4:
            features = features.unsqueeze(0)
            squeeze = True

        # 提取区域
        sd, sh, sw = region_slices

        # 需要将 volume 空间的 slices 映射到 feature map 空间
        # 假设 feature map 和 volume 的空间比例一致或已预处理
        region = features[:, :, sd, sh, sw]

        # Pooling
        if pool_type == "avg":
            pooled = region.mean(dim=(2, 3, 4))  # (B, C)
        elif pool_type == "max":
            pooled = region.amax(dim=(2, 3, 4))  # (B, C)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

        if squeeze:
            pooled = pooled.squeeze(0)

        return pooled

    def encode_cells(
        self,
        volume: torch.Tensor,
        cell_bounds_list: List[Tuple[slice, slice, slice]],
        pool_type: str = "avg",
    ) -> torch.Tensor:
        """编码多个 cells

        Args:
            volume: (B, C, D, H, W) 或 (C, D, H, W) 或 (D, H, W) 输入 volume
            cell_bounds_list: 每个 cell 的边界切片列表
            pool_type: pooling 类型

        Returns:
            (num_cells, emb_dim) cell embeddings
        """
        # 标准化输入维度
        if volume.dim() == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        elif volume.dim() == 4:
            volume = volume.unsqueeze(0)  # (1, C, D, H, W)

        # 获取 feature map
        with torch.no_grad():
            features = self.forward(volume)  # (1, C', D', H', W')

        # 计算 feature map 和 volume 的比例
        vol_shape = volume.shape[2:]  # (D, H, W)
        feat_shape = features.shape[2:]  # (D', H', W')
        ratios = [f / v for f, v in zip(feat_shape, vol_shape)]

        # 对每个 cell 进行 pooling
        embeddings = []
        for bounds in cell_bounds_list:
            # 将 volume 空间的 bounds 映射到 feature 空间
            feat_bounds = self._map_bounds(bounds, vol_shape, feat_shape)
            emb = self.pool_region(features, feat_bounds, pool_type)
            embeddings.append(emb.squeeze(0))

        return torch.stack(embeddings)  # (num_cells, C')

    def _map_bounds(
        self,
        bounds: Tuple[slice, slice, slice],
        vol_shape: Tuple[int, int, int],
        feat_shape: Tuple[int, int, int],
    ) -> Tuple[slice, slice, slice]:
        """将 volume 空间的 bounds 映射到 feature map 空间"""
        mapped = []
        for slc, vs, fs in zip(bounds, vol_shape, feat_shape):
            ratio = fs / vs
            start = int(slc.start * ratio) if slc.start else 0
            stop = int(slc.stop * ratio) if slc.stop else fs
            # 确保至少有一个元素
            if stop <= start:
                stop = start + 1
            mapped.append(slice(start, min(stop, fs)))
        return tuple(mapped)


class ToyEncoder3D(BaseEncoder3D):
    """简单的 3D CNN Encoder（测试用）"""

    def __init__(
        self,
        in_channels: int = 1,
        emb_dim: int = 256,
        hidden_channels: List[int] = None,
    ):
        super().__init__(emb_dim)
        hidden_channels = hidden_channels or [32, 64, 128]

        layers = []
        prev_ch = in_channels
        for ch in hidden_channels:
            layers.extend([
                nn.Conv3d(prev_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
            ])
            prev_ch = ch

        self.encoder = nn.Sequential(*layers)
        self.output_channels = hidden_channels[-1]

        # Projection to emb_dim
        self.proj = nn.Conv3d(self.output_channels, emb_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.proj(features)

    def get_output_dim(self) -> int:
        return self.emb_dim


class ResNet3DEncoder(BaseEncoder3D):
    """3D ResNet Encoder

    支持:
    - 从头训练
    - 加载 MedicalNet 预训练权重
    - 加载 torchvision 的 video 模型权重
    """

    def __init__(
        self,
        in_channels: int = 1,
        emb_dim: int = 256,
        depth: int = 18,  # 18, 34, 50, 101
        pretrained_path: Optional[str] = None,
    ):
        super().__init__(emb_dim)
        self.in_channels = in_channels

        # 构建 ResNet3D
        self.encoder = self._build_resnet3d(depth)
        self.output_channels = 512 if depth < 50 else 2048

        # 输入通道适配
        if in_channels != 3:
            self.input_conv = nn.Conv3d(in_channels, 3, kernel_size=1)
        else:
            self.input_conv = None

        # Projection
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(self.output_channels, emb_dim),
        )

        # 加载预训练权重
        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _build_resnet3d(self, depth: int) -> nn.Module:
        """构建 3D ResNet 骨干网络"""
        # 简化实现：使用基本的 3D ResNet 块
        # 实际使用时建议用 MONAI 或 torchvision

        class BasicBlock3D(nn.Module):
            expansion = 1

            def __init__(self, in_ch, out_ch, stride=1, downsample=None):
                super().__init__()
                self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride, 1, bias=False)
                self.bn1 = nn.BatchNorm3d(out_ch)
                self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm3d(out_ch)
                self.downsample = downsample

            def forward(self, x):
                identity = x
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                if self.downsample:
                    identity = self.downsample(x)
                return F.relu(out + identity)

        def make_layer(in_ch, out_ch, blocks, stride=1):
            downsample = None
            if stride != 1 or in_ch != out_ch:
                downsample = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, 1, stride, bias=False),
                    nn.BatchNorm3d(out_ch),
                )
            layers = [BasicBlock3D(in_ch, out_ch, stride, downsample)]
            for _ in range(1, blocks):
                layers.append(BasicBlock3D(out_ch, out_ch))
            return nn.Sequential(*layers)

        # ResNet-18 configuration
        if depth == 18:
            layers = [2, 2, 2, 2]
        elif depth == 34:
            layers = [3, 4, 6, 3]
        else:
            layers = [2, 2, 2, 2]  # fallback

        return nn.Sequential(
            nn.Conv3d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, 2, 1),
            make_layer(64, 64, layers[0]),
            make_layer(64, 128, layers[1], stride=2),
            make_layer(128, 256, layers[2], stride=2),
            make_layer(256, 512, layers[3], stride=2),
        )

    def _load_pretrained(self, path: str):
        """加载预训练权重"""
        try:
            state_dict = torch.load(path, map_location='cpu')
            # 处理可能的 key 前缀差异
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.encoder.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {path}")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_conv is not None:
            x = self.input_conv(x)
        return self.encoder(x)

    def get_output_dim(self) -> int:
        return self.emb_dim


class SwinUNETREncoder(BaseEncoder3D):
    """MONAI SwinUNETR Encoder

    需要安装 MONAI: pip install monai
    """

    def __init__(
        self,
        in_channels: int = 1,
        emb_dim: int = 256,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 48,
        pretrained: bool = True,
    ):
        super().__init__(emb_dim)
        self.feature_size = feature_size

        try:
            from monai.networks.nets import SwinUNETR
        except ImportError:
            raise ImportError(
                "MONAI is required for SwinUNETR. "
                "Install with: pip install monai"
            )

        # 使用 SwinUNETR 但只取 encoder 部分
        self.swin = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=1,  # dummy, we only use encoder
            feature_size=feature_size,
        )

        # 输出通道数（SwinUNETR encoder 的最后一层）
        self.output_channels = feature_size * 16  # 768 for feature_size=48

        # Projection
        self.proj = nn.Conv3d(self.output_channels, emb_dim, kernel_size=1)

        if pretrained:
            self._load_pretrained()

    def _load_pretrained(self):
        """加载 MONAI 预训练权重"""
        try:
            # MONAI Hub 提供预训练权重
            # 这里使用占位逻辑，实际需要从 MONAI Hub 下载
            print("Note: Load pretrained SwinUNETR weights from MONAI Hub")
            # self.swin.load_from(weights)
        except Exception as e:
            print(f"Warning: Failed to load pretrained SwinUNETR: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwinUNETR 的 encoder 部分
        # 获取最深层的 feature map
        hidden_states = self.swin.swinViT(x, self.swin.normalize)
        # 取最后一层特征
        enc_out = hidden_states[-1]  # (B, C, D/32, H/32, W/32)
        return self.proj(enc_out)

    def get_output_dim(self) -> int:
        return self.emb_dim


# ============================================================
# Encoder Factory
# ============================================================

ENCODER_REGISTRY = {
    "toy": ToyEncoder3D,
    "resnet3d": ResNet3DEncoder,
    "swin_unetr": SwinUNETREncoder,
}


def create_encoder(
    encoder_type: str,
    in_channels: int = 1,
    emb_dim: int = 256,
    pretrained: bool = False,
    **kwargs,
) -> BaseEncoder3D:
    """创建 3D Encoder

    Args:
        encoder_type: "toy", "resnet3d", "swin_unetr"
        in_channels: 输入通道数
        emb_dim: 输出 embedding 维度
        pretrained: 是否加载预训练权重
        **kwargs: 其他参数

    Returns:
        BaseEncoder3D 实例
    """
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Available: {list(ENCODER_REGISTRY.keys())}"
        )

    encoder_cls = ENCODER_REGISTRY[encoder_type]

    if encoder_type == "toy":
        return encoder_cls(in_channels=in_channels, emb_dim=emb_dim, **kwargs)
    elif encoder_type == "resnet3d":
        return encoder_cls(
            in_channels=in_channels,
            emb_dim=emb_dim,
            pretrained_path=kwargs.get("pretrained_path") if pretrained else None,
            **{k: v for k, v in kwargs.items() if k != "pretrained_path"},
        )
    elif encoder_type == "swin_unetr":
        return encoder_cls(
            in_channels=in_channels,
            emb_dim=emb_dim,
            pretrained=pretrained,
            **kwargs,
        )
    else:
        return encoder_cls(in_channels=in_channels, emb_dim=emb_dim, **kwargs)


def list_available_encoders() -> List[str]:
    """列出所有可用的 encoder 类型"""
    return list(ENCODER_REGISTRY.keys())


# ============================================================
# 向后兼容
# ============================================================

# 保持原有的 Encoder3D 类名可用
Encoder3D = ToyEncoder3D
