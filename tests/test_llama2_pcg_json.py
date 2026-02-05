from __future__ import annotations

from provetok.pcg.llama2_pcg import Llama2PCGConfig, parse_llm_json, sanitize_generation_dict


def test_parse_llm_json_extracts_first_object() -> None:
    text = 'preamble\\n{"a": 1}\\n{"b": 2}'
    d = parse_llm_json(text)
    assert d == {"a": 1}


def test_sanitize_generation_dict_multi_frame_and_constraints() -> None:
    text = """```json
{
  "frames": [
    {
      "finding": "nodule",
      "polarity": "present",
      "laterality": "left",
      "confidence": 0.9,
      "location": "unspecified",
      "size_bin": "unspecified",
      "severity": "unspecified",
      "uncertain": false
    },
    {
      "finding": "effusion",
      "polarity": "present",
      "laterality": "right",
      "confidence": 0.1,
      "location": "unspecified",
      "size_bin": "unspecified",
      "severity": "unspecified",
      "uncertain": false
    },
    {
      "finding": "atelectasis",
      "polarity": "present",
      "laterality": "bilateral",
      "confidence": 0.5,
      "location": "unspecified",
      "size_bin": "unspecified",
      "severity": "unspecified",
      "uncertain": false
    }
  ],
  "citations": {
    "0": [0, 1, 99],
    "1": [2, 3, 4]
  },
  "q": {
    "0": 0.9,
    "1": 0.2
  }
}
```"""

    d = parse_llm_json(text)
    cfg = Llama2PCGConfig(
        model_path="dummy",
        max_frames=2,
        topk_citations=2,
        tau_refuse=0.55,
    )
    gen = sanitize_generation_dict(d, token_ids=[0, 1, 2, 3], cfg=cfg)

    assert len(gen.frames) == 2  # max_frames enforced
    assert gen.citations[0] == [0, 1]  # filtered + topk enforced
    assert gen.citations[1] == [2, 3]  # topk enforced

    # refusal is derived from tau_refuse when missing
    assert gen.refusal[0] is False
    assert gen.refusal[1] is True

