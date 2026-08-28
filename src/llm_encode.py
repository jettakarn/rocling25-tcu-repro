from __future__ import annotations

"""Shared LLM encoder helpers for FP16 full-corpus embedding (paper path)."""

MODEL_PRESETS = {
    "deepseek": {
        "repo": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "notes": "Paper Table 1/3 primary encoder.",
        "force_qwen2_tokenizer": True,
    },
    "prover": {
        "repo": "deepseek-ai/DeepSeek-Prover-V1.5-RL",
        "notes": (
            "Paper Table 3 DeepSeek-Prover-V1.5-RL. "
            "tokenizer.json is ByteLevel BPE; use PreTrainedTokenizerFast."
        ),
        "force_qwen2_tokenizer": False,
        "bytelevel_fast_tokenizer": True,
    },
    "taide": {
        "repo": "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",
        "notes": "Paper Table 3 Llama3-TAIDE.",
        "force_qwen2_tokenizer": False,
    },
}


def load_llm_tokenizer(
    repo: str,
    *,
    force_qwen2: bool = False,
    bytelevel_fast: bool = False,
):
    """Load tokenizer; DeepSeek-R1-Qwen3 needs Qwen2 BPE, not LlamaTokenizerFast.

    DeepSeek-Prover ships ByteLevel BPE in tokenizer.json but advertises LlamaTokenizer;
    use PreTrainedTokenizerFast when bytelevel_fast=True.
    """
    from transformers import AutoTokenizer

    if force_qwen2:
        try:
            tok = AutoTokenizer.from_pretrained(
                repo, trust_remote_code=True, tokenizer_type="qwen2"
            )
        except TypeError:
            from transformers import Qwen2TokenizerFast

            tok = Qwen2TokenizerFast.from_pretrained(repo, trust_remote_code=True)
    elif bytelevel_fast:
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast.from_pretrained(repo, trust_remote_code=True)
    else:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def mean_pool(hidden, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
