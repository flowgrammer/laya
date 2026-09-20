"""``LayaForTypedDecision`` — encoder + typed decision head."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.utils import ModelOutput

from .calibration import calibrated_softmax
from .configuration_laya import LayaConfig
from .heads import act_features, make_act_head, make_head, make_scorer


@dataclass
class TypedDecisionOutput(ModelOutput):
    """Model forward output.

    ``probabilities`` and ``marker_mask`` are always populated.
    ``act_probabilities`` is populated only when ``config.n_act > 0``.
    """

    probabilities: torch.FloatTensor | None = None
    marker_mask: torch.BoolTensor | None = None
    act_probabilities: torch.FloatTensor | None = None


class LayaPreTrainedModel(PreTrainedModel):
    config_class = LayaConfig
    base_model_prefix = "encoder"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _no_split_modules: list[str] = []
    main_input_name = "input_ids"


class LayaForTypedDecision(LayaPreTrainedModel):
    """Bidirectional encoder + typed decision head.

    Submodule names deliberately match Convai's existing checkpoint layout
    (``encoder``, ``head``, ``type_emb``, ``scorer``, ``act_head``) so
    ``load_state_dict(..., strict=True)`` succeeds on the current ``model.safetensors``
    files hosted in ``convaiinnovations/laya*`` on the Hub.
    """

    def __init__(self, config: LayaConfig) -> None:
        super().__init__(config)
        hidden = config.encoder_config.hidden_size
        self.encoder = AutoModel.from_config(config.encoder_config)
        # ModernBERT's `reference_compile` defaults to "auto" and will torch.compile the
        # encoder. That is a loss for the batch sizes Laya runs (a handful of questions
        # per call) and can hang on some platforms — keep the eager path.
        try:
            self.encoder.config.reference_compile = False
        except AttributeError:
            pass

        self.head = make_head(hidden, config.head_layers)
        self.type_emb = nn.Embedding(3, hidden)
        self.scorer = make_scorer(hidden)
        self.act_head = make_act_head(hidden, config.n_act) if config.n_act > 0 else None

        self.register_buffer("temperature", torch.tensor(config.temperature))
        self._temperature_by_options = dict(config.temperature_by_options)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> TypedDecisionOutput:
        try:
            return self._run(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            device = input_ids.device
            if device.type == "cpu" or not _is_recoverable(exc):
                raise
            self._fallback_to_cpu(device, exc)
            cpu = torch.device("cpu")
            return self._run(
                input_ids.to(cpu),
                attention_mask.to(cpu),
                marker_pos.to(cpu),
                marker_mask.to(cpu),
                qtype.to(cpu),
            )

    # ------------------------------------------------------------------ internal

    def _run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> TypedDecisionOutput:
        device_type = input_ids.device.type
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=device_type == "cuda"):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            hidden = self.head(hidden, src_key_padding_mask=~attention_mask.bool())

        gather = marker_pos.clamp_min(0)[..., None].expand(-1, -1, hidden.size(-1))
        marker_hidden = torch.gather(hidden, 1, gather)
        logits = self.scorer(marker_hidden).squeeze(-1).float().masked_fill(~marker_mask, -1e4)

        probs = calibrated_softmax(logits, marker_mask, qtype, self.temperature, self._temperature_by_options)

        act_probs = None
        if self.act_head is not None:
            pooled = hidden[:, 0].float()
            features = act_features(logits, marker_mask)
            act_logits = self.act_head(torch.cat([pooled, features], dim=-1))
            act_probs = torch.softmax(act_logits, dim=-1)

        return TypedDecisionOutput(probabilities=probs, marker_mask=marker_mask, act_probabilities=act_probs)

    def _fallback_to_cpu(self, original_device: torch.device, reason: BaseException) -> None:
        self.to("cpu")
        warnings.warn(
            f"\n[hf_laya] Could not run the model on {original_device}, so it is running on CPU.\n"
            f"  Reason: {reason}\n"
            "  Inference will be roughly 10-15x slower (~200-500 ms rather than ~35 ms).\n"
            "  If this is a newer NVIDIA GPU (Blackwell / RTX 50-series), your PyTorch build\n"
            "  may not support its CUDA architecture:\n"
            "    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128\n"
            "  See https://pytorch.org/get-started/locally/",
            stacklevel=2,
        )


def _is_recoverable(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "memory" in message or "cuda" in message


def fix_tokenizer_config(model_dir: str | Path) -> None:
    """Normalize ``tokenizer_config.json`` for cross-version transformers compatibility.

    ModernBERT/mmBERT checkpoints saved by older stacks store ``tokenizer_class`` as
    ``"TokenizersBackend"`` and ``extra_special_tokens`` as a list — transformers expects
    ``"PreTrainedTokenizerFast"`` and a mapping. Silently no-op if the file is absent or
    malformed; this is a best-effort compatibility shim.
    """
    root = Path(model_dir)
    cfg_file = root / "tokenizer" / "tokenizer_config.json"
    if not cfg_file.exists():
        cfg_file = root / "tokenizer_config.json"
    if not cfg_file.exists():
        return
    try:
        tcfg = json.loads(cfg_file.read_text())
    except (OSError, json.JSONDecodeError):
        return

    changed = False
    if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
        tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        tcfg.pop("backend", None)
        tcfg.pop("is_local", None)
        changed = True

    extra = tcfg.get("extra_special_tokens")
    if isinstance(extra, list):
        tcfg["extra_special_tokens"] = {f"extra_{i}": t for i, t in enumerate(extra)}
        changed = True

    if changed:
        try:
            cfg_file.write_text(json.dumps(tcfg, indent=2))
        except OSError:
            pass
