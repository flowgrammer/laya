"""Config for ``LayaForTypedDecision``."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


class LayaConfig(PretrainedConfig):
    """Nests a real ``PretrainedConfig`` for the encoder backbone.

    ``n_act`` defaults to ``len(act_costs) + 1`` (mirroring upstream's
    ``build_model``), so every Convai checkpoint — which ships with a trained
    escalate head — loads with ``strict=True`` out of the box.
    """

    model_type = "laya"

    def __init__(
        self,
        encoder_config: dict | PretrainedConfig | None = None,
        head_layers: int = 2,
        n_act: int | None = None,
        act_costs: dict[str, float] | None = None,
        temperature: tuple[float, float, float] = (1.0, 1.0, 1.0),
        temperature_by_options: dict[str, float] | None = None,
        head_max_len: int = 192,
        max_len: int = 512,
        max_prefixes: int = 6,
        cost_wrong_act: float = 3.0,
        amp_dtype: str = "fp16",
        library_tag: str = "encoder-head",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder_config = _coerce_encoder_config(encoder_config)
        self.head_layers = head_layers
        self.act_costs = act_costs or {}
        self.n_act = n_act if n_act is not None else len(self.act_costs) + 1
        self.temperature = list(temperature)
        self.temperature_by_options = temperature_by_options or {}
        self.head_max_len = head_max_len
        self.max_len = max_len
        self.max_prefixes = max_prefixes
        self.cost_wrong_act = cost_wrong_act
        self.amp_dtype = amp_dtype
        self.library_tag = library_tag


def _coerce_encoder_config(value: dict | PretrainedConfig | None) -> PretrainedConfig:
    if isinstance(value, PretrainedConfig):
        return value
    if isinstance(value, dict) and value:
        payload = dict(value)
        model_type = payload.pop("model_type", "modernbert")
        return CONFIG_MAPPING[model_type](**payload)
    return AutoConfig.for_model("modernbert")
