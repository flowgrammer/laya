"""Smoke: the public surface imports and the package data is present."""

from __future__ import annotations

from importlib import resources


def test_public_api_imports() -> None:
    from hf_laya import (
        LayaConfig,
        LayaForTypedDecision,
        LayaPreTrainedModel,
        LayaProcessor,
        TypedDecisionOutput,
        __version__,
    )

    assert __version__
    assert LayaConfig.model_type == "laya"
    assert LayaPreTrainedModel.config_class is LayaConfig
    assert issubclass(LayaForTypedDecision, LayaPreTrainedModel)
    assert LayaProcessor.tokenizer_class == "AutoTokenizer"
    fields = TypedDecisionOutput.__dataclass_fields__.keys()
    assert fields >= {"probabilities", "marker_mask", "act_probabilities"}


def test_configs_are_package_data() -> None:
    """The three per-Hub-repo config.json files must be reachable via importlib.resources."""
    files = resources.files("hf_laya").joinpath("configs")
    names = {p.name for p in files.iterdir() if p.name.endswith(".json")}
    assert names == {"laya.json", "laya-multilingual.json", "laya-typed-decisions.json"}


def test_publisher_targets_align_with_package_configs() -> None:
    from hf_laya.convert_laya_to_hf import HUB_REPOS

    files = resources.files("hf_laya").joinpath("configs")
    package_configs = {p.name for p in files.iterdir() if p.name.endswith(".json")}
    assert set(HUB_REPOS.values()) == package_configs
