"""Register the ``typed-decision`` pipeline task at import time.

Idempotent: if ``transformers`` already knows the task, this package is a no-op — the
``import hf_decisions`` line is safe to keep once upstream support lands.
"""

from __future__ import annotations

from transformers import AutoModel
from transformers.pipelines import PIPELINE_REGISTRY, SUPPORTED_TASKS

from .pipeline import TypedDecisionPipeline

__version__ = "0.1.0"
TASK = "typed-decision"

if TASK not in SUPPORTED_TASKS:
    PIPELINE_REGISTRY.register_pipeline(
        TASK,
        pipeline_class=TypedDecisionPipeline,
        pt_model=AutoModel,
        default={"model": {"pt": ("convaiinnovations/laya", "main")}},
        type="text",
    )
