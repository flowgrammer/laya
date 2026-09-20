"""HF-shaped implementation of Laya typed-decision models."""

from .calibration import ece_score
from .configuration_laya import LayaConfig
from .email_helpers import clean_email_body, email_state
from .lang import analyse as detect_language
from .lang import detect_script, is_english
from .modeling_laya import LayaForTypedDecision, LayaPreTrainedModel, TypedDecisionOutput
from .presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from .processing_laya import LayaProcessor
from .routing import DEFAULT_MODELS, RouteDecision, RoutedResponse, Router

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MODELS",
    "LayaConfig",
    "LayaForTypedDecision",
    "LayaPreTrainedModel",
    "LayaProcessor",
    "RouteDecision",
    "RoutedResponse",
    "Router",
    "TypedDecisionOutput",
    "clean_email_body",
    "detect_language",
    "detect_script",
    "ece_score",
    "email_questions",
    "email_state",
    "guard_questions",
    "is_english",
    "moderation_questions",
    "router_questions",
    "triage_questions",
    "__version__",
]
