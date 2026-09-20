"""Wire-format schema for typed-decision models.

Three primitives (``choice``, ``score``, ``noul``) plus request and response envelopes.
The Python objects are also the JSON Schema for REST servers and non-Python SDKs.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

__version__ = "0.1.0"


class Immutable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChoiceQuestion(Immutable):
    kind: Literal["choice"] = "choice"
    id: str
    prompt: str
    options: Union[dict[str, str], list[str]]  # noqa: UP007  Pydantic evaluates at runtime on py3.9


class ScoreQuestion(Immutable):
    kind: Literal["score"] = "score"
    id: str
    prompt: str
    rubric: list[str] = Field(min_length=2)


class NoulQuestion(Immutable):
    kind: Literal["noul"] = "noul"
    id: str
    prompt: str
    criteria: dict[str, str] | None = None


Question = Annotated[
    Union[ChoiceQuestion, ScoreQuestion, NoulQuestion],
    Field(discriminator="kind"),
]


class ActionProbabilities(Immutable):
    """Distribution over the action head's outputs (e.g. ``{"act_probability": 0.9}``)."""

    probabilities: dict[str, float]


class ChoiceAnswer(Immutable):
    kind: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float
    action: ActionProbabilities | None = None


class ScoreAnswer(Immutable):
    kind: Literal["score"] = "score"
    score: float
    probabilities: dict[str, float]
    confidence: float
    action: ActionProbabilities | None = None
    legend: dict[str, str] | None = None


class NoulAnswer(Immutable):
    kind: Literal["noul"] = "noul"
    noul: float
    confidence: float
    action: ActionProbabilities | None = None


Answer = Annotated[
    Union[ChoiceAnswer, ScoreAnswer, NoulAnswer],
    Field(discriminator="kind"),
]


class Usage(Immutable):
    input_tokens: int = 0
    output_tokens: int = 0


class TypedDecisionRequest(Immutable):
    state: Union[str, dict, list]  # noqa: UP007  Pydantic evaluates at runtime on py3.9
    questions: list[Question] = Field(min_length=1)


class TypedDecisionResponse(Immutable):
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)
    request_id: str | None = None
    truncated: list[str] = Field(default_factory=list)
