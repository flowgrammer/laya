"""``TypedDecisionPipeline`` — the ``pipeline("typed-decision")`` implementation.

Fits the standard HF three-stage split. The model returns calibrated probabilities in
``TypedDecisionOutput.probabilities``; ``postprocess`` shapes them into the schema and,
when the model also returns ``act_probabilities``, attaches per-answer ``action`` fields.
``usage.input_tokens`` comes from the attention mask sum stashed on the pipeline instance
during preprocess (the three-stage split can't thread scalars through cleanly).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from hf_decisions_schema import Question, TypedDecisionRequest, TypedDecisionResponse
from transformers.pipelines.base import Pipeline


class TypedDecisionPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs: Any) -> tuple[dict, dict, dict]:
        preprocess: dict[str, Any] = {}
        postprocess: dict[str, Any] = {}
        if "questions" in kwargs:
            preprocess["questions"] = kwargs["questions"]
            postprocess["questions"] = kwargs["questions"]
        return preprocess, {}, postprocess

    def __call__(
        self,
        state: str | dict | list,
        questions: list[Question] | list[dict],
        **kwargs: Any,
    ) -> TypedDecisionResponse:
        request = TypedDecisionRequest(state=state, questions=questions)
        return super().__call__(
            request.state,
            questions=[q.model_dump() for q in request.questions],
            **kwargs,
        )

    def preprocess(self, state: Any, questions: list[dict] | None = None) -> dict[str, Any]:
        inputs = self.processor(state=state, questions=questions or [], return_tensors="pt")
        # Stash the input-token count on the instance for postprocess. The three-stage
        # pipeline API can't thread arbitrary scalars from preprocess into postprocess,
        # and rewrapping the model output would force a modeling_laya.py change.
        attention_mask = inputs.get("attention_mask")
        self._last_input_tokens = int(attention_mask.sum().item()) if attention_mask is not None else 0
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _forward(self, model_inputs: dict[str, Any]) -> Any:
        return self.model(**model_inputs)

    def postprocess(self, model_outputs: Any, questions: list[dict] | None = None) -> TypedDecisionResponse:
        probabilities = model_outputs.probabilities.detach().cpu().numpy()
        marker_mask = model_outputs.marker_mask.detach().cpu().numpy()
        act_probs = model_outputs.act_probabilities
        if act_probs is not None:
            act_probs = act_probs.detach().cpu().numpy()

        answers: dict[str, dict[str, Any]] = {}
        for i, q in enumerate(questions or []):
            k = int(marker_mask[i].sum())
            row_action = act_probs[i] if act_probs is not None else None
            answers[q["id"]] = _shape_answer(q, probabilities[i], k, row_action)

        return TypedDecisionResponse.model_validate({
            "answers": answers,
            "usage": {"input_tokens": getattr(self, "_last_input_tokens", 0), "output_tokens": 0},
        })


def _shape_answer(
    question: dict,
    probabilities: np.ndarray,
    k: int,
    action: np.ndarray | None = None,
) -> dict[str, Any]:
    p = probabilities[:k]
    confidence = round(_confidence(p), 4)
    kind = question["kind"]
    action_field = _shape_action(action)

    if kind == "choice":
        options = question["options"]
        keys = list(options.keys()) if isinstance(options, dict) else list(options)
        answer: dict[str, Any] = {
            "kind": "choice",
            "choice": keys[int(p.argmax())],
            "probabilities": {label: round(float(prob), 4) for label, prob in zip(keys, p)},
            "confidence": confidence,
        }
        if action_field is not None:
            answer["action"] = action_field
        return answer

    if kind == "score":
        rubric = question.get("rubric") or []
        answer = {
            "kind": "score",
            "score": round(float((np.arange(k) * p).sum()), 4),
            "probabilities": {str(i): round(float(prob), 4) for i, prob in enumerate(p)},
            "confidence": confidence,
            "legend": {str(i): rubric[i] for i in range(min(k, len(rubric)))},
        }
        if action_field is not None:
            answer["action"] = action_field
        return answer

    p_true = float(p[1])
    answer = {"kind": "noul", "noul": round(p_true, 4), "confidence": round(max(p_true, 1.0 - p_true), 4)}
    if action_field is not None:
        answer["action"] = action_field
    return answer


def _shape_action(action: np.ndarray | None) -> dict[str, Any] | None:
    """Upstream reports ``{"act_probability": act_probs[0]}``; we preserve that key."""
    if action is None:
        return None
    return {"probabilities": {"act_probability": round(float(action[0]), 4)}}


def _confidence(probabilities: np.ndarray) -> float:
    k = len(probabilities)
    if k < 2:
        return 1.0
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - entropy / np.log(k), 0.0, 1.0))
