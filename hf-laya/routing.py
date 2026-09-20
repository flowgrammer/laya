"""Route a request to the Laya checkpoint best suited to it.

Ported from ``laya.router``. Backed by ``AutoModel.from_pretrained(..., trust_remote_code=True)``
instead of ``laya.Agent``, and one Hub repo per checkpoint (no ``subfolder`` bundling).

Three checkpoints, measured on a shared benchmark (17,416 questions, one T4):

  english          convaiinnovations/laya                 421M  ModernBERT-large,  512 tokens
  multilingual     convaiinnovations/laya-multilingual    322M  mmBERT-base,      1024 tokens, 100+ langs
  typed-decisions  convaiinnovations/laya-typed-decisions 421M  ModernBERT-large, 1024 tokens,
                                                                fine-tuned on the typed-decisions workflows

The English checkpoint does not gently degrade off English, it collapses (0.100 on Hindi 20-way
MASSIVE intent, 0.855 ECE), so script detection is the primary routing signal. ``typed-decisions``
is never selected automatically unless ``auto_task_detection=True`` or ``task="typed-decisions"``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hf_decisions_schema import Answer, TypedDecisionResponse, Usage

DEFAULT_MODELS: dict[str, str] = {
    "english": "convaiinnovations/laya",
    "multilingual": "convaiinnovations/laya-multilingual",
    "typed-decisions": "convaiinnovations/laya-typed-decisions",
}

_ALIASES: dict[str, str] = {
    "en": "english", "laya": "english", "default": "english",
    "multi": "multilingual", "ml": "multilingual", "laya-multilingual": "multilingual",
    "typed": "typed-decisions", "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions", "decisions": "typed-decisions",
}

# Question-id signatures of the four typed-decisions workflows, used only when
# auto_task_detection is enabled.
_TYPED_DECISION_WORKFLOWS: dict[str, set[str]] = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service": {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing": {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
    "security_incidents": {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
}


class RouteDecision(dict):
    """The routing outcome: which model, why, and what was detected.

    Behaves as a ``dict`` so it serializes straight into an API response.
    """

    @property
    def model(self) -> str:
        return self["model"]

    @property
    def reason(self) -> str:
        return self["reason"]

    def __repr__(self) -> str:
        return f"RouteDecision(model={self['model']!r}, reason={self['reason']!r})"


def normalise_name(name: str) -> str:
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in DEFAULT_MODELS:
        raise ValueError(
            f"unknown model {name!r}; choose one of {sorted(DEFAULT_MODELS)} "
            f"(or an alias: {sorted(_ALIASES)})"
        )
    return key


def _question_ids(questions: Any) -> set[str]:
    """Extract question ids from either a schema-shaped list or a dict-keyed mapping."""
    if not questions:
        return set()
    if isinstance(questions, dict):
        return set(questions)
    ids: set[str] = set()
    for q in questions:
        if isinstance(q, dict) and "id" in q:
            ids.add(q["id"])
        else:
            qid = getattr(q, "id", None)
            if qid is not None:
                ids.add(qid)
    return ids


def match_typed_decisions_workflow(questions: Any) -> str | None:
    """Name of the typed-decisions workflow whose question ids these are, else ``None``.

    Requires an exact id-set match, so an unrelated schema that happens to contain 'urgency'
    is never captured.
    """
    ids = _question_ids(questions)
    if not ids:
        return None
    for workflow, signature in _TYPED_DECISION_WORKFLOWS.items():
        if ids == signature:
            return workflow
    return None


def _analyse(state: Any) -> dict[str, Any]:
    """Language/script analysis. Uses ``hf_laya.lang.analyse`` when available.

    TODO: sibling fork ports ``laya.lang`` to ``hf_laya.lang``. Until it lands, fall through
    to an "unknown script" result so callers default cleanly to ``self.default``.
    """
    try:
        from .lang import analyse
    except ImportError:
        return {
            "script": "unknown",
            "is_english": False,
            "language": None,
            "non_latin_fraction": 0.0,
        }
    return analyse(state)


@dataclass(frozen=True)
class RoutedResponse:
    """Response from ``Router.predict``: the typed answers plus the routing decision.

    Exposes ``answers`` / ``usage`` (delegating to the underlying ``TypedDecisionResponse``)
    and ``routing`` (the ``RouteDecision``) so both surfaces are one attribute-hop away::

        resp = router.predict(state, questions)
        resp.answers["department"].choice   # from TypedDecisionResponse
        resp.routing.model                  # from RouteDecision
    """

    response: TypedDecisionResponse
    routing: RouteDecision

    @property
    def answers(self) -> dict[str, Answer]:
        return self.response.answers

    @property
    def usage(self) -> Usage:
        return self.response.usage


class _Checkpoint:
    """One loaded ``(model, processor)`` pair — the HF-native replacement for ``laya.Agent``."""

    def __init__(
        self,
        repo: str,
        device: str | None = None,
        token: str | None = None,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        self.repo = repo
        self.model = AutoModel.from_pretrained(repo, trust_remote_code=True, token=token)
        self.processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True, token=token)
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()

    def predict(self, state: Any, questions: list[dict]) -> Any:
        import torch
        from hf_decisions.pipeline import _shape_answer
        from hf_decisions_schema import TypedDecisionResponse

        inputs = self.processor(state=state, questions=questions, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        tensors = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**tensors)
        probs = output.probabilities.detach().cpu().numpy()
        mask = output.marker_mask.detach().cpu().numpy()
        answers = {
            q["id"]: _shape_answer(q, probs[i], int(mask[i].sum()))
            for i, q in enumerate(questions)
        }
        return TypedDecisionResponse.model_validate({"answers": answers})


class Router:
    """Lazily load Laya checkpoints and dispatch each request to the right one.

        from hf_laya import Router
        r = Router(preload=True)
        r.predict("Mein Konto wurde zweimal belastet", questions).routing.model  # -> multilingual
        r.predict("I was charged twice", questions).routing.model                # -> english
        r.predict(state, questions, model="typed-decisions").routing.model       # -> typed-decisions

    Checkpoints are downloaded and built on first use. ``max_loaded`` caps how many stay
    resident; least-recently-used is evicted, because all three together are ~1.16B parameters.
    For a server or a demo, ``preload=True`` keeps every checkpoint resident so routing is free.
    """

    def __init__(
        self,
        models: dict[str, str] | None = None,
        device: str | None = None,
        token: str | None = None,
        max_loaded: int = 1,
        default: str = "english",
        auto_task_detection: bool = False,
        preload: bool = False,
    ) -> None:
        self.models: dict[str, str] = dict(DEFAULT_MODELS)
        if models:
            self.models.update({normalise_name(k): v for k, v in models.items()})
        self.device = device
        self.token = token or os.environ.get("HF_TOKEN")
        self.max_loaded = max(1, int(max_loaded))
        self.default = normalise_name(default)
        self.auto_task_detection = bool(auto_task_detection)
        self._checkpoints: dict[str, _Checkpoint] = {}
        self._order: list[str] = []  # least-recently-used first
        if preload:
            self.preload()

    # -------------------------------------------------------------------- loading
    def load(self, name: str) -> _Checkpoint:
        """Return the checkpoint for ``name``, downloading and building it on first use."""
        key = normalise_name(name)
        if key in self._checkpoints:
            self._touch(key)
            return self._checkpoints[key]
        checkpoint = _Checkpoint(self.models[key], device=self.device, token=self.token)
        self._checkpoints[key] = checkpoint
        self._order.append(key)
        self._evict()
        return checkpoint

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def _evict(self) -> None:
        while len(self._order) > self.max_loaded:
            victim = self._order.pop(0)
            self._checkpoints.pop(victim, None)
        # keep the two views consistent
        for k in list(self._checkpoints):
            if k not in self._order:
                self._checkpoints.pop(k, None)

    def attach(self, name: str, checkpoint: _Checkpoint) -> _Checkpoint:
        """Register an already-built checkpoint under ``name`` instead of loading a second copy."""
        key = normalise_name(name)
        self._checkpoints[key] = checkpoint
        self._touch(key)
        self.max_loaded = max(self.max_loaded, len(self._checkpoints))
        return checkpoint

    def preload(self, names: list[str] | None = None) -> Router:
        """Download and build checkpoints up front so no request pays a cold-load."""
        resolved = [normalise_name(n) for n in (names or list(self.models))]
        self.max_loaded = max(self.max_loaded, len(resolved), len(self._checkpoints))
        for n in resolved:
            if n not in self._checkpoints:
                self.load(n)
        return self

    def unload(self, name: str | None = None) -> None:
        """Free one checkpoint, or all of them."""
        if name is None:
            self._checkpoints.clear()
            self._order.clear()
            return
        key = normalise_name(name)
        self._checkpoints.pop(key, None)
        if key in self._order:
            self._order.remove(key)

    @property
    def loaded(self) -> list[str]:
        return list(self._order)

    # -------------------------------------------------------------------- routing
    def route(
        self,
        state: str | dict | list | None,
        questions: Any = None,
        model: str | None = None,
        task: str | None = None,
        lang: str | None = None,
        hint: str | None = None,
    ) -> RouteDecision:
        """Decide which checkpoint to use, without loading or running anything.

        Precedence: explicit ``model`` > explicit ``task`` > detected workflow (opt-in) >
        explicit ``lang`` (or ``hint``) > detected script/language > ``self.default``.
        """
        if model is not None:
            key = normalise_name(model)
            return RouteDecision(
                model=key, repo=self.models[key],
                reason=f"explicit model={model!r}",
                detection=None, workflow=None,
            )

        if task is not None:
            normalized = str(task).lower().replace("-", "_")
            resolved = "typed-decisions" if normalized == "typed_decisions" else task
            key = normalise_name(resolved)
            return RouteDecision(
                model=key, repo=self.models[key],
                reason=f"explicit task={task!r}",
                detection=None, workflow=None,
            )

        workflow = match_typed_decisions_workflow(questions)
        if workflow and self.auto_task_detection:
            return RouteDecision(
                model="typed-decisions", repo=self.models["typed-decisions"],
                reason=f"question ids match the {workflow!r} typed-decisions workflow",
                detection=None, workflow=workflow,
            )

        effective_lang = lang if lang is not None else hint
        if effective_lang is not None:
            head = str(effective_lang).lower().split("-")[0]
            key = "english" if head in ("en", "eng", "english") else "multilingual"
            source = "lang" if lang is not None else "hint"
            return RouteDecision(
                model=key, repo=self.models[key],
                reason=f"explicit {source}={effective_lang!r}",
                detection=None, workflow=workflow,
            )

        detection = _analyse(state)
        if detection["script"] == "unknown":
            key = self.default
            reason = f"no letters detected in state; using default ({key})"
        elif detection["script"] != "latin":
            key = "multilingual"
            non_latin_pct = 100 * float(detection["non_latin_fraction"])
            reason = (
                f"non-Latin script ({detection['script']}, {non_latin_pct:.0f}% of letters); "
                f"the English checkpoint cannot read it"
            )
        elif not detection["is_english"]:
            key = "multilingual"
            reason = f"Latin script but language looks like {detection['language']!r}, not English"
        else:
            key = "english"
            reason = "English Latin text"
        return RouteDecision(
            model=key, repo=self.models[key], reason=reason,
            detection=detection, workflow=workflow,
        )

    # -------------------------------------------------------------------- running
    def predict(
        self,
        state: str | dict | list,
        questions: list[dict],
        model: str | None = None,
        task: str | None = None,
        lang: str | None = None,
        hint: str | None = None,
    ) -> RoutedResponse:
        """Route, then answer every question in one forward pass on the chosen checkpoint.

        Returns a ``RoutedResponse`` exposing ``.answers`` / ``.usage`` (from the underlying
        ``TypedDecisionResponse``) and ``.routing`` (the ``RouteDecision`` that picked the
        checkpoint and explains why).
        """
        decision = self.route(state, questions, model=model, task=task, lang=lang, hint=hint)
        checkpoint = self.load(decision["model"])
        response = checkpoint.predict(state, questions)
        return RoutedResponse(response=response, routing=decision)

    def __repr__(self) -> str:
        return (
            f"Router(loaded={self.loaded}, max_loaded={self.max_loaded}, "
            f"default={self.default!r})"
        )
