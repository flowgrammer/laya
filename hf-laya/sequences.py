"""Option rendering and token packing for ``LayaProcessor``.

Layout produced per question::

    [CLS/BOS] <kind> question: <prompt> [SEP/EOS] [MASK] opt0 [MASK] opt1 ... [SEP/EOS] state [SEP/EOS]
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

QTYPES: dict[str, int] = {"choice": 0, "score": 1, "noul": 2}

_DEFAULT_NOUL_FALSE = "no, the statement does not hold"
_DEFAULT_NOUL_TRUE = "yes, the statement holds"


def serialize_state(state: str | Mapping | list) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def render_value(value: Any) -> str:
    """Render a criterion / rubric value — verbatim for strings, JSON for structured."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(question: Mapping[str, Any]) -> list[str]:
    """Turn a schema-shaped question into label-index-ordered option text."""
    kind = question["kind"]
    if kind == "choice":
        options = question["options"]
        if isinstance(options, list):
            return list(options)
        return [k if v in (None, "") else f"{k}: {render_value(v)}" for k, v in options.items()]
    if kind == "score":
        return [f"level {i}: {render_value(c)}" for i, c in enumerate(question["rubric"])]
    criteria = question.get("criteria") or {}
    false_text = render_value(criteria["false"]) if criteria.get("false") else _DEFAULT_NOUL_FALSE
    true_text = render_value(criteria["true"]) if criteria.get("true") else _DEFAULT_NOUL_TRUE
    return [f"false: {false_text}", f"true: {true_text}"]


def pack(
    tokenizer,
    state: str | Mapping | list,
    question: Mapping[str, Any],
    max_len: int,
    head_max_len: int,
) -> tuple[list[int], list[int]]:
    """Return ``(token_ids, marker_positions)`` for one question."""
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    mask_token = tokenizer.mask_token
    kind = question["kind"]

    prompt = str(question["prompt"]).replace(mask_token, " ")
    head_ids = tokenizer(f"{kind} question: {prompt}", add_special_tokens=False)["input_ids"]
    option_ids = [
        [tokenizer.mask_token_id]
        + tokenizer(f" {opt.replace(mask_token, ' ')}", add_special_tokens=False)["input_ids"][:48]
        for opt in render_options(question)
    ]

    budget = head_max_len - sum(len(o) for o in option_ids)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(option_ids)))
        option_ids = [o[:per] for o in option_ids]
        budget = head_max_len - sum(len(o) for o in option_ids)
    head_ids = head_ids[: max(8, budget)]

    ids: list[int] = [cls_id, *head_ids, sep_id]
    markers: list[int] = []
    for opts in option_ids:
        markers.append(len(ids))
        ids.extend(opts)
    ids.append(sep_id)

    room = max(0, max_len - len(ids) - 1)
    state_text = serialize_state(state).replace(mask_token, " ")
    state_ids = tokenizer(state_text, add_special_tokens=False)["input_ids"]
    ids.extend(state_ids[:room])
    ids.append(sep_id)

    ids = ids[:max_len]
    return ids, [m for m in markers if m < max_len]
