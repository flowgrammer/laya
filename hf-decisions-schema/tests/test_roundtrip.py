from __future__ import annotations

import pytest
from hf_decisions_schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    TypedDecisionRequest,
    TypedDecisionResponse,
)
from pydantic import ValidationError


def _request() -> TypedDecisionRequest:
    return TypedDecisionRequest(
        state="ticket text",
        questions=[
            ChoiceQuestion(id="dept", prompt="which team?", options={"a": "…", "b": "…"}),
            ChoiceQuestion(id="dept_list", prompt="which team?", options=["a", "b"]),
            ScoreQuestion(id="urg", prompt="how urgent?", rubric=["low", "med", "high"]),
            NoulQuestion(id="mgr", prompt="needs manager?"),
        ],
    )


def test_request_roundtrips_through_json() -> None:
    req = _request()
    assert TypedDecisionRequest.model_validate_json(req.model_dump_json()) == req


def test_options_accepts_dict_and_list() -> None:
    dict_q = _request().questions[0]
    list_q = _request().questions[1]
    assert isinstance(dict_q.options, dict) and isinstance(list_q.options, list)


def test_noul_criteria_optional() -> None:
    NoulQuestion(id="x", prompt="p")
    NoulQuestion(id="x", prompt="p", criteria={"true": "yes", "false": "no"})


def test_score_requires_two_levels() -> None:
    with pytest.raises(ValidationError):
        ScoreQuestion(id="s", prompt="p", rubric=["only-one"])


def test_response_structure() -> None:
    payload = {
        "answers": {
            "mgr": {"kind": "noul", "noul": 0.72, "confidence": 0.72},
        },
        "request_id": "req_abc",
        "truncated": [],
        "usage": {"input_tokens": 128, "output_tokens": 0},
    }
    resp = TypedDecisionResponse.model_validate(payload)
    assert resp.answers["mgr"].noul == 0.72
