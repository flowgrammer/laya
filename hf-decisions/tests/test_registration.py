from __future__ import annotations

import numpy as np


def test_import_registers_task() -> None:
    import hf_decisions
    from transformers.pipelines import SUPPORTED_TASKS

    assert hf_decisions.TASK in SUPPORTED_TASKS


def test_shape_answer_choice() -> None:
    from hf_decisions.pipeline import _shape_answer

    question = {"kind": "choice", "id": "dept", "prompt": "?", "options": {"a": "…", "b": "…", "c": "…"}}
    ans = _shape_answer(question, np.array([0.1, 0.7, 0.2]), 3)
    assert ans["kind"] == "choice"
    assert ans["choice"] == "b"
    assert ans["probabilities"] == {"a": 0.1, "b": 0.7, "c": 0.2}


def test_shape_answer_score_expectation() -> None:
    from hf_decisions.pipeline import _shape_answer

    question = {"kind": "score", "id": "urg", "prompt": "?", "rubric": ["low", "med", "high"]}
    ans = _shape_answer(question, np.array([0.2, 0.5, 0.3]), 3)
    assert ans["score"] == round(0.2 * 0 + 0.5 * 1 + 0.3 * 2, 4)


def test_shape_answer_noul() -> None:
    from hf_decisions.pipeline import _shape_answer

    question = {"kind": "noul", "id": "mgr", "prompt": "?"}
    ans = _shape_answer(question, np.array([0.28, 0.72]), 2)
    assert ans["noul"] == 0.72
    assert ans["confidence"] == 0.72
