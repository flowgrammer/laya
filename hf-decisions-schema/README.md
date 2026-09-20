# hf-decisions-schema

Wire-format schema for typed-decision models. Three primitives: `choice`, `score`, `noul`.

```python
from hf_decisions_schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, TypedDecisionRequest

req = TypedDecisionRequest(
    state="Package hasn't arrived after 12 days, tracking last updated Sept 8.",
    questions=[
        ChoiceQuestion(id="department", prompt="Which team owns this?", options={
            "refunds": "money back or store credit",
            "shipping": "carrier trace or reship",
            "escalations": "manager involvement",
        }),
        ScoreQuestion(id="urgency", prompt="How urgent?", rubric=["can wait", "this week", "same day"]),
        NoulQuestion(id="needs_manager", prompt="Does this need a manager?"),
    ],
)
req.model_dump_json()   # ready to POST /v1/decide
```

Not covered: Qwen-RLCD structured-JSON-extraction and cua-s1 form-actions — adjacent schemas.
