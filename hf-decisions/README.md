# hf-decisions

Registers `pipeline("typed-decision")` with HuggingFace transformers, so any Hub model that
exposes an `AutoProcessor` and a `LayaForTypedDecision`-shaped forward can be called through
the standard pipeline API today — without waiting for a transformers-core PR.

```python
import hf_decisions                             # side effect: registers the task
from transformers import pipeline

model = pipeline("typed-decision",
                 model="convaiinnovations/laya",
                 trust_remote_code=True)
resp  = model(state="Package hasn't arrived after 12 days.",
              questions=[...])                  # hf_decisions_schema.Question
```

Follows the standard three-stage `preprocess` → `_forward` → `postprocess` split:
preprocess runs the processor to build tensor inputs, forward runs the model, postprocess
turns the model's calibrated probabilities into a `hf_decisions_schema.TypedDecisionResponse`.

Idempotent: if `transformers` already registers `typed-decision` natively, importing this
package is a no-op. Safe to leave the import in place after upstream support lands.
