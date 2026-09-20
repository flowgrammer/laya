# hf-laya

HF-shaped Laya typed-decision models. Ships the three canonical remote-code files
(`configuration_laya.py`, `modeling_laya.py`, `processing_laya.py`) plus their
supporting modules (`heads.py`, `calibration.py`, `sequences.py`) and a publisher
(`convert_laya_to_hf.py`).

```python
from transformers import AutoModel, AutoProcessor
model     = AutoModel.from_pretrained("convaiinnovations/laya",     trust_remote_code=True)
processor = AutoProcessor.from_pretrained("convaiinnovations/laya", trust_remote_code=True)
inputs    = processor(state=state, questions=questions)
outputs   = model(**inputs)
```

Or in-process (no `trust_remote_code`):

```python
from hf_laya import LayaConfig, LayaForTypedDecision, LayaProcessor
```

Publish to Convai's three Hub repos:

```bash
hf-laya-publish convaiinnovations/laya
hf-laya-publish                                    # all three
```
