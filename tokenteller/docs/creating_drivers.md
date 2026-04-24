# Driver Guide

Tokenteller has three kinds of drivers:

- model drivers
- dataset drivers
- test drivers

The two most useful example files are:

- model example: [src/tokenteller/drivers/models/sentencepiece.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/models/sentencepiece.py)
- dataset example: [src/tokenteller/drivers/datasets/common_crawl.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/datasets/common_crawl.py)

## Model Driver

Start from [src/tokenteller/drivers/models/base.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/models/base.py)
or copy [src/tokenteller/drivers/models/template.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/models/template.py).

You only have to implement:

- `encode(text)`

You can also implement:

- `decode(token_ids)`

Put any setup in `__init__()`. This is where you should load the real tokenizer.

Example constructor inputs:

- model path
- checkpoint name
- tokenizer options

`encode()` must return a `TokenizationResult` with:

- `token_ids`: list of token ids
- `tokens`: list of readable token strings
- `token_count`: usually `len(token_ids)`
- `offsets`: token spans or `None`
- `raw`: any extra library output you want to keep

Small example:

```python
from tokenteller.core.types import TokenizationResult
from tokenteller.drivers.models.base import BaseModelDriver


class MyModelDriver(BaseModelDriver):
    def __init__(self, model_path: str, name: str = "my_model"):
        super().__init__(name=name)
        self.model = load_my_tokenizer(model_path)

    def encode(self, text: str) -> TokenizationResult:
        token_ids = self.model.encode(text)
        tokens = self.model.tokens(text)
        return TokenizationResult(
            token_ids=token_ids,
            tokens=tokens,
            token_count=len(token_ids),
            offsets=None,
            raw={},
        )
```

## Dataset Driver

Start from [src/tokenteller/drivers/datasets/base.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/datasets/base.py)
or copy [src/tokenteller/drivers/datasets/template.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/datasets/template.py).

You only have to implement:

- `iter_records(query)`

Put any setup in `__init__()`. This is where you should load the dataset or open the data source.

Example constructor inputs:

- file path
- split name
- API settings

`iter_records(query)` must yield `DatasetRecord` objects with:

- `id`: stable record id
- `text`: the text to tokenize
- `categories`: simple labels like language or domain
- `metadata`: any extra fields you want to keep

Small example:

```python
from collections.abc import Iterable

from tokenteller.core.types import DatasetQuery, DatasetRecord
from tokenteller.drivers.datasets.base import BaseDatasetDriver


class MyDatasetDriver(BaseDatasetDriver):
    def __init__(self, rows: list[dict], name: str = "my_dataset"):
        super().__init__(name=name)
        self.rows = rows

    def iter_records(self, query: DatasetQuery) -> Iterable[DatasetRecord]:
        rows = self.rows
        if query.limit is not None:
            rows = rows[: query.limit]

        for index, row in enumerate(rows, start=1):
            yield DatasetRecord(
                id=str(index),
                text=row["text"],
                categories={"language": row.get("language", "")},
                metadata=row,
            )
```

## Test Driver

Start from [src/tokenteller/testsuites/base.py](D:/Development/School/asml/tokenteller/src/tokenteller/testsuites/base.py)
or copy [src/tokenteller/testsuites/template.py](D:/Development/School/asml/tokenteller/src/tokenteller/testsuites/template.py).

You must implement:

- `name()`
- `run_case(tokenizer, record, context)`

`run_case(...)` must return a `TestCaseResult`.

Useful fields:

- `metrics`: numbers or summary values
- `artifacts`: extra detail like tokens or spans

The base class already handles:

- storing status
- storing results
- building summary rows
- printing a readable summary
- comparing two finished tests
- running records in parallel

## Constructor Parameters

The experiment does not build drivers for you.
You build the objects yourself, so your constructors can take whatever parameters you need.

Example:

```python
model = MyModelDriver(model_path="my.model")
dataset = MyDatasetDriver(data_path="data.jsonl")
test = MyTestDriver(label="Hindi prompts")
```

## Minimal Workflow

1. Copy a template.
2. Fill in the required method.
3. Create the driver object directly.
4. Add it to an `Experiment`.
5. Run the experiment.

Example:

```python
from tokenteller import Experiment
from tokenteller.core.types import DatasetQuery
from tokenteller.testsuites.metrics import TokenCountTest

experiment = Experiment()
experiment.add_model(MyModelDriver(model_path="my.model"), name="my_model")
experiment.add_dataset(MyDatasetDriver(data_path="data.jsonl"), name="my_dataset")
experiment.add_test(
    TokenCountTest(label="english sample"),
    model="my_model",
    dataset="my_dataset",
    query=DatasetQuery(limit=25),
)

report = experiment.run()
```

## Keep It Small

For this project, keep drivers simple:

- one file per model
- one file per dataset
- load things in `__init__()`
- do the real work in one required method
- add helper methods only if you actually need them
