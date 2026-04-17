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
- `run(context)`

Inside `run(...)`, store your own:

- `results`
- `summary`
- `warnings`

## Constructor Parameters

The experiment does not build drivers for you.
You build the objects yourself, so your constructors can take whatever parameters you need.

Example:

```python
model = MyModelDriver(model_path="my.model")
test = MyTestDriver(model=model, label="Hindi prompts")
```

## Minimal Workflow

1. Copy a template.
2. Create the dataset driver and query inside the test.
3. Fill in the required methods.
4. Add the test to an `Experiment`.
5. Run the experiment.

Example:

```python
from tokenteller import Experiment
from tokenteller.core.types import DatasetQuery, DatasetRecord, TestCaseResult, TestContext
from tokenteller.testsuites.base import BaseTestDriver


class EnglishTokenCountTest(BaseTestDriver):
    def __init__(self, model, label: str | None = None):
        super().__init__(model=model, label=label)
        self.dataset = MyDatasetDriver()
        self.query = DatasetQuery(limit=25)

    def name(self) -> str:
        return "token_count"

    def run(self, context: TestContext) -> None:
        records = list(self.dataset.iter_records(self.query))
        for record in records:
            tokenization = context.get_tokenization(self.model, record)
            self.results.append(
                TestCaseResult(
                    record_id=record.id,
                    tokenizer_name=self.model.name,
                    test_name=self.name(),
                    metrics={"token_count": tokenization.token_count},
                    artifacts={},
                )
            )


experiment = Experiment()
experiment.add_test(EnglishTokenCountTest(MyModelDriver(model_path="my.model"), label="english sample"))

report = experiment.run()
```

## Keep It Small

For this project, keep drivers simple:

- one file per model
- one file per dataset
- load things in `__init__()`
- do the real work in `run()`
- add helper methods only if you actually need them

For remote datasets such as Common Crawl, keep the query object generic inside
the test. It is better to reuse `DatasetQuery(filters=...)` and let the driver
translate a few plain keys like `domain`, `url`, or `status` than to create a
separate query class just for one backend.

## Query Shape

Keep dataset queries simple and consistent across drivers. `DatasetQuery` has
four fields:

- `filters`: a dictionary of plain filter values
- `limit`: max records to return
- `sample_strategy`: `head`, `tail`, or `random`
- `seed`: optional random seed

Example:

```python
query = DatasetQuery(
    filters={"language": "en"},
    limit=25,
    sample_strategy="random",
    seed=7,
)
```

Try to avoid creating a custom query object for one dataset. It is easier for
teammates if every driver still accepts `DatasetQuery`, even when the supported
filter keys differ.

## Common Crawl Queries

The Common Crawl driver is a good example of this pattern. It still uses
`DatasetQuery`, but it interprets a few filter keys specially:

- `domain`: fetch pages under a domain such as `example.org`
- `url`: fetch one exact URL or a Common Crawl URL query
- `url_pattern`: pass an explicit CDX query string such as `*.example.org/*`
- `status`: filter by HTTP status such as `200`
- `mime`: filter by mime type such as `text/html`

Examples:

```python
DatasetQuery(filters={"domain": "example.org"}, limit=10)
DatasetQuery(filters={"url": "https://commoncrawl.org/"}, limit=5)
DatasetQuery(filters={"url_pattern": "*.example.org/*", "status": "200"}, limit=20)
DatasetQuery(filters={"domain": "example.org", "mime": "text/html"}, limit=10)
```

If you want to use this driver, install the optional dependency:

```bash
pip install -e .[commoncrawl]
```

This keeps the public interface small:

- experiments only need to know about `DatasetQuery`
- each dataset driver can decide which filter keys it supports
- new drivers can stay beginner-friendly instead of adding a second query API
