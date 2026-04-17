# Tokenteller

Tokenteller is a small Python toolkit for building and running tokenizer experiments.

## Install

Install the package in editable mode while you work on the project.

```bash
pip install -e .
```

## Create An Experiment

Build your model driver objects first, then add self-contained tests to an `Experiment`.

```python
from my_project.my_dataset_driver import MyDatasetDriver
from my_project.my_model_driver import MyModelDriver
from tokenteller import Experiment
from tokenteller.core.types import DatasetQuery, DatasetRecord, RunConfig, TestCaseResult, TestContext
from tokenteller.testsuites.base import BaseTestDriver

model = MyModelDriver(model_path="my-model")


class EnglishTokenCountTest(BaseTestDriver):
    def __init__(self, label: str | None = None):
        super().__init__(label=label)
        self.dataset = MyDatasetDriver()
        self.query = DatasetQuery(filters={"language": "en"}, limit=50)

    def name(self) -> str:
        return "token_count"

    def get_records(self) -> list[DatasetRecord]:
        return list(self.dataset.iter_records(self.query))

    def run_case(self, tokenizer, record, context: TestContext) -> TestCaseResult:
        tokenization = context.get_tokenization(tokenizer, record)
        return TestCaseResult(
            record_id=record.id,
            tokenizer_name=tokenizer.name,
            test_name=self.name(),
            metrics={"token_count": tokenization.token_count},
            artifacts={},
        )


experiment = Experiment(
    run_config=RunConfig(baseline_tokenizer="my_model"),
)

experiment.add_model(model, name="my_model")
experiment.add_test(EnglishTokenCountTest(label="english token count"), model="my_model")
```

If you want tiny example drivers to copy from, start here:

- [src/tokenteller/drivers/models/sentencepiece.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/models/sentencepiece.py)
- [src/tokenteller/drivers/datasets/common_crawl.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/datasets/common_crawl.py)

The Common Crawl example keeps the same `DatasetQuery` shape as every other
dataset driver:

```python
from tokenteller.core.types import DatasetQuery
from tokenteller.drivers.datasets import CommonCrawlDatasetDriver

dataset = CommonCrawlDatasetDriver()
query = DatasetQuery(
    filters={"domain": "commoncrawl.org", "status": "200", "mime": "text/html"},
    limit=10,
)
```

## Dataset Queries

Every dataset driver receives the same `DatasetQuery` object:

- `filters`: simple equality filters
- `limit`: maximum number of records to return
- `sample_strategy`: `head`, `tail`, or `random`
- `seed`: random seed used with `sample_strategy="random"`

Example:

```python
query = DatasetQuery(
    filters={"language": "en"},
    limit=25,
    sample_strategy="random",
    seed=7,
)
```

The meaning of `filters` depends on the dataset driver:

- a small in-memory dataset might support keys like `language` or `domain`
- the Common Crawl driver understands `domain`, `url`, `url_pattern`, `status`, and `mime`

Common Crawl examples:

```python
DatasetQuery(filters={"domain": "example.org"}, limit=10)
DatasetQuery(filters={"url": "https://commoncrawl.org/"}, limit=5)
DatasetQuery(filters={"url_pattern": "*.example.org/*", "status": "200"}, limit=20)
DatasetQuery(filters={"domain": "example.org", "mime": "text/html"}, limit=10)
```

If you want to use the Common Crawl driver, install the optional dependency:

```bash
pip install -e .[commoncrawl]
```

## Run An Experiment

Call `run()` after all models and tests have been added.

```python
report = experiment.run()
```

## View Experiment Data

Use the report for experiment-wide output.

```python
print(report.summary_table())
print(report.summary)
print(report.results)
print(report.warnings)
```

Use the saved test objects for per-test output.

```python
test = experiment.tests[0]

print(test.status)
print(test.summary_table())
print(test.results)
print(test.warnings)
```

## Driver Instructions

Driver-writing instructions live here:

- [docs/creating_drivers.md](D:/Development/School/asml/tokenteller/docs/creating_drivers.md)
