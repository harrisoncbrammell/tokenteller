# Tokenteller

Tokenteller is a small Python toolkit for building and running tokenizer experiments.

## Install

Install the package in editable mode while you work on the project.

```bash
pip install -e .
```

## Create An Experiment

Build self-contained tests first, then add them to an `Experiment`.

```python
from my_project.my_dataset_driver import MyDatasetDriver
from my_project.my_model_driver import MyModelDriver
from tokenteller import Experiment
from tokenteller.core.types import DatasetQuery, DatasetRecord, RunConfig, TestCaseResult, TestContext
from tokenteller.testsuites.base import BaseTestDriver

class EnglishTokenCountTest(BaseTestDriver):
    def __init__(self, model, label: str | None = None):
        super().__init__(model=model, label=label)
        self.dataset = MyDatasetDriver()
        self.query = DatasetQuery(filters={"language": "en"}, limit=50)

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


experiment = Experiment(run_config=RunConfig())
experiment.add_test(EnglishTokenCountTest(MyModelDriver(model_path="my-model"), label="english token count"))
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

Call `run()` after all tests have been added.

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
print(test.summary)
print(test.results)
print(test.warnings)
```

## Driver Instructions

Driver-writing instructions live here:

- [docs/creating_drivers.md](D:/Development/School/asml/tokenteller/docs/creating_drivers.md)
