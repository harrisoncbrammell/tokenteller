# Driver Guide

Tokenteller is built around three kinds of drivers:

- model drivers
- dataset drivers
- test drivers

Each driver inherits from a base class near the drivers themselves:

- model drivers use [src/tokenteller/drivers/models/base.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/models/base.py)
- dataset drivers use [src/tokenteller/drivers/datasets/base.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/datasets/base.py)
- test drivers use [src/tokenteller/testsuites/base.py](D:/Development/School/asml/tokenteller/src/tokenteller/testsuites/base.py)

## Model Drivers

Start from `BaseModelDriver` when you want to support a new model tokenizer.

You must add:

- `encode(text)`

You may also add:

- `decode(token_ids)`
- `batch_encode(texts)`
- `token_count(text)`
- `fragmentation_stats(text)`

Put any model-specific setup in the constructor, such as:

- model path
- checkpoint name
- tokenizer options
- remote loading flags

`encode()` must return a `TokenizationResult`. Expected fields:

- `token_ids`
- `tokens`
- `token_count`
- `offsets`
- `raw`

Field meanings:

- `token_ids`: integer ids in model order
- `tokens`: readable token strings when available
- `token_count`: usually `len(token_ids)`
- `offsets`: optional `(start, end)` spans for each token
- `raw`: optional extra library output

Use the template in [models/template.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/models/template.py).

## Dataset Drivers

Start from `BaseDatasetDriver` when you want to support a new dataset source.

You must add:

- `iter_records(query)`

You may also add:

- `count(query)`

Put any dataset-specific setup in the constructor, such as:

- dataset path
- split name
- API client
- preload settings

`iter_records(query)` must yield `DatasetRecord` objects. Expected fields:

- `id`
- `text`
- `categories`
- `metadata`

Field meanings:

- `id`: stable record identifier
- `text`: the text that will be tokenized
- `categories`: simple grouping labels like language or domain
- `metadata`: any extra information you want to keep

Use `categories` for fields you want to group by later, such as:

- language
- domain
- source

Use the template in [datasets/template.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/datasets/template.py).

## Test Drivers

Start from `BaseTestDriver` when you want to add a new metric or analysis.

You must add:

- `name()`
- `run_case(tokenizer, record, context)`

Put any test-specific setup in the constructor. If you add your own `__init__()`,
call `super().__init__(label=...)` first so the built-in state tracking still works.

`run_case(...)` should return a `TestCaseResult`. In practice:

- `metrics` should hold the numeric or summary values you care about
- `artifacts` can hold extra detail like token lists or spans

The base class already provides `run_batch()`, result storage, text summaries,
and a default `compare()` method.

Each test object also stores:

- which model it was bound to
- which dataset it was bound to
- whether it has been run yet
- its saved results, warnings, and summary row

You can print a test object directly to see whether it is pending, failed, or completed.
You can also override `compare()` if it makes sense to compare two runs of the same test type.

Use the template in [testsuites/template.py](D:/Development/School/asml/tokenteller/src/tokenteller/testsuites/template.py).

## Constructor Parameters

The library does support whatever constructor parameters your driver needs.
The experiment only receives finished driver objects, so parameter handling stays
inside your driver classes.

Examples:

```python
model = MyModelDriver(model_path="my-model", trust_remote_code=True)
dataset = MyDatasetDriver(data_path="data.jsonl", split="train")
test = MyTestDriver(label="Hindi prompts")
```

## Minimal Workflow

1. Create the driver class from a template.
2. Fill in the required methods.
3. Instantiate the driver object directly.
4. Create an `Experiment`, add models and datasets, bind tests to them, and call `run()`.

Example:

```python
from tokenteller import Experiment
from tokenteller.core.types import DatasetQuery
from tokenteller.testsuites.metrics import TokenCountTest
from my_project.my_model_driver import MyModelDriver
from my_project.my_dataset_driver import MyDatasetDriver

experiment = Experiment()
experiment.add_model(MyModelDriver(model_path="my-model"), name="my_model")
experiment.add_dataset(MyDatasetDriver(data_path="my-data.jsonl"), name="my_dataset")
experiment.add_test(
    TokenCountTest(label="english sample"),
    model="my_model",
    dataset="my_dataset",
    query=DatasetQuery(limit=25),
)

report = experiment.run()
```

## Keep It Small

For a class project, keep drivers small:

- one class per model driver
- one class per dataset source
- one class per metric
- override only the methods you actually need

If a driver starts getting too long, then split helper logic into extra functions. Start simple first.
