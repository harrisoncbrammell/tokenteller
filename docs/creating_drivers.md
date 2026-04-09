# Creating Drivers

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

You must implement:

- `encode(text)`

Optional helper:

- `decode(token_ids)`
- `batch_encode(texts)`
- `token_count(text)`
- `fragmentation_stats(text)`

Your `encode()` method should return a `TokenizationResult`. That object contains:

- `token_ids`
- `tokens`
- `token_count`
- `offsets`
- `raw`

Use the template in [models/template.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/models/template.py).

## Dataset Drivers

Start from `BaseDatasetDriver` when you want to support a new dataset source.

You must implement:

- `iter_records(query)`

Optional helpers:

- `count(query)`

Your dataset should return `DatasetRecord` objects. Each record should include:

- `id`
- `text`
- `categories`
- `metadata`

Use `categories` for fields you want to group by later, such as:

- language
- domain
- source

Use the template in [datasets/template.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/datasets/template.py).

## Test Drivers

Start from `BaseTestDriver` when you want to add a new metric or analysis.

You must implement:

- `name()`
- `run_case(tokenizer, record, context)`

You do not need to write your own parallel runner unless you want custom behavior. The base class already provides `run_batch()`.
If you add your own `__init__()` to a test class, call `super().__init__(label=...)` first so the built-in status and result tracking still works.

Each test object also stores:

- which model it was bound to
- which dataset it was bound to
- whether it has been run yet
- its saved results, warnings, and summary row

You can print a test object directly to see whether it is pending, failed, or completed.
You can also override `compare()` if it makes sense to compare two runs of the same test type.

Use the template in [testsuites/template.py](D:/Development/School/asml/tokenteller/src/tokenteller/testsuites/template.py).

## Using A Driver

After writing a driver, instantiate it directly and pass the object into `Experiment`.

Model:

```python
model = MyModelDriver()
```

Dataset:

```python
dataset = MyDatasetDriver()
```

Test:

```python
test = MyTestDriver()
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

## Suggested Student Approach

For a class project, keep drivers small:

- one class per model driver
- one class per dataset source
- one class per metric
- override only the methods you actually need

If a driver starts getting too long, then split helper logic into extra functions. Start simple first.
