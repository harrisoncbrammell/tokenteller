# Tokenteller

Tokenteller is a small Python toolkit for building and running tokenizer experiments.

## Install

Install the package in editable mode while you work on the project.

```bash
pip install -e .
```

## Create An Experiment

Build your model and dataset driver objects first, then add them to an `Experiment`.

```python
from my_project.my_dataset_driver import MyDatasetDriver
from my_project.my_model_driver import MyModelDriver
from tokenteller import Experiment
from tokenteller.core.types import DatasetQuery, RunConfig
from tokenteller.testsuites.metrics import FragmentationTest, TokenCountTest

model = MyModelDriver(model_path="my-model")
dataset = MyDatasetDriver(data_path="my-data.jsonl")

experiment = Experiment(
    run_config=RunConfig(max_workers=4, baseline_tokenizer="my_model"),
)

experiment.add_model(model, name="my_model")
experiment.add_dataset(dataset, name="my_dataset")

experiment.add_test(
    TokenCountTest(label="english token count"),
    model="my_model",
    dataset="my_dataset",
    query=DatasetQuery(filters={"language": "en"}, limit=50),
)
experiment.add_test(
    FragmentationTest(label="english fragmentation"),
    model="my_model",
    dataset="my_dataset",
    query=DatasetQuery(filters={"language": "en"}, limit=50),
)
```

## Run An Experiment

Call `run()` after all models, datasets, and tests have been added.

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

If two tests are the same type, you can compare them.

```python
print(experiment.tests[0].compare(experiment.tests[1]))
```

## Driver Instructions

Driver-writing instructions live here:

- [docs/creating_drivers.md](D:/Development/School/asml/tokenteller/docs/creating_drivers.md)
