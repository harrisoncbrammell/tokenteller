# Tokenteller

Tokenteller is a small Python toolkit for comparing tokenization behavior.

This version does not ship with concrete model or dataset drivers. Instead, it
provides:

- base classes for model, dataset, and test drivers
- a small `Experiment` object
- templates your teammates can fill in

Example:

```python
from my_project.my_model_driver import MyModelDriver
from my_project.my_dataset_driver import MyDatasetDriver
from tokenteller import Experiment
from tokenteller.testsuites.metrics import FragmentationTest, TokenCountTest

experiment = Experiment()
experiment.add_model(MyModelDriver(), name="my_model")
experiment.add_dataset(MyDatasetDriver(), name="my_dataset")
experiment.add_test(TokenCountTest(), model="my_model", dataset="my_dataset")
experiment.add_test(FragmentationTest(), model="my_model", dataset="my_dataset")

report = experiment.run()

print(report.summary_table())
```

Setup:

```bash
pip install -e .
```

## Project Notes

This version is intentionally kept classroom-friendly:

- the public API is small
- the main workflow is `add_model()`, `add_dataset()`, `add_test()`, then `run()`
- models, datasets, and tests are passed around as normal Python objects
- new model/tokenizer drivers only need `encode()`
- new dataset drivers only need `iter_records()`
- built-in metrics are short and readable
- concrete model and dataset drivers are left for your teammates to implement

Driver templates and extension notes:

- [docs/creating_drivers.md](D:/Development/School/asml/tokenteller/docs/creating_drivers.md)
- [models/template.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/models/template.py)
- [datasets/template.py](D:/Development/School/asml/tokenteller/src/tokenteller/drivers/datasets/template.py)
- [testsuites/template.py](D:/Development/School/asml/tokenteller/src/tokenteller/testsuites/template.py)
