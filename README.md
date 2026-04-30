# tokenteller

## setup

run this in powershell from the project folder

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_ta.ps1
```

if that does not work then do this

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-ta.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

## run the example files

run any example like this

```powershell
.\.venv\Scripts\python.exe .\examples\n_random_wikipedia.py
.\.venv\Scripts\python.exe .\examples\n_random_opensub.py
.\.venv\Scripts\python.exe .\examples\n_random_common_crawl.py
```

to build the combined chart from the csv results run this

```powershell
.\.venv\Scripts\python.exe .\examples\visualize_tokenizer_comparison.py
```

## notes

- the first run may take a while because hugging face downloads datasets and tokenizers
- if hugging face warns about no token that is usually fine for class use
- the progress bar should update in place while the tests are running
