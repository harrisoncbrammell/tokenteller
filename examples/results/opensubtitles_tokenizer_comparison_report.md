# Hugging Face tokenizer comparison on OpenSubtitles records

## Sample

- **Dataset**: `sentence-transformers/parallel-sentences-opensubtitles`
- **Subset**: `all`
- **Text field**: `english`
- **Pool size**: first 10000 streamed records
- **Sample size**: 100 random records

## Test summary

| Tokenizer | Token count | Compression ratio | OOV rate | Fertility rate | Mean tokens per sentence | Word Count | Pieces per word | Max pieces per word | Estimated cost | Normalized sequence length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2 | 9.58 | 0.2916351138104564 | 0.0 | 1.4851752511176424 | 8.868333333333334 | 7.36 | 1.4851752511176424 | 5 | 0.001916 | 1.0 |
| llama-sp | 10.19 | 0.3065286382958585 | 0.0 | 1.558170838461056 | 9.433333333333334 | 7.36 | 1.558170838461056 | 5 | 0.003057 | 1.052000665269201 |
| t5 | 10.63 | 0.3226025215842076 | 0.0 | 1.6414236949845644 | 9.846666666666666 | 7.36 | 1.6414236949845644 | 6 | 0.0026575 | 1.1082641343954787 |
| wordpiece | 10.08 | 0.30435312333885517 | 0.0 | 1.5687360192544975 | 9.315 | 7.36 | 1.5687360192544975 | 7 | 0.0015119999999999999 | 1.0615529975622655 |

## Metric guide

- **Token count**: Average number of tokens produced per sampled subtitle record.
- **Compression ratio**: Average token-to-character ratio, which gives a rough sense of how compactly the tokenizer encodes the text.
- **OOV rate**: Average share of tokens treated as unknown or out-of-vocabulary by the tokenizer.
- **Fertility rate**: Average number of tokens generated per word.
- **Mean tokens per sentence**: Average token count per sentence across the sampled subtitle records.
- **Word Count**: Metric reported by the test summary.
- **Pieces per word**: Average number of subword pieces used to represent each word.
- **Max pieces per word**: Largest number of pieces used for any single word seen in the sampled subtitle records.
- **Estimated cost**: Estimated tokenization cost across the sampled records using the example price per 1,000 tokens.
- **Normalized sequence length**: Tokenizer length relative to the baseline tokenizer. `1.0` matches the baseline length; larger values mean more tokens than baseline.