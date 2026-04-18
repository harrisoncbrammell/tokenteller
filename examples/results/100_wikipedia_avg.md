# Hugging Face tokenizer comparison on 1,000 random Wikipedia articles

## Sample

- **Dataset**: `wikimedia/wikipedia`
- **Subset**: `20231101.en`
- **Pool size**: first 10000 streamed articles
- **Sample size**: 100 random articles

## Test summary

| Tokenizer | Token count | Compression ratio | OOV rate | Fertility rate | Mean tokens per sentence | Word Count | Pieces per word | Max pieces per word | Estimated cost | Normalized sequence length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2 | 2116.6 | 0.24268262064464424 | 0.0 | 1.5584563615172415 | 41.123724690938104 | 1437.26 | 1.4502493725234769 | 50 | 0.42332000000000003 | 1.0 |
| llama-sp | 2391.63 | 0.27123076621397507 | 0.0 | 1.7415540555760953 | 45.07573482655894 | 1437.26 | 1.598144117962772 | 52 | 0.717489 | 1.116427369241473 |
| t5 | 2239.27 | 0.25121103377576987 | 0.0043019114014177995 | 1.6119008665496233 | 42.86291407709453 | 1437.26 | 1.6119008665496233 | 56 | 0.5598175 | 1.0373238668992335 |
| wordpiece | 1966.56 | 0.2179388029788114 | 7.547169811320755e-05 | 1.3978507314572557 | 36.68216300369765 | 1437.26 | 1.3978507314572557 | 52 | 0.294984 | 0.9022440637527808 |

## Metric guide

- **Token count**: Average number of tokens produced per sampled article.
- **Compression ratio**: Average token-to-character ratio, which gives a rough sense of how compactly the tokenizer encodes the text.
- **OOV rate**: Average share of tokens treated as unknown or out-of-vocabulary by the tokenizer.
- **Fertility rate**: Average number of tokens generated per word.
- **Mean tokens per sentence**: Average token count per sentence across the sampled articles.
- **Word Count**: Metric reported by the test summary.
- **Pieces per word**: Average number of subword pieces used to represent each word.
- **Max pieces per word**: Largest number of pieces used for any single word seen in the sampled articles.
- **Estimated cost**: Estimated tokenization cost across the sampled articles using the example price per 1,000 tokens.
- **Normalized sequence length**: Tokenizer length relative to the baseline tokenizer. `1.0` matches the baseline length; larger values mean more tokens than baseline.