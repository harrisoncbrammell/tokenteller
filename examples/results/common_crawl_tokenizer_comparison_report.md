# Hugging Face tokenizer comparison on Common Crawl records

## Sample

- **Dataset**: `allenai/c4`
- **Subset**: `en`
- **Text field**: `text`
- **Record type**: one cleaned Common Crawl document from the C4 dataset, using the document text field.
- **Sampling method**: bounded random sample.
- **Pool size**: first 10000 streamed documents
- **Sample size**: 100 random documents
- **How records were chosen**: we first streamed the first 10000 documents from the dataset, then used a fixed random seed (`38192`) to select 100 documents from that pooled set.

## Test summary

| Tokenizer | Token count | Compression ratio | OOV rate | Fertility rate | Mean tokens per sentence | Word Count | Pieces per word | Max pieces per word | Estimated cost | Normalized sequence length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2 | 435.86 | 0.22120418613279852 | 0.0 | 1.3215480255198369 | 20.19658491160656 | 335.52 | 1.2996853490590068 | 46 | 0.087172 | 1.0 |
| llama-sp | 456.89 | 0.23250882122939845 | 0.0 | 1.3899926266651437 | 21.181657265104594 | 335.52 | 1.3525809571931173 | 45 | 0.137067 | 1.0501815691399425 |
| t5 | 460.11 | 0.23376507082047962 | 0.00015404040404040403 | 1.3961528299934491 | 21.332917418387282 | 335.52 | 1.3961528299934491 | 46 | 0.1150275 | 1.0561580833731954 |
| wordpiece | 429.42 | 0.21681816943008325 | 0.00011764705882352942 | 1.295001086219992 | 19.78204524488285 | 335.52 | 1.295001086219992 | 43 | 0.064413 | 0.9808317423514592 |

## Metric guide

- **Token count**: Average number of tokens produced per sampled Common Crawl document.
- **Compression ratio**: Average token-to-character ratio, which gives a rough sense of how compactly the tokenizer encodes the text.
- **OOV rate**: Average share of tokens treated as unknown or out-of-vocabulary by the tokenizer.
- **Fertility rate**: Average number of tokens generated per word.
- **Mean tokens per sentence**: Average token count per sentence across the sampled Common Crawl documents.
- **Word Count**: Metric reported by the test summary.
- **Pieces per word**: Average number of subword pieces used to represent each word.
- **Max pieces per word**: Largest number of pieces used for any single word seen in the sampled Common Crawl documents.
- **Estimated cost**: Estimated tokenization cost across the sampled documents using the example price per 1,000 tokens.
- **Normalized sequence length**: Tokenizer length relative to the baseline tokenizer. `1.0` matches the baseline length; larger values mean more tokens than baseline.