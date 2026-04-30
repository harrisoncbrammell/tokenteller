from example_utils import run_random_sample_experiment


DATASET_ID = "sentence-transformers/parallel-sentences-opensubtitles"
DATASET_SUBSET = "all"
TEXT_FIELD = "english"
SAMPLE_SIZE = 100
POOL_SIZE = 10000
SAMPLE_SEED = 73545
OUTPUT_STEM = "opensubtitles_tokenizer_comparison"

METRIC_DESCRIPTIONS = {
    "token_count": "Average number of tokens produced per sampled subtitle record.",
    "compression_ratio": "Average token-to-character ratio, which gives a rough sense of how compactly the tokenizer encodes the text.",
    "oov_rate": "Average share of tokens treated as unknown or out-of-vocabulary by the tokenizer.",
    "fertility_rate": "Average number of tokens generated per word.",
    "mean_tokens_per_sentence": "Average token count per sentence across the sampled subtitle records.",
    "pieces_per_word": "Average number of subword pieces used to represent each word.",
    "max_pieces_per_word": "Largest number of pieces used for any single word seen in the sampled subtitle records.",
    "estimated_cost": "Estimated tokenization cost across the sampled records using the example price per 1,000 tokens.",
    "nsl": "Tokenizer length relative to the baseline tokenizer. `1.0` matches the baseline length; larger values mean more tokens than baseline.",
}


def main() -> None:
    run_random_sample_experiment(
        dataset_id=DATASET_ID,
        subset=DATASET_SUBSET,
        text_field=TEXT_FIELD,
        source_name="opensubtitles",
        sample_name="sampled_opensubtitles",
        sample_size=SAMPLE_SIZE,
        pool_size=POOL_SIZE,
        sample_seed=SAMPLE_SEED,
        output_stem=OUTPUT_STEM,
        summary_title="# Hugging Face tokenizer comparison on OpenSubtitles records",
        record_description="one subtitle line or short subtitle segment from a single dataset row, using the English text field.",
        metric_descriptions=METRIC_DESCRIPTIONS,
        pool_message=f"Building a pool from the first {POOL_SIZE} OpenSubtitles records...",
        selected_message="Selected {count} random records from the pooled set.",
        done_message="Compared Hugging Face tokenizers on random OpenSubtitles records.",
    )


if __name__ == "__main__":
    main()
