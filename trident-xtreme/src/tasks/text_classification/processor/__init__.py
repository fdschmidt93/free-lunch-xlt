from transformers.tokenization_utils_base import (BatchEncoding,
                                                  PreTrainedTokenizerBase)

from src.utils.hooks import processing_hooks


# @processing_hooks
def preprocess_fn(
    examples: dict,
    column_names: dict,  # {text, text_pair, labels}
    tokenizer: PreTrainedTokenizerBase,
) -> BatchEncoding:
    """


    Extracted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py
    """
    # Tokenize the texts
    text_column = column_names["text"]
    text_pair_column = column_names.get("text_pair", None)

    text = examples[text_column]
    text_pair = examples[text_pair_column] if text_pair_column is not None else None

    if isinstance(text, list):
        text = [instance.strip() for instance in text]
    if isinstance(text_pair, list):
        text_pair = [instance.strip() for instance in text_pair]
    if isinstance(text, str):
        text = text.strip()
    if isinstance(text_pair, str):
        text_pair = text_pair.strip()

    result = tokenizer(
        text=text,
        text_pair=text_pair,
    )

    # Map labels to IDs (not necessary for GLUE tasks)
    # if label_to_id is not None and "label" in examples:
    #     result["label"] = [
    #         (label_to_id[l] if l != -1 else -1) for l in examples["label"]
    #     ]
    return result
