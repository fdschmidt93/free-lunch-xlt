from typing import Callable, Union

from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import (BatchEncoding,
                                                  PreTrainedTokenizerBase)


def tokenize_function(
    examples, column_names: dict, tokenizer: Union[Callable, PreTrainedTokenizerBase]
) -> BatchEncoding:
    return tokenizer(examples[column_names["text"]])


def group_texts(
    examples: dict,
    max_seq_length: int,
):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result


def preprocess_fn(
    dataset: Dataset,
    column_names: dict,
    tokenizer: Union[Callable, PreTrainedTokenizerBase],
    max_seq_length: int = 512,
) -> Dataset:
    remove_columns = dataset.column_names
    if isinstance(remove_columns, dict):
        remove_columns = list(remove_columns.values())[0]
    dataset = dataset.map(
        function=tokenize_function,
        fn_kwargs={"tokenizer": tokenizer, "column_names": column_names},
        remove_columns=remove_columns,
        batched=True,
    )
    dataset = dataset.map(
        function=group_texts, fn_kwargs={"max_seq_length": max_seq_length}, batched=True
    )
    return dataset
