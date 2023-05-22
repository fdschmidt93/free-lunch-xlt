from typing import Union

import torch
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset, load_metric
from pytorch_lightning import LightningDataModule
from transformers.tokenization_utils_base import (BatchEncoding,
                                                  PreTrainedTokenizerBase)
from trident import TridentDataModule, TridentModule


def preprocess_fn(
    examples: list,
    tokenizer: PreTrainedTokenizerBase,
    column_names: dict,
    label2id: dict,
    label_all_tokens: bool = True,
    max_length: int = 256,
) -> BatchEncoding:
    """

    Notes:
        * -100 is `ignore_idx` for loss computation
    Credits to Huggingface's `run_ner.py <>`

    """
    # Tokenize the texts
    text_column = column_names["text"]
    label_column = column_names["label"]

    tokenized_inputs = tokenizer(
        examples[text_column],
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    labels = []
    for i, label in enumerate(examples[label_column]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                token_label = label[word_idx]
                l = (
                    token_label
                    if isinstance(token_label, int)
                    else label2id[token_label]
                )
                label_ids.append(l)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if label_all_tokens:
                    token_label = label[word_idx]
                    l = (
                        token_label
                        if isinstance(token_label, int)
                        else label2id[token_label]
                    )
                else:
                    l = -100
                label_ids.append(l)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_label_mapping(datamodule: LightningDataModule, label_column: str):
    # set/getattr to avoid linter complaints
    dataset = None
    label_list = []

    def extract_or_check_features(dataset, name):
        dataset_features = dataset.features[label_column].feature.names
        nonlocal label_list
        if len(label_list) == 0:
            label_list = dataset_features
        else:
            assert (
                label_list == dataset_features
            ), f"{name} has {dataset_features} labels, but initial dataset has {label_list}"

    for split in ["train", "val", "test"]:
        if dataset := getattr(datamodule, f"dataset_{split}", None):
            if isinstance(dataset, dict):
                for name, dataset_ in dataset.items():
                    extract_or_check_features(dataset_, name)
            else:
                extract_or_check_features(dataset, split)

    assert len(label_list) > 0
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label_list)
    setattr(datamodule, "label2id", label2id)
    setattr(datamodule, "id2label", id2label)
    setattr(datamodule, "num_labels", num_labels)
    setattr(datamodule, "label_list", label_list)


def module_setup(module: TridentModule, *args, **kwargs):
    from transformers.models.auto.modeling_auto import \
        AutoModelForTokenClassification

    module.model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=module.hparams.model.pretrained_model_name_or_path,
        label2id=module.trainer.datamodule.label2id,
        id2label=module.trainer.datamodule.id2label,
        num_labels=module.trainer.datamodule.num_labels,
    )


# TODO: improve stateful transfer such that `on_after_setup` is not required
# the issue is label2id can only be inferred after the datamodule
def on_after_setup(
    datamodule: TridentDataModule,
    tokenizer: PreTrainedTokenizerBase,
    column_names: dict,
    label_all_tokens: bool = True,
    batched: bool = True,
    num_proc: int = 1,
    max_length: int = 510,
):
    get_label_mapping(datamodule, column_names["label"])

    for split in ["train", "val", "test"]:
        dataset_name = f"dataset_{split}"
        if dataset := getattr(datamodule, dataset_name):
            if not isinstance(dataset, dict):
                setattr(
                    datamodule,
                    dataset_name,
                    dataset.map(
                        preprocess_fn,
                        batched=batched,
                        num_proc=num_proc,
                        fn_kwargs={
                            "tokenizer": tokenizer,
                            "column_names": column_names,
                            "label_all_tokens": label_all_tokens,
                            "label2id": datamodule.label2id,
                            "max_length": max_length,
                        },
                    ),
                )
            else:
                for ds_name, ds in dataset.items():
                    dataset[ds_name] = ds.map(
                        preprocess_fn,
                        batched=batched,
                        num_proc=num_proc,
                        fn_kwargs={
                            "tokenizer": tokenizer,
                            "column_names": column_names,
                            "label_all_tokens": label_all_tokens,
                            "label2id": datamodule.label2id,
                            "max_length": max_length,
                        },
                    )


def prepare_step_outputs(
    module: TridentModule, outputs: dict, *args, **kwargs
) -> dict[str, list]:

    predictions = outputs["logits"]
    labels = outputs["labels"].long().detach().cpu().numpy().tolist()
    predictions = (
        torch.argmax(predictions, dim=-1).long().detach().cpu().numpy().tolist()
    )
    # Remove ignored index (special tokens)
    true_predictions = [
        [
            module.trainer.datamodule.label_list[p]
            for (p, l) in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [
            module.trainer.datamodule.label_list[l]
            for (p, l) in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(predictions, labels)
    ]
    return {"predictions": true_predictions, "references": true_labels}


def compute_metrics(logits, labels, label_list):
    predictions = torch.argmax(logits, dim=-1).long().detach().cpu().numpy().tolist()
    labels = labels.long().detach().cpu().numpy().tolist()
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (_, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric = load_metric("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)

    return results["overall_f1"]


def remap_maskhaner_labels(examples: dict) -> dict:
    """Label indices larger than 6 are remapped to 0."""
    examples["ner_tags"] = [
        [tag if tag < 7 else 0 for tag in instance] for instance in examples["ner_tags"]
    ]
    return examples


def load_maskhaner(*args, **kwargs) -> Union[dict[str, Dataset], Dataset]:

    """Remap MasakhaNER labels to WikiANN.

        Assumes WikiANN and MasakhaNER labels are aligned up to the final two labels.

        - `B-DATE` and `I-DATE` are remapped to 0 (`O`)
        - The feature column of the dataset are augmented to only comprise 2 labels and cut the last two features

        >>    wikiann_labels = {
        >>        "O": 0,
        >>        "B-PER": 1,
        >>        "I-PER": 2,
        >>        "B-ORG": 3,
        >>        "I-ORG": 4,
        >>        "B-LOC": 5,
        >>        "I-LOC": 6,
        >>    }

        >>    masakhaner_labels = {
        >>        "O": 0,
        >>        "B-PER": 1,
        >>        "I-PER": 2,
        >>        "B-ORG": 3,
        >>        "I-ORG": 4,
    _   >>        "B-LOC": 5,
        >>        "I-LOC": 6,
        >>        "B-DATE": 7,
        >>        "I-DATE": 8,
        >>    }

    """
    dataset = load_dataset("masakhaner", *args, **kwargs)
    dataset = dataset.map(remap_maskhaner_labels, batched=True)
    if isinstance(dataset, dict):
        for key, value in dataset.items():
            value.features["ner_tags"].feature.names = value.features[
                "ner_tags"
            ].feature.names[:-2]
            value.features["ner_tags"].feature.num_classes = 7
            dataset[key] = value
    else:
        dataset.features["ner_tags"].feature.names = dataset.features[
            "ner_tags"
        ].feature.names[:-2]
        dataset.features["ner_tags"].feature.num_classes = 7
    return dataset
