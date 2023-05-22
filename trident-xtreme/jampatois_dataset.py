# ADAPTED FROM: https://huggingface.co/datasets/juletxara/tydiqa_xtreme/blob/main/tydiqa_xtreme.py
# Why an own dataset? jultxara did not use "goldp" dataset for Korean as per Github page
# Creating own dataset with assertion that context[answer_start + len(answer)] == answer (cf. line 150)

import csv

import datasets
from datasets import DownloadManager
from datasets.tasks import QuestionAnsweringExtractive

# TODO(tydiqa): BibTeX citation
_CITATION = """\
@article{tydiqa,
title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
year    = {2020},
journal = {Transactions of the Association for Computational Linguistics}
}
"""

# TODO(tydiqa):
_DESCRIPTION = """\
TyDi QA is a question answering dataset covering 11 typologically diverse languages with 204K question-answer pairs.
The languages of TyDi QA are diverse with regard to their typology -- the set of linguistic features that each language
expresses -- such that we expect models performing well on this set to generalize across a large number of the languages
in the world. It contains language phenomena that would not be found in English-only corpora. To provide a realistic
information-seeking task and avoid priming effects, questions are written by people who want to know the answer, but
don’t know the answer yet, (unlike SQuAD and its descendents) and the data is collected directly in each language without
the use of translation (unlike MLQA and XQuAD).

We also include "translate-train" and "translate-test" splits for each non-English languages from XTREME (Hu et al., 2020). These splits are the automatic translations from English to each target language used in the XTREME paper [https://arxiv.org/abs/2003.11080]. The "translate-train" split purposefully ignores the non-English TyDiQA-GoldP training data to simulate the transfer learning scenario where original-language data is not available and system builders must rely on labeled English data plus existing machine translation systems.
"""

_VERSION = datasets.Version("1.1.0", "")


class JamPatoisNLIConfig(datasets.BuilderConfig):
    """BuilderConfig for TydiQa."""

    def __init__(self, **kwargs):
        """

        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(JamPatoisNLIConfig, self).__init__(version=_VERSION, **kwargs)


class JamPatoisNLI(datasets.GeneratorBasedBuilder):
    """TyDi QA: Information-Seeking QA in Typologically Diverse Languages."""

    BUILDER_CONFIGS = [
        JamPatoisNLIConfig(
            description=f"JamPatoisNLI train and test splits, with machine-translated "
            "translate-train/translate-test splits "
            "from XTREME (Hu et al., 2020).",
        )
    ]

    def _info(self):
        # TODO(tydiqa): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            features=datasets.Features(
                {
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label": datasets.ClassLabel(
                        names=["entailment", "neutral", "contradiction"]
                    ),
                }
            ),
            supervised_keys=None,
            homepage="TODO",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question",
                    context_column="context",
                    answers_column="answers",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(tydiqa): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs

        files = dl_manager.download_and_extract(
            "https://huggingface.co/datasets/Ruth-Ann/jampatoisnli/resolve/main/jampatois-nli-data.zip"
        )
        filepaths = list(dl_manager.iter_files(files))
        return [
            datasets.SplitGenerator(
                name="test",
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepaths": filepaths},
            )
        ]
        # filepaths = {"test": p for s in SPLIT for p in filepaths if p.endswith(f"{s}.csv")}
        # assert len(filepaths) == 3
        # return [
        #     datasets.SplitGenerator(
        #         name=split,
        #         # These kwargs will be passed to _generate_examples
        #         gen_kwargs={"filepath": path},
        #     )
        #     for split, path in filepaths.items()
        # ]

    def _generate_examples(self, filepaths):
        """Yields examples."""
        idx = -1
        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    idx += 1
                    yield idx, {
                        "premise": row["premise"],
                        "hypothesis": row["hypothesis"],
                        "label": row["label"],
                    }
