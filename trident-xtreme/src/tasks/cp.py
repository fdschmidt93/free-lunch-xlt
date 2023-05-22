from pathlib import Path
import shutil

cwd = Path.cwd()
qa = cwd.joinpath("question_answering")

for f in ["text_classification", "token_classification", "language_modeling", "multiple_choice"]:
    shutil.copytree(qa, cwd.joinpath(f))
