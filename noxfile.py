import os
from pathlib import Path

import nox
from laminci import convert_executable_md_files, upload_docs_artifact
from laminci.nox import build_docs, install_lamindb, run, run_pre_commit

IS_PR = os.getenv("GITHUB_EVENT_NAME") != "push"


GROUPS = {}
GROUPS["mlops"] = ["mnist.ipynb", "wandb.ipynb", "mlflow.ipynb", "croissant.ipynb"]


@nox.session
def lint(session: nox.Session) -> None:
    run_pre_commit(session)


@nox.session
@nox.parametrize(
    "group",
    [
        "mlops",
    ],
)
def build(session, group):
    install_lamindb(session, branch="main")
    convert_executable_md_files()
    run(
        session,
        "uv pip install --system torchvision lightning wandb mlflow ipywidgets pytest",
    )
    run(session, f"pytest -s ./tests/test_notebooks.py::test_{group}")
    for path in Path(f"./docs_{group}").glob("*"):
        path.rename(f"./docs/{path.name}")
    build_docs(session, strict=True)
    upload_docs_artifact(aws=True)
