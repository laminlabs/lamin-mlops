from pathlib import Path
import nox
import os
from laminci import upload_docs_artifact
from laminci.nox import run_pre_commit, build_docs, run, install_lamindb

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
@nox.session()
def build(session, group):
    install_lamindb(session, branch="main")
    run(
        session,
        "uv pip install --system torchvision lightning wandb mlflow ipywidgets pytest",
    )
    run(session, f"pytest -s ./tests/test_notebooks.py::test_{group}")
    for path in Path(f"./docs_{group}").glob("*"):
        path.rename(f"./docs/{path.name}")
    build_docs(session, strict=True)
    upload_docs_artifact(aws=True)
