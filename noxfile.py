from pathlib import Path
import nox
from laminci import upload_docs_artifact
from laminci.nox import run_pre_commit, build_docs, run


GROUPS = {}
GROUPS["mlops"] = ["mnist.ipynb", "wandb.ipynb", "mlflow.ipynb"]


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
    run(
        session,
        "uv pip install --system 'lamindb[jupyter]' torchvision lightning wandb mlflow pytest",
    )
    run(session, f"pytest -s ./tests/test_notebooks.py::test_{group}")
    for path in Path(f"./docs_{group}").glob("*"):
        path.rename(f"./docs/{path.name}")
    build_docs(session, strict=True)
    upload_docs_artifact(aws=True)
