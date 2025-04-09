import nox
from subprocess import run
from laminci import upload_docs_artifact, run_notebooks
from laminci.nox import run_pre_commit, build_docs


@nox.session
def lint(session: nox.Session) -> None:
    run_pre_commit(session)


@nox.session()
def build(session):
    run(
        session,
        "uv pip install --system 'lamindb[jupyter]' torchvision lightning wandb mlflow",
    )
    run_notebooks("./docs")
    build_docs(session, strict=True)
    upload_docs_artifact(aws=True)
