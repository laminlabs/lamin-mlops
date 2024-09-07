import nox
from subprocess import run
from laminci import upload_docs_artifact, run_notebooks
from laminci.nox import build_docs, run_pre_commit

nox.options.default_venv_backend = "none"


@nox.session
def lint(session: nox.Session) -> None:
    run_pre_commit(session)


@nox.session()
def build(session):
    run(
        "uv pip install --system 'lamindb[jupyter,aws]' torch torchvision lightning wandb",
        shell=True,
    )
    run_notebooks("./docs")
    build_docs(session, strict=True)
    upload_docs_artifact(aws=True)
