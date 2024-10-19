import nox
from subprocess import run
from laminci import upload_docs_artifact, run_notebooks
from laminci.nox import run_pre_commit


@nox.session
def lint(session: nox.Session) -> None:
    run_pre_commit(session)


@nox.session()
def build(session):
    run("uv pip install --system 'scipy>=1.12.0,<1.13.0rc1'", shell=True)
    run(
        "uv pip install --system 'lamindb[jupyter,aws]' torch torchvision lightning wandb",
        shell=True,
    )
    run_notebooks("./docs")
    run("lndocs --strict", shell=True)
    upload_docs_artifact(aws=True)
