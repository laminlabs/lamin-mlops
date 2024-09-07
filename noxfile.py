import nox
from laminci import upload_docs_artifact
from laminci.nox import build_docs, run_pre_commit, run_pytest, run

# we'd like to aggregate coverage information across sessions
# and for this the code needs to be located in the same
# directory in every github action runner
# this also allows to break out an installation section
nox.options.default_venv_backend = "none"


@nox.session
def lint(session: nox.Session) -> None:
    run_pre_commit(session)


@nox.session()
def build(session):
    run(
        session,
        "uv pip install --system 'lamindb[jupyter,aws]' torch torchvision lightning wandb",
    )
    run_pytest(session)
    build_docs(session, strict=True)
    upload_docs_artifact(aws=True)
