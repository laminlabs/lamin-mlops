import sys
from pathlib import Path

import nbproject_test as test

sys.path[:0] = [str(Path(__file__).parent.parent)]

from noxfile import GROUPS  # noqa

DOCS = Path(__file__).parents[1] / "docs/"


def test_mlops():
    for filename in GROUPS["mlops"]:
        print(filename)
        test.execute_notebooks(DOCS / filename, write=True)
