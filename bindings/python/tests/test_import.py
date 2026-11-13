from importlib.metadata import version

import lucem


def test_version():
    # The extension's version is the installed package's (the workspace version, set at build)
    assert lucem.__version__ == version("lucem")
