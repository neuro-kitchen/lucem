import sys

import pytest

import lucem
from lucem import _cli


def test_plugin_search_lists_folders_and_never_fails():
    assert isinstance(lucem.ml.plugin_dirs(), list)
    assert isinstance(lucem.ml.plugins(), list)
    assert isinstance(lucem.ml.problems(), list)


def test_a_missing_plugin_program_is_a_runtime_error():
    with pytest.raises(RuntimeError):
        lucem.ml.Plugin("lucem-ml-there-is-no-such-plugin")


def test_the_app_is_found_beside_python_or_on_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    assert lucem.app.program() is None
    with pytest.raises(FileNotFoundError, match="lucem\\[app\\]"):
        lucem.app.open()
    exe = tmp_path / ("lucem-app.exe" if sys.platform == "win32" else "lucem-app")
    exe.write_text("")
    assert lucem.app.program() == exe


def test_the_command_sends_other_commands_to_the_full_cli(capsys):
    with pytest.raises(SystemExit, match="cargo install"):
        _cli.main(["serve"])
    _cli.main(["version"])
    assert capsys.readouterr().out.strip() == lucem.__version__
