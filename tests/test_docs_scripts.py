"""Smoke tests for the plot scripts in docs/.

These scripts are not exercised by the docs build (RTD only runs Sphinx), so
they can rot silently when the API changes — docs/create_logo.py did exactly
that (feedback.md item 11). Running them headless catches import- and
API-level breakage; the plots themselves are not checked.
"""

import runpy
from pathlib import Path

import matplotlib
import pytest

DOCS_DIR = Path(__file__).parent.parent / "docs"

SCRIPTS = [
    "create_logo.py",
    "plot_pulses.py",
    "changing_t_drain_plot.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_docs_script_runs(script, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # saved figures land in the temp directory
    matplotlib.use("Agg", force=True)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)
    runpy.run_path(str(DOCS_DIR / script), run_name="__main__")
