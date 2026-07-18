"""Paths to datasets bundled with the :mod:`gpe` distribution."""

from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
GPE_DATA_PATH = DATA_DIR / "gpe.jsonl"
GRAPH_DATA_DIR = DATA_DIR / "graph"


def bundled_data_path() -> Path:
    """Return the path to the bundled GPE evaluation dataset."""
    return GPE_DATA_PATH


def bundled_graph_dir() -> Path:
    """Return the directory containing the bundled KnownGraph records."""
    return GRAPH_DATA_DIR
