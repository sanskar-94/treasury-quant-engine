import pytest


def pytest_collection_modifyitems(config, items):
    """Skip curve-dependent tests when the processed data has not been built.

    ``data/processed`` is gitignored, so a clean clone - CI included - has no
    curve. Tests that assert against real history are still worth having; they
    just cannot run until ``tqe data`` and ``tqe curve`` have. Marking them
    rather than deleting them keeps the assertions in the repo.
    """
    import pathlib

    curve = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed" / "curve.parquet"
    if curve.exists():
        return
    skip = pytest.mark.skip(reason="needs data/processed/curve.parquet; run `tqe data && tqe curve`")
    for item in items:
        if "needs_curve" in item.keywords:
            item.add_marker(skip)
