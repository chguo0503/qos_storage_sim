"""Actual package sources used by input and simulator experiment runners."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source_files() -> tuple[str, ...]:
    """Return sorted project-relative paths, including every nested module.

    Discover at call time so new policy/input modules are covered too. The
    historical ``data`` catalog remains at the repository root. Results,
    traces, generated caches and documentation are not simulator source code;
    individual experiment extensions record their own provenance separately.
    """
    paths = {ROOT / "data"}
    for package in ("simulator", "inputs"):
        paths.update((ROOT / package).rglob("*.py"))
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"missing experiment source files: {missing}")
    return tuple(sorted(path.relative_to(ROOT).as_posix() for path in paths))
