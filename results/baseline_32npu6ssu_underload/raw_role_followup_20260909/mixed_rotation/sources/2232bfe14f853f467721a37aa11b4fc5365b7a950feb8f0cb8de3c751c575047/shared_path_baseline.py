"""Baseline: every I/O stays on Path0; no pressure query or CIR update."""


def baseline_path_ids(io_count, path_id=0):
    return (path_id,) * io_count


if __name__ == "__main__":
    print(baseline_path_ids(8))
