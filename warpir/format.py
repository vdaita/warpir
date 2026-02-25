from __future__ import annotations

from subprocess import PIPE, run


def format_cpp(code: str) -> str:
    """
    Format C++ code using clang-format.
    Requires the `clang-format` package/executable in the environment.
    """
    proc = run(
        ["clang-format"],
        input=code.encode("utf-8"),
        stdout=PIPE,
        stderr=PIPE,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"clang-format failed: {err.strip()}")
    return proc.stdout.decode("utf-8")
