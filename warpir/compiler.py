from __future__ import annotations

from pathlib import Path
from typing import Optional
from subprocess import PIPE, run

from .flow import Program


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

def try_format_cpp(code: str) -> str:
    try:
        return format_cpp(code)
    except Exception:
        return code


def emit_cpp(program: Program, pretty: bool = True) -> str:
    code = str(program)
    return try_format_cpp(code) if pretty else code


def write_cpp(program: Program, path: str | Path, pretty: bool = True) -> Path:
    path = Path(path)
    path.write_text(emit_cpp(program, pretty=pretty))
    return path
