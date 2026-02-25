from __future__ import annotations

from pathlib import Path
from typing import Optional

from .format import format_cpp
from .flow import Program


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
