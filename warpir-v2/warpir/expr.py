from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

class Expr:
    pass


@dataclass(frozen=True)
class VarExpr(Expr):
    name: str


@dataclass(frozen=True)
class LiteralExpr(Expr):
    value: int | float | str | bool


@dataclass(frozen=True)
class BinaryExpr(Expr):
    lhs: Expr
    op: str
    rhs: Expr


@dataclass(frozen=True)
class CallExpr(Expr):
    callee: str
    args: Sequence[Expr] = field(default_factory=tuple)


@dataclass(frozen=True)
class RawExpr(Expr):
    code: str

