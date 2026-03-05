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


class Stmt:
    pass

@dataclass(frozen=True)
class SeqStmt(Stmt):
    stmts: Sequence[Stmt] = field(default_factory=tuple)

@dataclass(frozen=True)
class DeclStmt(Stmt):
    name: str
    type_ref: TypeRef
    init: Expr | None = None

@dataclass(frozen=True)
class AssignStmt(Stmt):
    target: Expr
    value: Expr

@dataclass(frozen=True)
class ExprStmt(Stmt):
    expr: Expr

@dataclass(frozen=True)
class ForStmt(Stmt):
    init: Stmt | None
    cond: Expr | None
    step: Stmt | None
    body: SeqStmt

@dataclass(frozen=True)
class TypeRef:
    name: str


@dataclass(frozen=True)
class Param:
    name: str
    type_ref: TypeRef
    qualifier: str | None = None

@dataclass(frozen=True)
class Kernel:
    name: str
    params: Sequence[Param] = field(default_factory=tuple)
    body: SeqStmt = field(default_factory=SeqStmt)
