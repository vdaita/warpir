from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .types import TypeRef


class Op:
    pass


@dataclass(frozen=True)
class SeqOp(Op):
    ops: Sequence[Op] = field(default_factory=tuple)


@dataclass(frozen=True)
class Value:
    name: str
    type_ref: TypeRef


@dataclass(frozen=True)
class ExprArg:
    code: str
    type_ref: TypeRef


@dataclass(frozen=True)
class CallOp(Op):
    callee: str
    args: Sequence[Value | ExprArg] = field(default_factory=tuple)
    results: Sequence[Value] = field(default_factory=tuple)


@dataclass(frozen=True)
class ForOp(Op):
    iter_value: Value
    start: ExprArg
    stop: ExprArg
    step: ExprArg
    body: SeqOp


@dataclass(frozen=True)
class Param:
    name: str
    type_ref: TypeRef


@dataclass(frozen=True)
class Kernel:
    name: str
    params: Sequence[Param] = field(default_factory=tuple)
    body: SeqOp = field(default_factory=SeqOp)
