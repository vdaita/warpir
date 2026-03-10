from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Union, Dict

from .types import TypeRef


@dataclass(frozen=True)
class Value:
    """An SSA value: immutable, uniquely named, typed."""
    name: str
    type: TypeRef

    def __repr__(self) -> str:
        return f"%{self.name}: {self.type}"


class Op:
    """Base class for all IR operations."""
    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({fields})"
        
    def substitute(self, replacements: Dict[Value, Union[Value, int]]) -> 'Op':
        def replace(v):
            if isinstance(v, Value):
                return replacements.get(v, v)
            if isinstance(v, tuple):
                return tuple(replace(x) for x in v)
            return v
        return type(self)(**{k: replace(v) for k, v in vars(self).items()})


# ---------------------------------------------------------------------------
# Tile operations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZeroOp(Op):
    """Initialize a register tile to zero."""
    result: Value


@dataclass(frozen=True)
class TMALoadOp(Op):
    """Async TMA load from global memory into shared memory."""
    result: Value
    source: Value
    coords: tuple[Union[Value, int], ...]


@dataclass(frozen=True)
class WaitOp(Op):
    """Wait for pending TMA loads to complete."""
    values: tuple[Value, ...]


@dataclass(frozen=True)
class MMAOp(Op):
    """Matrix multiply-accumulate: result = a @ b + accum."""
    result: Value
    a: Value
    b: Value
    accum: Value


@dataclass(frozen=True)
class TMAStoreOp(Op):
    """Store a tile to global memory via TMA."""
    source: Value
    dest: Value
    coords: tuple[Union[Value, int], ...]


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class YieldOp(Op):
    """Terminates a loop body; passes values to the next iteration."""
    values: tuple[Value, ...]


@dataclass(frozen=True)
class IterArg:
    """A loop-carried value.

    On the first iteration, ``block_arg`` takes the value of ``init``.
    On subsequent iterations it receives the corresponding ``YieldOp`` value.
    """
    block_arg: Value
    init: Value


@dataclass(frozen=True)
class ForOp(Op):
    """For-loop with loop-carried dependencies (MLIR scf.for style).

    ``results`` correspond 1:1 with ``iter_args`` — after the loop
    completes, ``results[i]`` holds the final yielded value for
    ``iter_args[i]``.

    When ``tile_size`` is set the loop iterates over tiles:
    the effective bound is ``ceil(stop / tile_size)`` rather than
    ``stop`` directly.
    """
    induction_var: Value
    start: Union[Value, int]
    stop: Union[Value, int]
    step: Union[Value, int]
    iter_args: tuple[IterArg, ...]
    body: tuple[Op, ...]
    results: tuple[Value, ...]
    tile_size: Union[int, None] = None


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Kernel:
    name: str
    inputs: tuple[Value, ...] = field(default_factory=tuple)
    outputs: tuple[Value, ...] = field(default_factory=tuple)
    body: tuple[Op, ...] = field(default_factory=tuple)
