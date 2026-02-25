from typing import Union, List, Optional, Protocol

from .layouts import SharedTileType, RegTileType

class Expr(Protocol):
    def __str__(self) -> str: ...

ExprLike = Union["Expr", str, int, float]

def _fmt(value: ExprLike) -> str:
    return str(value)

_INFIX_OPS = {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "==", "!=", "<", ">", "<=", ">="}

class BuiltinExpr:
    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text

class Symbol:
    def __init__(self, name: str):
        self.name = name

    def __getattr__(self, field: str):
        return FieldRef(self, field)

    def __str__(self) -> str:
        return self.name

class FieldRef:
    def __init__(self, base, field: str):
        self.base = base
        self.field = field

    def __str__(self) -> str:
        return f"{_fmt(self.base)}.{self.field}"

class Coord:
    def __init__(self, *items):
        self.items = items

    def __str__(self) -> str:
        return "{" + ", ".join(_fmt(i) for i in self.items) + "}"

class SizeBytesOfTypeOf:
    def __init__(self, var):
        self.var = var

    def __str__(self) -> str:
        return f"size_bytes<typeof({_fmt(self.var)})>"


class BlockIdx:
    x = BuiltinExpr("blockIdx.x")
    y = BuiltinExpr("blockIdx.y")
    z = BuiltinExpr("blockIdx.z")

class ThreadIdx:
    x = BuiltinExpr("threadIdx.x")
    y = BuiltinExpr("threadIdx.y")
    z = BuiltinExpr("threadIdx.z")

class WarpIdExpr(BuiltinExpr):
    def __init__(self):
        super().__init__("kittens::warpid()")

class WarpGroupIdExpr(BuiltinExpr):
    def __init__(self):
        super().__init__("kittens::warpid() / 4")

WarpId = WarpIdExpr()
WarpGroupId = WarpGroupIdExpr()
LaneId = BuiltinExpr("warpgroup::laneid()")

class UnaryOp:
    def __init__(self, op: str, arg: ExprLike):
        self.op = op
        self.arg = arg

    def __str__(self) -> str:
        return f"{self.op}({_fmt(self.arg)})"

class BinaryOp:
    def __init__(self, op: str, lhs: ExprLike, rhs: ExprLike):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        if self.op in _INFIX_OPS:
            return f"({_fmt(self.lhs)} {self.op} {_fmt(self.rhs)})"
        return f"{self.op}({_fmt(self.lhs)}, {_fmt(self.rhs)})"

class ConstantOp:
    def __init__(self, value: Union[int, float, str]):
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

class ZeroOp(ConstantOp):
    def __init__(self):
        super().__init__(0)

class LoadOp(BinaryOp):
    def __init__(self, dst: ExprLike, src: ExprLike, idx: Optional[ExprLike] = None, callee: str = "warpgroup::load"):
        self.callee = callee
        self.dst = dst
        self.src = src
        self.idx = idx

    def __str__(self) -> str:
        if self.idx is None:
            return f"{self.callee}({_fmt(self.dst)}, {_fmt(self.src)})"
        return f"{self.callee}({_fmt(self.dst)}, {_fmt(self.src)}, {_fmt(self.idx)})"

class TMALoadOp:
    def __init__(self, dst: ExprLike, src: ExprLike, coord: ExprLike, semaphore: Optional[ExprLike] = None, callee: str = "tma::load_async"):
        self.callee = callee
        self.dst = dst
        self.src = src
        self.coord = coord
        self.semaphore = semaphore

    def __str__(self) -> str:
        if self.semaphore is None:
            return f"{self.callee}({_fmt(self.dst)}, {_fmt(self.src)}, {_fmt(self.coord)})"
        return f"{self.callee}({_fmt(self.dst)}, {_fmt(self.src)}, {_fmt(self.coord)}, {_fmt(self.semaphore)})"

class MMAOp:
    def __init__(self, accum: ExprLike, a: ExprLike, b: ExprLike, callee: str = "warpgroup::mma_AB"):
        self.callee = callee
        self.accum = accum
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return f"{self.callee}({_fmt(self.accum)}, {_fmt(self.a)}, {_fmt(self.b)})"

class StoreOp: # going to be just like the LoadOp
    def __init__(self, dst: ExprLike, src: ExprLike, idx: Optional[ExprLike] = None, callee: str = "warpgroup::store"):
        self.callee = callee
        self.dst = dst
        self.src = src
        self.idx = idx

    def __str__(self) -> str:
        if self.idx is None:
            return f"{self.callee}({_fmt(self.dst)}, {_fmt(self.src)})"
        return f"{self.callee}({_fmt(self.dst)}, {_fmt(self.src)}, {_fmt(self.idx)})"

class TMAStoreOp:
    def __init__(self, dst: ExprLike, src: ExprLike, coord: ExprLike, semaphore: Optional[ExprLike] = None, callee: str = "tma::store_async"):
        self.callee = callee
        self.dst = dst
        self.src = src
        self.coord = coord
        self.semaphore = semaphore

    def __str__(self) -> str:
        if self.semaphore is None:
            return f"{self.callee}({_fmt(self.dst)}, {_fmt(self.src)}, {_fmt(self.coord)})"
        return f"{self.callee}({_fmt(self.dst)}, {_fmt(self.src)}, {_fmt(self.coord)}, {_fmt(self.semaphore)})"

class WaitOp:
    def __init__(self, sem: ExprLike, phase: ExprLike, callee: str = "wait"):
        self.callee = callee
        self.sem = sem
        self.phase = phase

    def __str__(self) -> str:
        return f"{self.callee}({_fmt(self.sem)}, {_fmt(self.phase)})"

class ArriveOp:
    def __init__(self, sem: ExprLike, count: ExprLike = 1, callee: str = "arrive"):
        self.callee = callee
        self.sem = sem
        self.count = count

    def __str__(self) -> str:
        return f"{self.callee}({_fmt(self.sem)}, {_fmt(self.count)})"

class ExpectBytesOp:
    def __init__(self, sem: ExprLike, nbytes: ExprLike, callee: str = "tma::expect_bytes"):
        self.callee = callee
        self.sem = sem
        self.nbytes = nbytes

    def __str__(self) -> str:
        return f"{self.callee}({_fmt(self.sem)}, {_fmt(self.nbytes)})"

class MMAWaitOp:
    def __init__(self, callee: str = "warpgroup::mma_async_wait"):
        self.callee = callee

    def __str__(self) -> str:
        return f"{self.callee}()"
