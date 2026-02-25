from typing import Optional, Sequence, Union
from abc import ABC, abstractmethod
from jinja2 import Environment
from .layouts import Var, SharedTileType
from .ops import TMALoadOp, WaitOp, ExpectBytesOp, ArriveOp, SizeBytesOfTypeOf, ExprLike, BinaryOp

class Stmt(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

_ENV = Environment(trim_blocks=True, lstrip_blocks=True)

class ForStmt(Stmt):
    def __init__(self, init: ExprLike, cond: ExprLike, step: ExprLike, body: Stmt):
        self.init = init
        self.cond = cond
        self.step = step
        self.body = body

    @classmethod
    def range(cls, var: str, start: Union[int, str], stop: Union[int, str], step: Union[int, str] = 1, body: Optional[Stmt] = None):
        init = f"int {var} = {start}"
        cond = f"{var} < {stop}"
        step_expr = f"{var} += {step}"
        return cls(init, cond, step_expr, body or NoStmt())

    def __str__(self) -> str:
        tmpl = _ENV.from_string(
            """for ({{ init }}; {{ cond }}; {{ step }}) {\n{% if body %}{{ body }}\n{% endif %}}"""
        )
        init = str(self.init).rstrip().rstrip(";")
        step = str(self.step).rstrip().rstrip(";")
        body = str(self.body).rstrip()
        return tmpl.render(init=init, cond=str(self.cond), step=step, body=body)

class WhileStmt(Stmt):
    def __init__(self, cond: ExprLike, body: Stmt):
        self.cond = cond
        self.body = body

    def __str__(self) -> str:
        tmpl = _ENV.from_string(
            """while ({{ cond }}) {\n{% if body %}{{ body }}\n{% endif %}}"""
        )
        body = str(self.body).rstrip()
        return tmpl.render(cond=str(self.cond), body=body)

class NoStmt(Stmt):
    def __init__(self):
        pass
    
    def __str__(self) -> str:
        return ""

class SeqStmt(Stmt):
    def __init__(self, stmts: Sequence[Stmt]):
        self.stmts = stmts
    
    def __str__(self):
        rendered = [str(stmt).rstrip() for stmt in self.stmts if str(stmt).strip()]
        return "\n".join(rendered)

class RawStmt(Stmt):
    def __init__(self, code: str):
        self.code = code

    def __str__(self) -> str:
        return self.code.rstrip()

class ExprStmt(Stmt):
    def __init__(self, expr: ExprLike):
        self.expr = expr

    def __str__(self) -> str:
        text = str(self.expr).rstrip()
        if not text:
            return ""
        if text.endswith(";"):
            return text
        return f"{text};"

class AssignStmt(Stmt):
    def __init__(self, lhs: ExprLike, rhs: ExprLike):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        tmpl = _ENV.from_string("{{ lhs }} = {{ rhs }};")
        return tmpl.render(lhs=str(self.lhs), rhs=str(self.rhs))

class DeclStmt(Stmt):
    def __init__(self, var: Var, init: Optional[ExprLike] = None):
        self.var = var
        self.init = init

    def __str__(self) -> str:
        if self.init is None:
            return self.var.define()
        tmpl = _ENV.from_string("{{ type }} {{ name }} = {{ init }};")
        return tmpl.render(type=self.var.var_type, name=self.var.name, init=str(self.init))

class SharedAllocStmt(Stmt):
    def __init__(self, name: str, tile_type, allocator: str = "al", count: Optional[ExprLike] = None):
        self.name = name
        self.tile_type = tile_type
        self.allocator = allocator
        self.count = count

    def __str__(self) -> str:
        if self.count is None:
            tmpl = _ENV.from_string("{{ type }} (&{{ name }}) = {{ alloc }}.allocate<{{ type }}>();")
            return tmpl.render(type=self.tile_type, name=self.name, alloc=self.allocator)
        tmpl = _ENV.from_string(
            "{{ type }} (&{{ name }})[{{ count }}] = {{ alloc }}.allocate<{{ type }}, {{ count }}>();"
        )
        return tmpl.render(type=self.tile_type, name=self.name, alloc=self.allocator, count=str(self.count))

class IfStmt(Stmt):
    def __init__(self, cond: ExprLike, then_stmt: Stmt, else_stmt: Optional[Stmt] = None):
        self.cond = cond
        self.then_stmt = then_stmt
        self.else_stmt = else_stmt

    def __str__(self) -> str:
        tmpl = _ENV.from_string(
            """if ({{ cond }}) {\n{% if then_body %}{{ then_body }}\n{% endif %}}{% if else_body %} else {\n{{ else_body }}\n}{% endif %}"""
        )
        then_body = str(self.then_stmt).rstrip()
        else_body = str(self.else_stmt).rstrip() if self.else_stmt is not None else ""
        return tmpl.render(cond=str(self.cond), then_body=then_body, else_body=else_body)

class Warpgroup(Stmt): # TODO: implement a full, functional warpgroup object that allows for splitting tasks
    def __init__(self, warp_id: int, wg_vars: Sequence[Var], stmt: Stmt):
        self.id = warp_id
        self.wg_vars = wg_vars
        self.stmt = stmt

    def __str__(self) -> str:
        tmpl = _ENV.from_string(
            """if (warpgroupid == {{ wid }}) {\n{% if body %}{{ body }}\n{% endif %}}"""
        )
        vars_block = "\n".join([v.define() for v in self.wg_vars]).rstrip()
        body_stmt = self.stmt
        if vars_block:
            body_stmt = SeqStmt([RawStmt(vars_block), self.stmt])
        body = str(body_stmt).rstrip()
        return tmpl.render(wid=self.id, body=body)

class Tile:
    def __init__(
        self,
        name: str,
        tile_type,
        use_semaphores: bool = True,
        full_sem: Optional[str] = None,
        empty_sem: Optional[str] = None,
        allocator: str = "al",
    ):
        self.name = name
        self.tile_type = tile_type
        self.use_semaphores = use_semaphores
        self.full_sem = full_sem or f"full_{name}"
        self.empty_sem = empty_sem or f"empty_{name}"
        self.allocator = allocator

    def _is_shared(self) -> bool:
        return isinstance(self.tile_type, SharedTileType)

    def declare(self) -> Stmt:
        if self._is_shared():
            return SharedAllocStmt(self.name, self.tile_type, allocator=self.allocator)
        return DeclStmt(Var(self.name, self.tile_type))

    def declare_semaphores(self, shared: bool = True) -> Stmt:
        if not self.use_semaphores:
            return NoStmt()
        prefix = "__shared__ " if shared else ""
        return RawStmt(f"{prefix}semaphore {self.full_sem}, {self.empty_sem};")

    def init_semaphores(self, full_init: Union[str, int], empty_init: Union[str, int]) -> Stmt:
        if not self.use_semaphores:
            return NoStmt()
        return SeqStmt([
            RawStmt(f"init_semaphore({self.full_sem}, {full_init}, 1);"),
            RawStmt(f"init_semaphore({self.empty_sem}, {empty_init}, 0);"),
        ])

    def produce(
        self,
        src: ExprLike,
        coord: ExprLike,
        phase: Optional[ExprLike] = None,
        expect_bytes: Optional[ExprLike] = None,
        callee: str = "tma::load_async",
    ) -> Stmt:
        if not self.use_semaphores:
            return ExprStmt(TMALoadOp(self.name, src, coord, None, callee=callee))
        seq: list[Stmt] = []
        if phase is not None:
            seq.append(ExprStmt(WaitOp(self.empty_sem, phase)))
        if expect_bytes is None:
            expect_bytes = SizeBytesOfTypeOf(self.name)
        if expect_bytes is not None:
            seq.append(ExprStmt(ExpectBytesOp(self.full_sem, expect_bytes)))
        seq.append(ExprStmt(TMALoadOp(self.name, src, coord, self.full_sem, callee=callee)))
        return SeqStmt(seq)

    def consume(
        self,
        phase: Optional[ExprLike],
        body: Stmt,
        arrive: bool = True,
        arrive_count: ExprLike = 1,
    ) -> Stmt:
        if not self.use_semaphores:
            return body
        seq: list[Stmt] = []
        if phase is not None:
            seq.append(ExprStmt(WaitOp(self.full_sem, phase)))
        seq.append(body)
        if arrive:
            seq.append(ExprStmt(ArriveOp(self.empty_sem, arrive_count)))
        return SeqStmt(seq)

class TileQueue:
    def __init__(self, name: str, tiles: Sequence[Tile], full_sem: Optional[str] = None, empty_sem: Optional[str] = None):
        self.name = name
        self.tiles = tiles
        self.full_sem = full_sem or f"full_{name}"
        self.empty_sem = empty_sem or f"empty_{name}"
        for t in self.tiles:
            t.full_sem = self.full_sem
            t.empty_sem = self.empty_sem

    def declare_semaphores(self, shared: bool = True) -> Stmt:
        prefix = "__shared__ " if shared else ""
        return RawStmt(f"{prefix}semaphore {self.full_sem}, {self.empty_sem};")

    def init_semaphores(self, full_init: Union[str, int], empty_init: Union[str, int]) -> Stmt:
        return SeqStmt([
            RawStmt(f"init_semaphore({self.full_sem}, {full_init}, 1);"),
            RawStmt(f"init_semaphore({self.empty_sem}, {empty_init}, 0);"),
        ])

    def produce(
        self,
        loads: Sequence[tuple[Tile, ExprLike, ExprLike]],
        phase: ExprLike,
        expect_bytes: Optional[ExprLike] = None,
    ) -> Stmt:
        if expect_bytes is None:
            sizes = [SizeBytesOfTypeOf(t.name) for t, _, _ in loads]
            if sizes:
                total: ExprLike = sizes[0]
                for s in sizes[1:]:
                    total = BinaryOp("+", total, s)
                expect_bytes = total
        seq: list[Stmt] = [ExprStmt(WaitOp(self.empty_sem, phase))]
        if expect_bytes is not None:
            seq.append(ExprStmt(ExpectBytesOp(self.full_sem, expect_bytes)))
        for t, src, coord in loads:
            seq.append(ExprStmt(TMALoadOp(t.name, src, coord, self.full_sem)))
        return SeqStmt(seq)

    def consume(self, phase: ExprLike, body: Stmt, arrive: bool = True, arrive_count: ExprLike = 1) -> Stmt:
        seq: list[Stmt] = [ExprStmt(WaitOp(self.full_sem, phase)), body]
        if arrive:
            seq.append(ExprStmt(ArriveOp(self.empty_sem, arrive_count)))
        return SeqStmt(seq)

KITTENS_TEMPLATE = """
#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include "kittens.cuh"

using namespace kittens;

struct kernel_globals {
    {%- for var in kernel_vars %}
    {{ var.define() }}
    {%- endfor %}
};

__global__ void kernel(const __grid_constant__ kernel_globals g) {
    {{ kernel_stmt }}
}
"""
class Program:
    def __init__(self, input_vars: Sequence[Var], kernel_vars: Sequence[Var], kernel_stmt: Stmt):
        self.input_vars = input_vars
        self.kernel_vars = kernel_vars
        self.kernel_stmt = kernel_stmt
    
    def __str__(self):
        t = _ENV.from_string(KITTENS_TEMPLATE)
        return t.render(
            kernel_vars=self.kernel_vars,
            kernel_stmt=self.kernel_stmt
        )
