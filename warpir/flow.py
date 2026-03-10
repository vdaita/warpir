from __future__ import annotations

from jinja2 import Template
from warpir.layouts import SharedVecType
from typing import Optional, Sequence, Union, List
from abc import ABC, abstractmethod
from jinja2 import Environment
from dataclasses import dataclass
from .layouts import SharedTileType, ScalarType, VarType, SharedSemaphoreType, GlobalType
from enum import Enum

class Level(str, Enum):
    warpgroup = "warpgroup"
    block = "block"


class Expr(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    def to_stmt(self):
        return ExprStmt(self)

    def __eq__(self, other):
        if not isinstance(other, Expr):
            return False
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

class Stmt(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    def __eq__(self, other):
        if not isinstance(other, Stmt):
            return False
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

_ENV = Environment(trim_blocks=True, lstrip_blocks=True)

class OpCall(Expr):
    def __init__(self, function_name: str, inputs: List[Expr]):
        self.function_name = function_name
        self.inputs = inputs
        
    def __str__(self):
        expr_list = ", ".join([str(op_input) for op_input in self.inputs])
        return f"{self.function_name}({expr_list})"

class ExprStmt(Stmt):
    def __init__(self, call: Expr):
        self.call = call
    
    def __str__(self):
        return f"{self.call};"

class BinaryOp(Expr):
    def __init__(self, a: Expr, b: Expr, op_type: str):
        self.a = a 
        self.b = b
        self.op_type = op_type

    def __str__(self):
        return f"({self.a} {self.op_type} {self.b})"

class ForStmt(Stmt):
    def __init__(self, init: Expr, cond: Expr, step: Expr, body: Stmt):
        self.init = init
        self.cond = cond
        self.step = step
        self.body = body

    def __str__(self) -> str:
        tmpl = _ENV.from_string(
            """for ({{ init }}; {{ cond }}; {{ step }}) {\n{% if body %}{{ body }}\n{% endif %}}"""
        )
        init = str(self.init).rstrip().rstrip(";")
        step = str(self.step).rstrip().rstrip(";")
        body = str(self.body).rstrip()
        return tmpl.render(init=init, cond=str(self.cond), step=step, body=body)

class NoStmt(Stmt):
    def __init__(self):
        pass
    
    def __str__(self) -> str:
        return ""

class SeqStmt(Stmt):
    def __init__(self, stmts: Sequence[Stmt]):
        self.stmts = []
        for s in stmts:
            if isinstance(s, Stmt):
                self.stmts.append(s)
            else:
                self.stmts.append(ExprStmt(s))
    
    def __str__(self):
        rendered = [str(stmt).rstrip() for stmt in self.stmts if str(stmt).strip()]
        return "\n".join(rendered)

class RawExpr(Expr):
    def __init__(self, code) -> None:
        self.code = str(code)
    
    def __str__(self) -> str:
        return self.code.rstrip()
zero, one = RawExpr(0), RawExpr(1)

class RawStmt(Stmt):
    def __init__(self, code: str):
        self.code = str(code)

    def __str__(self) -> str:
        return self.code.rstrip()

class ExprStmt(Stmt):
    def __init__(self, expr: Expr):
        self.expr = expr

    def __str__(self) -> str:
        text = str(self.expr).rstrip()
        if not text:
            return ""
        if text.endswith(";"):
            return text
        return f"{text};"

class AssignExpr(Expr):
    def __init__(self, lhs: Expr, rhs: Expr):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        tmpl = _ENV.from_string("{{ lhs }} = {{ rhs }}")
        return tmpl.render(lhs=str(self.lhs), rhs=str(self.rhs))

class IfStmt(Stmt):
    def __init__(self, cond: Expr, then_stmt: Stmt, else_stmt: Optional[Stmt] = None):
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

def lane0_if(stmt: Stmt) -> Stmt:
    return IfStmt(BinaryOp(RawExpr("0"), RawExpr("warpgroup::laneid()"), "=="), stmt)

def thread0_if(stmt: Stmt) -> Stmt:
    return IfStmt(BinaryOp(RawExpr("0"), RawExpr("threadIdx.x"), "=="), stmt)

class DeclStmt(Stmt):
    def __init__(self, var: Var):
        self.var = var

    def __str__(self):
        if type(self.var.var_type) == SharedTileType or type(self.var.var_type) == SharedVecType or type(self.var.var_type) == SharedSemaphoreType:
            return f"__shared__ {self.var.var_type} {self.var.name};"
        else:
            return f"{self.var.var_type} {self.var.name};"

class Var(Expr):
    def __init__(self, name: str, var_type: VarType, parent: Optional[KernelGlobals] = None):
        self.name = name
        self.var_type = var_type
        self.parent = parent

    def declare(self) -> Stmt:
        return DeclStmt(self)
    
    def __str__(self):
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name

    def __eq__(self, other):
        if not isinstance(other, Var):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(str(self))

class KernelGlobals():
    def __init__(self, name: str):
        self.vars = []
    
    def add_var(self, var: Var):
        var.parent = self
        self.vars.append(var)

    def __str__(self) -> str:
        return "globals"

class Coord(Expr):
    def __init__(self, elements: List[Expr]):
        self.elements = elements

    def __str__(self) -> str:
        elem_list = ", ".join([str(element) for element in self.elements])
        return "{" + elem_list + "}"

class SizeBytesExpr(Expr):
    def __init__(self, expr: Expr):
        self.expr = expr

    def __str__(self) -> str:
        return f"size_bytes<typeof({self.expr})>"

class Tile(Var):
    def __init__(
        self,
        name: str,
        var_type,
        use_semaphores: bool = False,
        num_consumers: int = 1
    ):
        super().__init__(name, var_type)
        self.use_semaphores = use_semaphores
        self.num_consumers = num_consumers
        self.var = Var(self.name, self.var_type)

        if use_semaphores:
            self._manager = TileGroup(name, [self], num_consumers)

    def __str__(self):
        return str(self.var)

    def __eq__(self, other):
        if not isinstance(other, Tile):
            return False
        return (self.var, self.use_semaphores, self.num_consumers) == (other.var, other.use_semaphores, other.num_consumers)

    def __hash__(self):
        return hash(str(self))

    def declare(self) -> Stmt:
        stmts: List[Stmt] = [self.var.declare()]
        if self.use_semaphores:
            stmts.append(self._manager.initialize())
        return SeqStmt(stmts)

    def load_global(self, src: Var, coord: Coord):
        assert type(src.var_type) == GlobalType
        return SeqStmt([
            ExprStmt(OpCall("kittens::warp::load", [self, src, coord])),
            ExprStmt(OpCall("__syncthreads", []))
        ])

    def warp_store_global(self, dst: Var, coord: Coord):
        assert type(dst.var_type) == GlobalType
        return ExprStmt(OpCall("kittens::warp::store", [self, dst, coord]))

    def warpgroup_store_global(self, dst: Var, coord: Coord):
        assert type(dst.var_type) == GlobalType
        return ExprStmt(OpCall("warpgroup::store", [self, dst, coord]))

    def load_shared(self, src: Var):
        assert type(src.var_type) == SharedTileType or type(self.var_type) == SharedVecType
        return SeqStmt([
            ExprStmt(OpCall("kittens::warp::load", [self, src])),
            ExprStmt(OpCall("__syncthreads", []))
        ])

    def async_load_global(self, src: Var, coord: Coord) -> Stmt:
        assert type(src.var_type) == GlobalType
        return self._manager.async_load_global([MemLoad(src, self, coord)])

    def wait_full(self, level):
        return self._manager.wait_full(level)

    def wait_empty(self, level):
        return self._manager.wait_empty(level)

    def arrive_empty(self):
        return self._manager.arrive_empty()

@dataclass
class MemLoad:
    source: Var
    dest: Tile
    coord: Coord

class TileGroup:
    def __init__(
        self,
        name: str,
        tiles: List[Tile],
        num_consumers: int = 1
    ):
        self.name = name
        self.tiles = tiles
        self.num_consumers = num_consumers

        self.full_sem = Var(f"full_{self.name}", SharedSemaphoreType())
        self.empty_sem = Var(f"empty_{self.name}", SharedSemaphoreType())
        self.full_tic = Var(f"tic_full_{self.name}", ScalarType("int"))
        self.empty_tic = Var(f"tic_empty_{self.name}", ScalarType("int"))

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, TileGroup):
            return False
        return (self.name, self.tiles, self.num_consumers) == (other.name, other.tiles, other.num_consumers)

    def __hash__(self):
        return hash(str(self))

    def initialize(self):
        stmts: List[Stmt] = []
        stmts.append(self.full_sem.declare())
        stmts.append(self.empty_sem.declare())
        stmts.append(self.full_tic.declare())
        stmts.append(self.empty_tic.declare())
        stmts.append(AssignExpr(self.full_tic, zero).to_stmt())
        stmts.append(AssignExpr(self.empty_tic, zero).to_stmt())
        stmts.append(
            thread0_if(ExprStmt(OpCall("init_semaphore", [self.full_sem, zero, one])))
        )
        stmts.append(
            thread0_if(ExprStmt(OpCall("init_semaphore", [self.empty_sem, RawExpr(f"{self.num_consumers}"), RawExpr("0")])))
        )
        stmts.append(
            thread0_if(ExprStmt(OpCall("arrive", [self.empty_sem, RawExpr(self.num_consumers)])))
        )
        return SeqStmt(stmts)
    
    def async_load_global(self, loads: List[MemLoad]) -> Stmt:
        num_bytes = SizeBytesExpr(self.tiles[0])
        for tile_idx in range(1, len(self.tiles)):
            num_bytes = BinaryOp(num_bytes, SizeBytesExpr(self.tiles[tile_idx]), "+")
        stmts: List[Stmt] = [
            ExprStmt(OpCall("tma::expect_bytes", [self.full_sem, num_bytes]))
        ] + [
            ExprStmt(OpCall("tma::load_async", [load.dest, load.source, load.coord, self.full_sem])) for load in loads if load.dest in self.tiles
        ]
        return lane0_if(SeqStmt(stmts))

    def wait_full(self, level):
        stmts = [
            ExprStmt(OpCall("wait", [self.full_sem, self.full_tic])),
            AssignExpr(self.full_tic, BinaryOp(self.full_tic, RawExpr("1"), "^")).to_stmt(),
        ]
        if level == Level.block:
            stmts.append(OpCall("__syncthreads", []).to_stmt())
        return SeqStmt(stmts)
    
    def wait_empty(self, level):
        stmts = [
            ExprStmt(OpCall("wait", [self.empty_sem, self.empty_tic])),
            AssignExpr(self.empty_tic, BinaryOp(self.empty_tic, RawExpr("1"), "^")).to_stmt()
        ]
        if level == Level.block:
            stmts.append(OpCall("__syncthreads", []).to_stmt())
        return SeqStmt(stmts)

    def arrive_empty(self):
        return lane0_if(OpCall("arrive", [self.empty_sem, RawExpr(1)]).to_stmt())
        
KITTENS_TEMPLATE = """
#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include "kittens.cuh"

using namespace kittens;

{{ constants }}

struct global_vars {
    {%- for var in kernel_vars %}
    {{ var.declare() }}
    {%- endfor %}
};

__global__ void kernel(const __grid_constant__ global_vars globals) {
    {{ kernel_stmt }}
}

{{ launch_code }}
"""
class Program:
    def __init__(self, kernel_vars: KernelGlobals, kernel_stmt: Stmt, 
                 constants: str = "", launch_code: str = ""
                ):
        self.kernel_vars = kernel_vars
        self.kernel_stmt = kernel_stmt
        self.constants = constants
        self.launch_code = launch_code
    
    def __str__(self):
        template = Template(KITTENS_TEMPLATE)
        return template.render(
            constants=self.constants,
            kernel_stmt=self.kernel_stmt,
            kernel_vars=self.kernel_vars.vars,
            launch_code=self.launch_code
        )
