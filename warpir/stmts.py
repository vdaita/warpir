from __future__ import annotations

from copy import deepcopy
from jinja2 import Template
from warpir.layouts import SharedVecType
from typing import Optional, Sequence, Union, List, Dict, TYPE_CHECKING
from abc import ABC, abstractmethod
from jinja2 import Environment
from dataclasses import dataclass
from .layouts import SharedTileType, ScalarType, VarType, SharedSemaphoreType, GlobalType
from enum import Enum

if TYPE_CHECKING:
    from warpir.helpers import Tile

class Level(str, Enum):
    warpgroup = "warpgroup"
    block = "block"


class Expr(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    def to_stmt(self):
        return ExprStmt(self)

    def replace_vars(self, replacements: Dict['Var', 'Expr']) -> 'Expr':
        return deepcopy(self)

    def replace_var(self, var: 'Var', replacement: 'Expr') -> 'Expr':
        return self.replace_vars({var: replacement})

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

    def replace_vars(self, replacements: Dict['Var', 'Expr']) -> 'Stmt':
        return deepcopy(self)

    def replace_var(self, var: 'Var', replacement: 'Expr') -> 'Stmt':
        return self.replace_vars({var: replacement})

    def remove_var(self, var: 'Var') -> 'Stmt':
        return deepcopy(self)

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

    def replace_vars(self, replacements: Dict['Var', Expr]) -> Expr:
        expr = deepcopy(self)
        expr.inputs = [inp.replace_vars(replacements) for inp in expr.inputs]
        return expr

class ExprStmt(Stmt):
    def __init__(self, call: Expr):
        self.call = call
    
    def __str__(self):
        return f"{self.call};"

    def replace_vars(self, replacements: Dict['Var', Expr]) -> 'ExprStmt':
        stmt = deepcopy(self)
        stmt.call = stmt.call.replace_vars(replacements)
        return stmt

class BinaryOp(Expr):
    def __init__(self, a: Expr, b: Expr, op_type: str):
        self.a = a 
        self.b = b
        self.op_type = op_type

    def __str__(self):
        return f"({self.a} {self.op_type} {self.b})"

    def replace_vars(self, replacements: Dict['Var', Expr]) -> Expr:
        expr = deepcopy(self)
        expr.a = expr.a.replace_vars(replacements)
        expr.b = expr.b.replace_vars(replacements)
        return expr

class ForStmt(Stmt):
    def __init__(self, init: Expr, cond: Expr, step: Expr, body: Stmt, inputs: List[Var], yields: List[Var]):
        self.init = init
        self.cond = cond
        self.step = step
        self.body = body
        
        self.inputs = inputs
        self.yields = yields

    def __str__(self) -> str:
        tmpl = _ENV.from_string(
            """for ({{ init }}; {{ cond }}; {{ step }}) {\n{% if body %}{{ body }}\n{% endif %}}"""
        )
        init = str(self.init).rstrip().rstrip(";")
        step = str(self.step).rstrip().rstrip(";")
        body = str(self.body).rstrip()
        return tmpl.render(init=init, cond=str(self.cond), step=step, body=body)

    def replace_vars(self, replacements: Dict['Var', Expr]) -> 'ForStmt':
        stmt = deepcopy(self)
        stmt.init = stmt.init.replace_vars(replacements)
        stmt.cond = stmt.cond.replace_vars(replacements)
        stmt.step = stmt.step.replace_vars(replacements)
        stmt.body = stmt.body.replace_vars(replacements)
        return stmt

    def remove_var(self, var: 'Var') -> 'ForStmt':
        stmt = deepcopy(self)
        stmt.body = stmt.body.remove_var(var)
        return stmt

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

    def replace_vars(self, replacements: Dict['Var', Expr]) -> 'SeqStmt':
        stmt = deepcopy(self)
        stmt.stmts = [child.replace_vars(replacements) for child in stmt.stmts]
        return stmt

    def remove_var(self, var: 'Var') -> 'SeqStmt':
        stmt = deepcopy(self)
        filtered: List[Stmt] = []
        for child in stmt.stmts:
            if isinstance(child, DeclStmt) and child.var == var:
                continue
            filtered.append(child.remove_var(var))
        stmt.stmts = filtered
        return stmt

class RawExpr(Expr):
    def __init__(self, code) -> None:
        self.code = str(code)
    
    def __str__(self) -> str:
        return self.code.rstrip()

    def replace_vars(self, replacements: Dict['Var', Expr]) -> Expr:
        return deepcopy(self)
zero, one = RawExpr(0), RawExpr(1)

class RawStmt(Stmt):
    def __init__(self, code: str):
        self.code = str(code)

    def __str__(self) -> str:
        return self.code.rstrip()

    def replace_vars(self, replacements: Dict['Var', Expr]) -> 'RawStmt':
        return deepcopy(self)

class AssignExpr(Expr):
    def __init__(self, lhs: Expr, rhs: Expr):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        tmpl = _ENV.from_string("{{ lhs }} = {{ rhs }}")
        return tmpl.render(lhs=str(self.lhs), rhs=str(self.rhs))

    def replace_vars(self, replacements: Dict['Var', Expr]) -> Expr:
        expr = deepcopy(self)
        expr.lhs = expr.lhs.replace_vars(replacements)
        expr.rhs = expr.rhs.replace_vars(replacements)
        return expr

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

    def replace_vars(self, replacements: Dict['Var', Expr]) -> 'IfStmt':
        stmt = deepcopy(self)
        stmt.cond = stmt.cond.replace_vars(replacements)
        stmt.then_stmt = stmt.then_stmt.replace_vars(replacements)
        if stmt.else_stmt is not None:
            stmt.else_stmt = stmt.else_stmt.replace_vars(replacements)
        return stmt

    def remove_var(self, var: 'Var') -> 'IfStmt':
        stmt = deepcopy(self)
        stmt.then_stmt = stmt.then_stmt.remove_var(var)
        if stmt.else_stmt is not None:
            stmt.else_stmt = stmt.else_stmt.remove_var(var)
        return stmt

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

    def replace_vars(self, replacements: Dict['Var', Expr]) -> 'DeclStmt':
        return self

class Var(Expr):
    def __init__(self, name: str, var_type: VarType, parent: Optional[KernelGlobals] = None, annotations: List[str] = []):
        self.name = name
        self.var_type = var_type
        self.parent = parent
        self.annotations = annotations # helps keep track of attributes like this being async (which will make it easier for me to know that I need to add semaphores and waiting to this at the final stage)

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

    def replace_vars(self, replacements: Dict['Var', Expr]) -> Expr:
        return replacements.get(self, deepcopy(self))

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

    def replace_vars(self, replacements: Dict['Var', Expr]) -> Expr:
        coord = deepcopy(self)
        coord.elements = [element.replace_vars(replacements) for element in coord.elements]
        return coord

class SizeBytesExpr(Expr):
    def __init__(self, expr: Expr):
        self.expr = expr

    def __str__(self) -> str:
        return f"size_bytes<typeof({self.expr})>"
    
    def replace_vars(self, replacements: Dict['Var', Expr]) -> Expr:
        expr = deepcopy(self)
        expr.expr = expr.expr.replace_vars(replacements)
        return expr
    
class TileLoadOp(Stmt):
    def __init__(self, tile: 'Tile', source: Var, coord: Coord, async_load: bool = False):
        self.tile = tile
        self.source = source
        self.coord = coord
        self.async_load = async_load

    def __str__(self) -> str:
        tag = "async " if self.async_load else ""
        return f"{tag}TileLoad({self.tile}, {self.source}, {self.coord})"

    def replace_vars(self, replacements: Dict['Var', Expr]) -> 'TileLoadOp':
        stmt = deepcopy(self)
        stmt.tile = stmt.tile.replace_vars(replacements)
        stmt.source = stmt.source.replace_vars(replacements)
        stmt.coord = stmt.coord.replace_vars(replacements)
        return stmt

    def lower(self) -> Stmt:
        if self.async_load:
            raise NotImplementedError("Async tile loads need TileGroup analysis first")
        return self.tile.load_global(self.source, self.coord)
    
