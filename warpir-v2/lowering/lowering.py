from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod

from .ir import (
    AssignStmt,
    BinaryExpr,
    CallExpr,
    DeclStmt,
    Expr,
    ExprStmt,
    ForStmt,
    IfStmt,
    Kernel,
    LiteralExpr,
    Module,
    Param,
    RawExpr,
    ReturnStmt,
    SeqStmt,
    Stmt,
    VarExpr,
    WhileStmt,
)


@dataclass
class LoweringPipeline:
    """Minimal pass pipeline placeholder.

    Start with identity behavior so we can add normalization passes later.
    """

    def lower_module(self, module: Module) -> Module:
        return module


class ThunderKittensLowerer:
    """Lower v2 IR into ThunderKittens-flavored CUDA text.

    This is intentionally small: it emits one .cu translation unit with:
      - ThunderKittens include/preamble
      - one __global__ function per Kernel node
    """

    def __init__(self, indent: str = "    "):
        self.indent = indent

    def lower(self, module: Module) -> str:
        kernel_text = "\n\n".join(self._visit_kernel(k) for k in module.kernels)
        preamble = (
            '#include <cuda_runtime.h>\n'
            '#include "kittens.cuh"\n\n'
            "using namespace kittens;\n"
        )
        return f"{preamble}\n{kernel_text}\n"

    def _visit_kernel(self, kernel: Kernel) -> str:
        params = ", ".join(self._visit_param(p) for p in kernel.params)
        body = self._visit_seq(kernel.body, depth=1)
        return f"__global__ void {kernel.name}({params}) {{\n{body}\n}}"

    def _visit_param(self, p: Param) -> str:
        qual = f"{p.qualifier} " if p.qualifier else ""
        return f"{qual}{p.type_ref.name} {p.name}"

    def _visit_seq(self, seq: SeqStmt, depth: int) -> str:
        return "\n".join(self._visit_stmt(s, depth) for s in seq.stmts)

    @singledispatchmethod
    def _visit_stmt(self, stmt: Stmt, depth: int) -> str:
        raise TypeError(f"Unsupported stmt: {type(stmt).__name__}")

    @_visit_stmt.register
    def _(self, stmt: DeclStmt, depth: int) -> str:
        pad = self.indent * depth
        if stmt.init is None:
            return f"{pad}{stmt.type_ref.name} {stmt.name};"
        return f"{pad}{stmt.type_ref.name} {stmt.name} = {self._visit_expr(stmt.init)};"

    @_visit_stmt.register
    def _(self, stmt: AssignStmt, depth: int) -> str:
        pad = self.indent * depth
        return f"{pad}{self._visit_expr(stmt.target)} = {self._visit_expr(stmt.value)};"

    @_visit_stmt.register
    def _(self, stmt: ExprStmt, depth: int) -> str:
        pad = self.indent * depth
        return f"{pad}{self._visit_expr(stmt.expr)};"

    @_visit_stmt.register
    def _(self, stmt: ReturnStmt, depth: int) -> str:
        pad = self.indent * depth
        if stmt.value is None:
            return f"{pad}return;"
        return f"{pad}return {self._visit_expr(stmt.value)};"

    @_visit_stmt.register
    def _(self, stmt: IfStmt, depth: int) -> str:
        pad = self.indent * depth
        then_body = self._visit_seq(stmt.then_body, depth + 1)
        out = f"{pad}if ({self._visit_expr(stmt.cond)}) {{\n{then_body}\n{pad}}}"
        if stmt.else_body is not None:
            else_body = self._visit_seq(stmt.else_body, depth + 1)
            out += f" else {{\n{else_body}\n{pad}}}"
        return out

    @_visit_stmt.register
    def _(self, stmt: ForStmt, depth: int) -> str:
        pad = self.indent * depth
        init = self._visit_for_slot(stmt.init)
        cond = self._visit_expr(stmt.cond) if stmt.cond is not None else ""
        step = self._visit_for_slot(stmt.step)
        body = self._visit_seq(stmt.body, depth + 1)
        return f"{pad}for ({init}; {cond}; {step}) {{\n{body}\n{pad}}}"

    @_visit_stmt.register
    def _(self, stmt: WhileStmt, depth: int) -> str:
        pad = self.indent * depth
        body = self._visit_seq(stmt.body, depth + 1)
        return f"{pad}while ({self._visit_expr(stmt.cond)}) {{\n{body}\n{pad}}}"

    @_visit_stmt.register
    def _(self, stmt: SeqStmt, depth: int) -> str:
        return self._visit_seq(stmt, depth)

    def _visit_for_slot(self, slot: Stmt | None) -> str:
        if slot is None:
            return ""
        text = self._visit_stmt(slot, depth=0).strip()
        return text[:-1] if text.endswith(";") else text

    @singledispatchmethod
    def _visit_expr(self, expr: Expr | None) -> str:
        raise TypeError(f"Unsupported expr: {type(expr).__name__}")

    @_visit_expr.register
    def _(self, expr: type(None)) -> str:
        if expr is None:
            return ""
        return ""

    @_visit_expr.register
    def _(self, expr: VarExpr) -> str:
        return expr.name

    @_visit_expr.register
    def _(self, expr: LiteralExpr) -> str:
        if isinstance(expr.value, str):
            return expr.value
        if expr.value is True:
            return "true"
        if expr.value is False:
            return "false"
        return str(expr.value)

    @_visit_expr.register
    def _(self, expr: BinaryExpr) -> str:
        return f"({self._visit_expr(expr.lhs)} {expr.op} {self._visit_expr(expr.rhs)})"

    @_visit_expr.register
    def _(self, expr: CallExpr) -> str:
        args = ", ".join(self._visit_expr(arg) for arg in expr.args)
        return f"{expr.callee}({args})"

    @_visit_expr.register
    def _(self, expr: RawExpr) -> str:
        return expr.code
