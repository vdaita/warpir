from __future__ import annotations

from functools import singledispatchmethod
from typing import Sequence

from warpir_v2.ir import CallOp, ExprArg, ForOp, Kernel, Op, Param, SeqOp, Value
from warpir_v2.ir import SharedTileType


class ThunderKittensLowerer:
    def __init__(self, indent: str = "  "):
        self.indent = indent

    def lower(self, kernel: Kernel) -> str:
        includes = (
            "\n#include \"kittens.cuh\"\n"
            "#include <chrono>\n"
            "#include <cuda_runtime.h>\n"
            "#include <iostream>\n"
            "#include <random>\n\n"
            "using namespace kittens;\n"
        )
        globals_struct = self._lower_globals_struct(kernel.params)
        locals_to_declare = self._collect_local_values(kernel)
        local_decl_lines = [self._declare_local(v, depth=1) for v in locals_to_declare]
        op_lines = self._lower_op(kernel.body, depth=1)
        body = "\n".join(local_decl_lines + [op_lines] if local_decl_lines else [op_lines])
        return (
            f"{includes}\n"
            f"{globals_struct}\n\n"
            f"__global__ void {kernel.name}(const __grid_constant__ global_vars globals) {{\n"
            f"{body}\n"
            "}\n"
        )

    def _lower_globals_struct(self, params: Sequence[Param]) -> str:
        lines = [f"{self.indent}{p.type_ref} {p.name};" for p in params]
        return "struct global_vars {\n" + "\n".join(lines) + "\n};"

    def _declare_local(self, value: Value, depth: int) -> str:
        prefix = "__shared__ " if isinstance(value.type_ref, SharedTileType) else ""
        return f"{self.indent * depth}{prefix}{value.type_ref} {value.name};"

    def _collect_local_values(self, kernel: Kernel) -> list[Value]:
        param_names = {p.name for p in kernel.params}
        seen: set[str] = set()
        ordered: list[Value] = []

        def add_value(v: Value, loop_vars: set[str]) -> None:
            if v.name in param_names or v.name in loop_vars or v.name in seen:
                return
            seen.add(v.name)
            ordered.append(v)

        def walk(op: Op, loop_vars: set[str]) -> None:
            if isinstance(op, SeqOp):
                for child in op.ops:
                    walk(child, loop_vars)
                return
            if isinstance(op, CallOp):
                for arg in op.args:
                    if isinstance(arg, Value):
                        add_value(arg, loop_vars)
                for result in op.results:
                    add_value(result, loop_vars)
                return
            if isinstance(op, ForOp):
                inner_loop_vars = set(loop_vars)
                inner_loop_vars.add(op.iter_value.name)
                walk(op.body, inner_loop_vars)
                return

        walk(kernel.body, set())
        return ordered

    def _arg_text(self, arg: Value | ExprArg) -> str:
        if isinstance(arg, Value):
            return arg.name
        return arg.code

    @singledispatchmethod
    def _lower_op(self, op: Op, depth: int) -> str:
        raise TypeError(f"Unsupported op: {type(op).__name__}")

    @_lower_op.register
    def _(self, op: SeqOp, depth: int) -> str:
        return "\n".join(self._lower_op(child, depth) for child in op.ops)

    @_lower_op.register
    def _(self, op: CallOp, depth: int) -> str:
        args = ", ".join(self._arg_text(arg) for arg in op.args)
        if len(op.results) == 0:
            return f"{self.indent * depth}{op.callee}({args});"
        if len(op.results) == 1:
            return f"{self.indent * depth}{op.results[0].name} = {op.callee}({args});"
        lhs = ", ".join(v.name for v in op.results)
        return f"{self.indent * depth}std::tie({lhs}) = {op.callee}({args});"

    @_lower_op.register
    def _(self, op: ForOp, depth: int) -> str:
        pad = self.indent * depth
        body = self._lower_op(op.body, depth + 1)
        return (
            f"{pad}for ({op.iter_value.type_ref} {op.iter_value.name} = {op.start.code}; "
            f"({op.iter_value.name} < {op.stop.code}); "
            f"{op.iter_value.name} = ({op.iter_value.name} + {op.step.code})) {{\n"
            f"{body}\n"
            f"{pad}}}"
        )
