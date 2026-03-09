from __future__ import annotations

from warpir_v2.ir.ops import (
    ForOp,
    Kernel,
    MMAOp,
    Op,
    TMALoadOp,
    TMAStoreOp,
    Value,
    WaitOp,
    YieldOp,
    ZeroOp,
)
from warpir_v2.ir.types import (
    GPUType,
    GlobalType,
    IntType,
    RegTileType,
    SharedTileType,
    TileLayout,
    TypeRef,
)


class ThunderKittensLowerer:
    BUILTINS = frozenset({
        "blockIdx.x", "blockIdx.y", "blockIdx.z",
        "threadIdx.x", "threadIdx.y", "threadIdx.z",
        "warpgroupid",
    })

    def __init__(self):
        self._reset()

    def _reset(self):
        self._aliases: dict[str, str] = {}
        self._globals: set[str] = set()
        self._sem_counter = 0
        self._wait_group_sems: dict[frozenset[str], str] = {}
        self._load_to_sem: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lower(self, kernel: Kernel) -> str:
        self._reset()

        for v in (*kernel.inputs, *kernel.outputs):
            self._globals.add(v.name)

        self._build_aliases(kernel.body)
        self._analyze_semaphores(kernel.body)

        decls = self._collect_declarations(kernel)
        sem_decls = self._emit_sem_decls()
        body_lines = self._lower_ops(kernel.body, depth=1)

        return self._assemble(kernel, decls, sem_decls, body_lines)

    # ------------------------------------------------------------------
    # Alias resolution (SSA iter_args → single mutable variable)
    # ------------------------------------------------------------------

    def _build_aliases(self, ops: tuple[Op, ...]) -> None:
        for op in ops:
            if not isinstance(op, ForOp):
                continue
            for i, ia in enumerate(op.iter_args):
                self._aliases[ia.init.name] = ia.block_arg.name
                if i < len(op.results):
                    self._aliases[op.results[i].name] = ia.block_arg.name
            for body_op in op.body:
                if isinstance(body_op, YieldOp):
                    for i, yv in enumerate(body_op.values):
                        if i < len(op.iter_args):
                            self._aliases[yv.name] = op.iter_args[i].block_arg.name
            self._build_aliases(op.body)

    def _resolve(self, name: str) -> str:
        visited: set[str] = set()
        while name in self._aliases and name not in visited:
            visited.add(name)
            name = self._aliases[name]
        return name

    def _val_name(self, v: Value) -> str:
        resolved = self._resolve(v.name)
        if resolved in self._globals:
            return f"globals.{resolved}"
        return resolved

    # ------------------------------------------------------------------
    # Semaphore analysis
    # ------------------------------------------------------------------

    def _analyze_semaphores(self, ops: tuple[Op, ...]) -> None:
        for op in ops:
            if isinstance(op, WaitOp):
                group_key = frozenset(self._resolve(v.name) for v in op.values)
                if group_key not in self._wait_group_sems:
                    sem = f"sem_{self._sem_counter}"
                    self._sem_counter += 1
                    self._wait_group_sems[group_key] = sem
                sem = self._wait_group_sems[group_key]
                for v in op.values:
                    self._load_to_sem[self._resolve(v.name)] = sem
            elif isinstance(op, ForOp):
                self._analyze_semaphores(op.body)

    # ------------------------------------------------------------------
    # ThunderKittens type rendering
    # ------------------------------------------------------------------

    def _render_tk_type(self, t: TypeRef) -> str:
        if isinstance(t, SharedTileType):
            prefix = "st_bf" if t.data_type == GPUType.bf16 else "st_fl"
            if t.layout == TileLayout.row_major:
                return f"{prefix}<{t.rows}, {t.cols}>"
            return f"{prefix}<{t.rows}, {t.cols}, ducks::st_layout::col>"

        if isinstance(t, RegTileType):
            prefix = "rt_bf" if t.data_type == GPUType.bf16 else "rt_fl"
            layout = (
                "ducks::rt_layout::row"
                if t.layout == TileLayout.row_major
                else "ducks::rt_layout::col"
            )
            return f"{prefix}<{t.rows}, {t.cols}, {layout}>"

        if isinstance(t, GlobalType):
            sub = self._render_tk_type(t.sub_tile_type)
            return f"gl<{t.data_type.value}, {t.batch}, {t.depth}, {t.rows}, {t.cols}, {sub}>"

        if isinstance(t, IntType):
            return "int"

        raise TypeError(f"Cannot render type to TK: {t}")

    # ------------------------------------------------------------------
    # Coordinate rendering
    # ------------------------------------------------------------------

    def _render_coord(self, c: Value | int) -> str:
        if isinstance(c, int):
            return str(c)
        return self._val_name(c)

    def _render_coords(self, coords: tuple[Value | int, ...]) -> str:
        return "{" + ", ".join(self._render_coord(c) for c in coords) + "}"

    # ------------------------------------------------------------------
    # Variable declarations
    # ------------------------------------------------------------------

    def _collect_declarations(self, kernel: Kernel) -> list[str]:
        decls: list[str] = []
        declared: set[str] = set()

        def try_decl(name: str, typ: TypeRef, shared: bool = False) -> None:
            resolved = self._resolve(name)
            if resolved in declared or resolved in self._globals or resolved in self.BUILTINS:
                return
            declared.add(resolved)
            prefix = "__shared__ " if shared else ""
            decls.append(f"{prefix}{self._render_tk_type(typ)} {resolved};")

        def walk(ops: tuple[Op, ...]) -> None:
            for op in ops:
                if isinstance(op, ZeroOp):
                    try_decl(op.result.name, op.result.type)
                elif isinstance(op, TMALoadOp):
                    try_decl(op.result.name, op.result.type, shared=True)
                elif isinstance(op, ForOp):
                    try_decl(op.induction_var.name, op.induction_var.type)
                    walk(op.body)

        walk(kernel.body)
        return decls

    def _emit_sem_decls(self) -> list[str]:
        lines: list[str] = []
        for sem in sorted(self._wait_group_sems.values()):
            tic = f"tic_{sem}"
            lines.append(f"__shared__ semaphore {sem};")
            lines.append(f"int {tic} = 0;")
            lines.append(f"if (threadIdx.x == 0) {{")
            lines.append(f"  init_semaphore({sem}, 0, 1);")
            lines.append(f"}}")
        if lines:
            lines.append("__syncthreads();")
        return lines

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble(
        self,
        kernel: Kernel,
        decls: list[str],
        sem_decls: list[str],
        body_lines: list[str],
    ) -> str:
        L: list[str] = []
        L.append('#include "kittens.cuh"')
        L.append("using namespace kittens;")
        L.append("")
        L.append("struct global_vars {")
        for v in (*kernel.inputs, *kernel.outputs):
            L.append(f"  {self._render_tk_type(v.type)} {v.name};")
        L.append("};")
        L.append("")
        L.append(
            f"__global__ void {kernel.name}"
            "(const __grid_constant__ global_vars globals) {"
        )
        for d in decls:
            L.append(f"  {d}")
        for s in sem_decls:
            L.append(f"  {s}")
        if decls or sem_decls:
            L.append("")
        L.extend(body_lines)
        L.append("}")
        L.append("")
        return "\n".join(L)

    # ------------------------------------------------------------------
    # Op lowering
    # ------------------------------------------------------------------

    def _lower_ops(self, ops: tuple[Op, ...], depth: int) -> list[str]:
        lines: list[str] = []
        i = 0
        while i < len(ops):
            if isinstance(ops[i], TMALoadOp):
                batch: list[TMALoadOp] = []
                while i < len(ops) and isinstance(ops[i], TMALoadOp):
                    batch.append(ops[i])  # type: ignore[arg-type]
                    i += 1
                lines.extend(self._lower_tma_batch(batch, depth))
            else:
                lines.extend(self._lower_op(ops[i], depth))
                i += 1
        return lines

    def _lower_tma_batch(self, loads: list[TMALoadOp], depth: int) -> list[str]:
        pad = "  " * depth
        pad1 = "  " * (depth + 1)

        groups: dict[str, list[TMALoadOp]] = {}
        for ld in loads:
            sem = self._load_to_sem[self._resolve(ld.result.name)]
            groups.setdefault(sem, []).append(ld)

        lines: list[str] = []
        for sem, group_loads in groups.items():
            expect_parts = [
                f"size_bytes<typeof({self._val_name(ld.result)})>"
                for ld in group_loads
            ]
            expect_expr = " + ".join(expect_parts)

            lines.append(f"{pad}if (warpgroup::laneid() == 0) {{")
            lines.append(f"{pad1}tma::expect_bytes({sem}, {expect_expr});")
            for ld in group_loads:
                result = self._val_name(ld.result)
                source = self._val_name(ld.source)
                coords = self._render_coords(ld.coords)
                lines.append(
                    f"{pad1}tma::load_async({result}, {source}, {coords}, {sem});"
                )
            lines.append(f"{pad}}}")
        return lines

    def _lower_op(self, op: Op, depth: int) -> list[str]:
        pad = "  " * depth

        if isinstance(op, ZeroOp):
            return [f"{pad}kittens::warp::zero({self._val_name(op.result)});"]

        if isinstance(op, WaitOp):
            group_key = frozenset(self._resolve(v.name) for v in op.values)
            sem = self._wait_group_sems[group_key]
            tic = f"tic_{sem}"
            return [
                f"{pad}wait({sem}, {tic});",
                f"{pad}{tic} ^= 1;",
                f"{pad}__syncthreads();",
            ]

        if isinstance(op, MMAOp):
            accum = self._val_name(op.accum)
            a = self._val_name(op.a)
            b = self._val_name(op.b)
            return [
                f"{pad}warpgroup::mma_AB({accum}, {a}, {b});",
                f"{pad}warpgroup::mma_async_wait();",
                f"{pad}__syncthreads();",
            ]

        if isinstance(op, TMAStoreOp):
            source = self._val_name(op.source)
            dest = self._val_name(op.dest)
            coords = self._render_coords(op.coords)
            return [f"{pad}warpgroup::store({source}, {dest}, {coords});"]

        if isinstance(op, YieldOp):
            return []

        if isinstance(op, ForOp):
            return self._lower_for(op, depth)

        return []

    def _lower_for(self, op: ForOp, depth: int) -> list[str]:
        pad = "  " * depth
        iv = self._resolve(op.induction_var.name)

        start = (
            str(op.start)
            if isinstance(op.start, int)
            else self._val_name(op.start)
        )

        if op.tile_size is not None:
            stop_val = (
                self._val_name(op.stop)
                if isinstance(op.stop, Value)
                else str(op.stop)
            )
            stop = f"(({stop_val} + {op.tile_size - 1}) / {op.tile_size})"
        else:
            stop_val = (
                str(op.stop)
                if isinstance(op.stop, int)
                else self._val_name(op.stop)
            )
            stop = stop_val

        step = (
            str(op.step)
            if isinstance(op.step, int)
            else self._val_name(op.step)
        )

        lines = [f"{pad}for ({iv} = {start}; {iv} < {stop}; {iv} += {step}) {{"]
        lines.extend(self._lower_ops(op.body, depth + 1))
        lines.append(f"{pad}}}")
        return lines
