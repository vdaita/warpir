from __future__ import annotations

from warpir.ir.ops import (
    AllocSharedOp,
    BufSlotExpr,
    CopyOp,
    DivRowOp,
    Exp2Op,
    ForOp,
    Kernel,
    MMABufOp,
    MMAOp,
    MulOp,
    MulRowOp,
    MulScalarOp,
    NegInftyOp,
    Op,
    RowMaxOp,
    RowSumOp,
    SubOp,
    SubRowOp,
    TMALoadBufOp,
    TMALoadOp,
    TMAStoreOp,
    Value,
    WaitBufOp,
    WaitOp,
    WarpSpecializedRegionOp,
    YieldOp,
    ZeroOp,
)
from warpir.ir.types import (
    ColVecType,
    GPUType,
    GlobalType,
    IntType,
    RegTileType,
    SharedBufferType,
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
        self._buf_allocations: dict[str, SharedBufferType] = {}
        self._buf_num_stages: int = 0
        self._has_warp_specialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lower(self, kernel: Kernel) -> str:
        self._reset()

        for v in (*kernel.inputs, *kernel.outputs):
            self._globals.add(v.name)

        self._build_aliases(kernel.body)
        self._analyze_semaphores(kernel.body)
        self._analyze_buffer_ops(kernel.body)

        decls = self._collect_declarations(kernel)
        sem_decls = self._emit_sem_decls()
        body_lines = self._lower_ops(kernel.body, depth=1)

        return self._assemble(kernel, decls, sem_decls, body_lines)

    # ------------------------------------------------------------------
    # Alias resolution (SSA iter_args → single mutable variable)
    # ------------------------------------------------------------------

    def _build_aliases(self, ops: tuple[Op, ...]) -> None:
        for op in ops:
            if isinstance(op, ForOp):
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
            elif isinstance(op, WarpSpecializedRegionOp):
                for i, ia in enumerate(op.consumer_iter_args):
                    self._aliases[ia.init.name] = ia.block_arg.name
                    if i < len(op.consumer_results):
                        self._aliases[op.consumer_results[i].name] = ia.block_arg.name
                for body_op in op.consumer_body:
                    if isinstance(body_op, YieldOp):
                        for i, yv in enumerate(body_op.values):
                            if i < len(op.consumer_iter_args):
                                self._aliases[yv.name] = op.consumer_iter_args[i].block_arg.name
                self._build_aliases(op.consumer_body)
                self._build_aliases(op.consumer_setup)
                self._build_aliases(op.consumer_finish)

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
    # Semaphore analysis (SSA WaitOp path)
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
    # Buffer op analysis
    # ------------------------------------------------------------------

    def _analyze_buffer_ops(self, ops: tuple[Op, ...]) -> None:
        for op in ops:
            if isinstance(op, AllocSharedOp):
                if isinstance(op.result.type, SharedBufferType):
                    self._buf_allocations[op.result.name] = op.result.type
                    self._buf_num_stages = max(
                        self._buf_num_stages, op.result.type.count,
                    )
            elif isinstance(op, WarpSpecializedRegionOp):
                self._has_warp_specialized = True
            elif isinstance(op, ForOp):
                self._analyze_buffer_ops(op.body)

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

        if isinstance(t, ColVecType):
            prefix = "rt_bf" if t.data_type == GPUType.bf16 else "rt_fl"
            return f"col_vec<{prefix}<{t.rows}, {t.cols}>>"

        if isinstance(t, IntType):
            return "int"

        raise TypeError(f"Cannot render type to TK: {t}")

    # ------------------------------------------------------------------
    # Coordinate / buffer-index rendering
    # ------------------------------------------------------------------

    def _render_coord(self, c: Value | int) -> str:
        if isinstance(c, int):
            return str(c)
        name = self._val_name(c)
        if name in ("blockIdx.x", "blockIdx.y", "blockIdx.z"):
            return f"(int){name}"
        return name

    def _render_coords(self, coords: tuple[Value | int, ...]) -> str:
        return "{" + ", ".join(self._render_coord(c) for c in coords) + "}"

    def _render_buf_idx(self, slot, buf_name: str) -> str:
        if isinstance(slot, int):
            return str(slot)
        # BufSlotExpr
        mod = slot.modulus
        if slot.value is None:
            return "_compute_buf"
        val = self._val_name(slot.value)
        eff_offset = slot.offset % mod
        if eff_offset == 0:
            return f"{val} % {mod}"
        return f"({val} + {eff_offset}) % {mod}"

    # ------------------------------------------------------------------
    # Variable declarations
    # ------------------------------------------------------------------

    def _collect_declarations(self, kernel: Kernel) -> list[str]:
        decls: list[str] = []
        declared: set[str] = set()

        # Buffer array declarations via dynamic shared allocator
        if self._buf_allocations:
            decls.append("extern __shared__ alignment_dummy __shm[];")
            decls.append("shared_allocator al((int*)&__shm[0]);")
            for name, bt in self._buf_allocations.items():
                ts = self._render_tk_type(bt.tile_type)
                decls.append(
                    f"{ts} (&{name})[{bt.count}] = "
                    f"al.allocate<{ts}, {bt.count}>();"
                )

        def try_decl(name: str, typ: TypeRef, shared: bool = False) -> None:
            resolved = self._resolve(name)
            if resolved in declared or resolved in self._globals or resolved in self.BUILTINS:
                return
            declared.add(resolved)
            prefix = "__shared__ " if shared else ""
            decls.append(f"{prefix}{self._render_tk_type(typ)} {resolved};")

        def walk(ops: tuple[Op, ...]) -> None:
            for op in ops:
                if isinstance(op, (AllocSharedOp, TMALoadBufOp)):
                    continue
                if isinstance(op, WarpSpecializedRegionOp):
                    try_decl(op.induction_var.name, op.induction_var.type)
                    continue
                if hasattr(op, 'result') and isinstance(getattr(op, 'result'), Value):
                    is_shared = isinstance(op, TMALoadOp)
                    try_decl(op.result.name, op.result.type, shared=is_shared)
                if isinstance(op, ForOp):
                    try_decl(op.induction_var.name, op.induction_var.type)
                    walk(op.body)

        walk(kernel.body)

        if self._buf_allocations and not self._has_warp_specialized:
            decls.append("int _compute_buf = 0;")

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
        n = self._buf_num_stages
        if n and not self._has_warp_specialized:
            tic_init = ", ".join(["0"] * n)
            lines.append(f"__shared__ semaphore sem_pipe[{n}];")
            lines.append(f"int tic[{n}] = {{{tic_init}}};")
            lines.append(f"if (threadIdx.x == 0) {{")
            lines.append(f"  for (int _s = 0; _s < {n}; _s++) "
                         f"init_semaphore(sem_pipe[_s], 0, 1);")
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
            if isinstance(ops[i], WarpSpecializedRegionOp):
                lines.extend(self._lower_warp_specialized(ops[i], depth))  # type: ignore[arg-type]
                i += 1
            elif isinstance(ops[i], TMALoadOp):
                batch: list[TMALoadOp] = []
                while i < len(ops) and isinstance(ops[i], TMALoadOp):
                    batch.append(ops[i])  # type: ignore[arg-type]
                    i += 1
                lines.extend(self._lower_tma_batch(batch, depth))
            elif isinstance(ops[i], TMALoadBufOp):
                batch_buf: list[TMALoadBufOp] = []
                while i < len(ops) and isinstance(ops[i], TMALoadBufOp):
                    batch_buf.append(ops[i])  # type: ignore[arg-type]
                    i += 1
                lines.extend(self._lower_tma_buf_batch(batch_buf, depth))
            else:
                lines.extend(self._lower_op(ops[i], depth))
                i += 1
        return lines

    # ------------------------------------------------------------------
    # SSA TMA load batching (non-pipelined path)
    # ------------------------------------------------------------------

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

            lines.append(f"{pad}if (threadIdx.x == 0) {{")
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

    # ------------------------------------------------------------------
    # Buffer TMA load batching (pipelined path)
    # ------------------------------------------------------------------

    def _lower_tma_buf_batch(
        self, loads: list[TMALoadBufOp], depth: int
    ) -> list[str]:
        pad = "  " * depth
        pad1 = "  " * (depth + 1)

        def _slot_key(slot):
            if isinstance(slot, int):
                return ("lit", slot)
            return ("expr", slot.value.name if slot.value else None,
                    slot.offset, slot.modulus)

        groups: list[list[TMALoadBufOp]] = []
        cur: list[TMALoadBufOp] = [loads[0]]
        for ld in loads[1:]:
            if _slot_key(ld.slot) == _slot_key(cur[0].slot):
                cur.append(ld)
            else:
                groups.append(cur)
                cur = [ld]
        groups.append(cur)

        lines: list[str] = []
        lines.append(f"{pad}if (threadIdx.x == 0) {{")
        for group in groups:
            slot_idx = self._render_buf_idx(group[0].slot, group[0].buf.name)
            sem_expr = f"sem_pipe[{slot_idx}]"

            expect_parts = [
                f"size_bytes<typeof({ld.buf.name}[0])>"
                for ld in group
            ]
            expect_expr = " + ".join(expect_parts)
            lines.append(f"{pad1}tma::expect_bytes({sem_expr}, {expect_expr});")
            for ld in group:
                idx = self._render_buf_idx(ld.slot, ld.buf.name)
                source = self._val_name(ld.source)
                coords = self._render_coords(ld.coords)
                lines.append(
                    f"{pad1}tma::load_async("
                    f"{ld.buf.name}[{idx}], {source}, {coords}, {sem_expr});"
                )
        lines.append(f"{pad}}}")
        if depth == 1:
            lines.append(f"{pad}__syncthreads();")
        return lines

    # ------------------------------------------------------------------
    # Single-op lowering
    # ------------------------------------------------------------------

    def _lower_op(self, op: Op, depth: int) -> list[str]:
        pad = "  " * depth

        if isinstance(op, AllocSharedOp):
            return []

        if isinstance(op, ZeroOp):
            return [f"{pad}kittens::warp::zero({self._val_name(op.result)});"]

        if isinstance(op, NegInftyOp):
            return [f"{pad}kittens::warp::neg_infty({self._val_name(op.result)});"]

        if isinstance(op, WaitOp):
            group_key = frozenset(self._resolve(v.name) for v in op.values)
            sem = self._wait_group_sems[group_key]
            tic = f"tic_{sem}"
            return [
                f"{pad}wait({sem}, {tic});",
                f"{pad}{tic} ^= 1;",
                f"{pad}__syncthreads();",
            ]

        if isinstance(op, WaitBufOp):
            slot_idx = self._render_buf_idx(op.slot, "")
            return [
                f"{pad}wait(sem_pipe[{slot_idx}], tic[{slot_idx}]);",
                f"{pad}tic[{slot_idx}] ^= 1;",
                f"{pad}__syncthreads();",
            ]

        if isinstance(op, MMAOp):
            result = self._val_name(op.result)
            accum = self._val_name(op.accum)
            a = self._val_name(op.a)
            b = self._val_name(op.b)
            lines: list[str] = []
            if op.transpose_b:
                lines.append(f"{pad}warpgroup::mm_ABt({result}, {a}, {b});")
            else:
                if result != accum:
                    lines.append(
                        f"{pad}kittens::warp::copy({result}, {accum});"
                    )
                lines.append(f"{pad}warpgroup::mma_AB({result}, {a}, {b});")
            lines.append(f"{pad}warpgroup::mma_async_wait();")
            lines.append(f"{pad}__syncthreads();")
            return lines

        if isinstance(op, MMABufOp):
            a_is_buf = op.a_buf.name in self._buf_allocations
            b_is_buf = op.b_buf.name in self._buf_allocations

            if a_is_buf:
                a_idx = self._render_buf_idx(op.a_slot, op.a_buf.name)
                a_expr = f"{op.a_buf.name}[{a_idx}]"
            else:
                a_expr = self._val_name(op.a_buf)

            if b_is_buf:
                b_idx = self._render_buf_idx(op.b_slot, op.b_buf.name)
                b_expr = f"{op.b_buf.name}[{b_idx}]"
            else:
                b_expr = self._val_name(op.b_buf)

            result = self._val_name(op.result)
            accum = self._val_name(op.accum)
            lines: list[str] = []
            if op.transpose_b:
                lines.append(f"{pad}warpgroup::mm_ABt({result}, {a_expr}, {b_expr});")
            else:
                if result != accum:
                    lines.append(
                        f"{pad}kittens::warp::copy({result}, {accum});"
                    )
                lines.append(f"{pad}warpgroup::mma_AB({result}, {a_expr}, {b_expr});")
            lines.append(f"{pad}warpgroup::mma_async_wait();")
            lines.append(f"{pad}__syncthreads();")
            return lines

        if isinstance(op, CopyOp):
            return [
                f"{pad}kittens::warp::copy("
                f"{self._val_name(op.result)}, {self._val_name(op.input)});"
            ]

        if isinstance(op, MulScalarOp):
            return [
                f"{pad}kittens::warp::mul("
                f"{self._val_name(op.result)}, "
                f"{self._val_name(op.input)}, {op.scalar}f);"
            ]

        if isinstance(op, SubOp):
            return [
                f"{pad}kittens::warp::sub("
                f"{self._val_name(op.result)}, "
                f"{self._val_name(op.a)}, {self._val_name(op.b)});"
            ]

        if isinstance(op, MulOp):
            return [
                f"{pad}kittens::warp::mul("
                f"{self._val_name(op.result)}, "
                f"{self._val_name(op.a)}, {self._val_name(op.b)});"
            ]

        if isinstance(op, Exp2Op):
            return [
                f"{pad}kittens::warp::exp2("
                f"{self._val_name(op.result)}, {self._val_name(op.input)});"
            ]

        if isinstance(op, RowMaxOp):
            return [
                f"{pad}kittens::warp::row_max("
                f"{self._val_name(op.result)}, "
                f"{self._val_name(op.tile)}, {self._val_name(op.prev)});"
            ]

        if isinstance(op, RowSumOp):
            return [
                f"{pad}kittens::warp::row_sum("
                f"{self._val_name(op.result)}, "
                f"{self._val_name(op.tile)}, {self._val_name(op.prev)});"
            ]

        if isinstance(op, SubRowOp):
            return [
                f"{pad}kittens::warp::sub_row("
                f"{self._val_name(op.result)}, "
                f"{self._val_name(op.tile)}, {self._val_name(op.vec)});"
            ]

        if isinstance(op, MulRowOp):
            return [
                f"{pad}kittens::warp::mul_row("
                f"{self._val_name(op.result)}, "
                f"{self._val_name(op.tile)}, {self._val_name(op.vec)});"
            ]

        if isinstance(op, DivRowOp):
            return [
                f"{pad}kittens::warp::div_row("
                f"{self._val_name(op.result)}, "
                f"{self._val_name(op.tile)}, {self._val_name(op.vec)});"
            ]

        if isinstance(op, TMAStoreOp):
            source = self._val_name(op.source)
            dest = self._val_name(op.dest)
            coords = self._render_coords(op.coords)
            return [f"{pad}warpgroup::store({dest}, {source}, {coords});"]

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

    # ------------------------------------------------------------------
    # Warp-specialized lowering (producer/consumer with full/empty sems)
    # ------------------------------------------------------------------

    def _ws_stop_expr(self, op: WarpSpecializedRegionOp) -> str:
        if op.tile_size is not None:
            stop_val = (
                self._val_name(op.stop) if isinstance(op.stop, Value) else str(op.stop)
            )
            return f"(({stop_val} + {op.tile_size - 1}) / {op.tile_size})"
        return self._val_name(op.stop) if isinstance(op.stop, Value) else str(op.stop)

    def _ws_consumer_decls(self, op: WarpSpecializedRegionOp) -> list[str]:
        """Collect register-tile / col-vec declarations for the consumer branch."""
        decls: list[str] = []
        declared: set[str] = set()

        def try_decl(name: str, typ: TypeRef) -> None:
            resolved = self._resolve(name)
            if resolved in declared or resolved in self._globals or resolved in self.BUILTINS:
                return
            declared.add(resolved)
            decls.append(f"{self._render_tk_type(typ)} {resolved};")

        for ia in op.consumer_iter_args:
            try_decl(ia.block_arg.name, ia.block_arg.type)

        all_ops = list(op.consumer_setup) + list(op.consumer_body) + list(op.consumer_finish)
        for cop in all_ops:
            if isinstance(cop, (AllocSharedOp, TMALoadBufOp)):
                continue
            if hasattr(cop, 'result') and isinstance(getattr(cop, 'result'), Value):
                try_decl(cop.result.name, cop.result.type)

        return decls

    def _lower_warp_specialized(
        self, op: WarpSpecializedRegionOp, depth: int
    ) -> list[str]:
        pad = "  " * depth
        p1 = "  " * (depth + 1)
        ns = op.num_stages
        iv = self._resolve(op.induction_var.name)
        stop = self._ws_stop_expr(op)

        lines: list[str] = []
        lines.append(f"{pad}const int warpid = kittens::warpid();")
        lines.append(f"{pad}const int warpgroupid = warpid / 4;")
        lines.append("")

        # ── Full/empty semaphore init ─────────────────────────────────
        nc = op.num_consumer_warpgroups
        lines.append(f"{pad}__shared__ semaphore full[{ns}], empty[{ns}];")
        lines.append(f"{pad}if (threadIdx.x == 0) {{")
        lines.append(f"{p1}for (int _i = 0; _i < {ns}; _i++) {{")
        lines.append(f"{p1}  init_semaphore(full[_i], 0, 1);")
        lines.append(f"{p1}  init_semaphore(empty[_i], {nc}, 0);")
        lines.append(f"{p1}}}")
        lines.append(f"{pad}}}")
        lines.append(f"{pad}__syncthreads();")
        lines.append("")

        # ── Producer branch ───────────────────────────────────────────
        lines.append(f"{pad}if (warpgroupid == 0) {{ // producer")
        lines.extend(self._ws_lower_producer(op, depth + 1, iv, stop))

        # ── Consumer branch ───────────────────────────────────────────
        lines.append(f"{pad}}} else {{ // consumer")
        lines.extend(self._ws_lower_consumer(op, depth + 1, iv, stop))
        lines.append(f"{pad}}}")
        return lines

    def _ws_lower_producer(
        self, op: WarpSpecializedRegionOp, depth: int, iv: str, stop: str
    ) -> list[str]:
        pad = "  " * depth
        p1 = "  " * (depth + 1)
        p2 = "  " * (depth + 2)
        ns = op.num_stages

        lines: list[str] = []
        lines.append(f"{pad}warpgroup::decrease_registers<32>();")
        lines.append(f"{pad}if (warpgroup::laneid() == 0) {{")
        lines.append(f"{p1}int _p = 0, _qidx = 0;")
        lines.append(f"{p1}for ({iv} = 0; {iv} < {stop}; {iv} += 1, _qidx++) {{")
        lines.append(f"{p2}if (_qidx == {ns}) {{ _qidx = 0; _p ^= 1; }}")
        lines.append(f"{p2}wait(empty[_qidx], _p);")

        # Group all TMALoadBufOps — they share the same full[_qidx] semaphore
        load_ops = [o for o in op.producer_body if isinstance(o, TMALoadBufOp)]
        if load_ops:
            expect_parts = [f"size_bytes<typeof({ld.buf.name}[0])>" for ld in load_ops]
            lines.append(f"{p2}tma::expect_bytes(full[_qidx], {' + '.join(expect_parts)});")
            for ld in load_ops:
                source = self._val_name(ld.source)
                coords = self._render_coords(ld.coords)
                lines.append(
                    f"{p2}tma::load_async("
                    f"{ld.buf.name}[_qidx], {source}, {coords}, full[_qidx]);"
                )

        lines.append(f"{p1}}}")  # end for
        lines.append(f"{pad}}}")  # end laneid
        return lines

    def _ws_lower_consumer(
        self, op: WarpSpecializedRegionOp, depth: int, iv: str, stop: str
    ) -> list[str]:
        pad = "  " * depth
        p1 = "  " * (depth + 1)
        p2 = "  " * (depth + 2)
        ns = op.num_stages

        lines: list[str] = []
        lines.append(f"{pad}warpgroup::increase_registers<256>();")

        # Consumer-local register declarations
        for d in self._ws_consumer_decls(op):
            lines.append(f"{pad}{d}")

        # Consumer setup (zero accumulators, etc.)
        for setup_op in op.consumer_setup:
            lines.extend(self._lower_op(setup_op, depth))
        lines.append("")

        # Kickstart producer by signalling empty semaphores
        lines.append(f"{pad}if (warpgroup::laneid() == 0)")
        lines.append(f"{p1}for (int _i = 0; _i < {ns}; _i++) arrive(empty[_i], 1);")
        lines.append("")

        # Consumer loop
        lines.append(f"{pad}int _p = 0, _qidx = 0;")
        lines.append(f"{pad}for ({iv} = 0; {iv} < {stop}; {iv} += 1, _qidx++) {{")
        lines.append(f"{p1}if (_qidx == {ns}) {{ _qidx = 0; _p ^= 1; }}")
        lines.append(f"{p1}wait(full[_qidx], _p);")

        for body_op in op.consumer_body:
            if isinstance(body_op, MMABufOp):
                accum = self._val_name(body_op.accum)
                a_buf = body_op.a_buf.name
                b_buf = body_op.b_buf.name
                lines.append(f"{p1}warpgroup::mma_AB({accum}, {a_buf}[_qidx], {b_buf}[_qidx]);")
                lines.append(f"{p1}warpgroup::mma_async_wait();")
            elif isinstance(body_op, YieldOp):
                pass
            else:
                lines.extend(self._lower_op(body_op, depth + 1))

        lines.append(f"{p1}if (warpgroup::laneid() == 0) arrive(empty[_qidx], 1);")
        lines.append(f"{pad}}}")  # end for
        lines.append("")

        # Post-loop (store, etc.)
        for finish_op in op.consumer_finish:
            lines.extend(self._lower_op(finish_op, depth))

        return lines
