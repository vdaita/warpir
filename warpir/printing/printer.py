from __future__ import annotations

from warpir.ir.ops import (
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


def print_kernel(kernel: Kernel) -> None:
    """Print an SSA IR kernel to stdout."""
    print(format_kernel(kernel))


def format_kernel(kernel: Kernel) -> str:
    """Return a human-readable SSA IR string for *kernel*."""
    lines: list[str] = []
    inputs_str = ", ".join(repr(v) for v in kernel.inputs)
    outputs_str = ", ".join(repr(v) for v in kernel.outputs)
    lines.append(f"kernel {kernel.name}({inputs_str}) -> ({outputs_str}) {{")
    _format_ops(kernel.body, 1, lines)
    lines.append("}")
    return "\n".join(lines)


def _format_coord(c: Value | int) -> str:
    if isinstance(c, int):
        return str(c)
    return repr(c)


def _format_ops(ops: tuple[Op, ...], indent: int, lines: list[str]) -> None:
    pad = "  " * indent
    for op in ops:
        if isinstance(op, ZeroOp):
            lines.append(f"{pad}{repr(op.result)} = zero")

        elif isinstance(op, TMALoadOp):
            coords = ", ".join(_format_coord(c) for c in op.coords)
            lines.append(
                f"{pad}{repr(op.result)} = tma_load({repr(op.source)}, [{coords}])"
            )

        elif isinstance(op, WaitOp):
            vals = ", ".join(repr(v) for v in op.values)
            lines.append(f"{pad}wait({vals})")

        elif isinstance(op, MMAOp):
            lines.append(
                f"{pad}{repr(op.result)} = mma({repr(op.a)}, {repr(op.b)}, {repr(op.accum)})"
            )

        elif isinstance(op, TMAStoreOp):
            coords = ", ".join(_format_coord(c) for c in op.coords)
            lines.append(
                f"{pad}tma_store({repr(op.source)}, {repr(op.dest)}, [{coords}])"
            )

        elif isinstance(op, YieldOp):
            vals = ", ".join(repr(v) for v in op.values)
            lines.append(f"{pad}yield({vals})")

        elif isinstance(op, ForOp):
            iter_args_str = ", ".join(
                f"{repr(ia.block_arg)} = {repr(ia.init)}"
                for ia in op.iter_args
            )
            results_str = ", ".join(repr(r) for r in op.results)
            tile_str = f" tile_size={op.tile_size}" if op.tile_size else ""
            lines.append(
                f"{pad}({results_str}) = for {repr(op.induction_var)} "
                f"= {op.start} to {op.stop} step {op.step}{tile_str}"
            )
            lines.append(f"{pad}    iter_args({iter_args_str}) {{")
            _format_ops(op.body, indent + 2, lines)
            lines.append(f"{pad}}}")
