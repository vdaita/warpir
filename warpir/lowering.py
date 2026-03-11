from __future__ import annotations

from warpir import Program
from warpir.stmts import Stmt, SeqStmt, TileLoadOp, ForStmt, IfStmt


def lower_tile_load_op(tile_load: TileLoadOp) -> Stmt:
    return tile_load.lower()


def lower_stmt(stmt: Stmt) -> Stmt:
    if isinstance(stmt, TileLoadOp):
        return lower_tile_load_op(stmt)

    if isinstance(stmt, SeqStmt):
        lowered_children: list[Stmt] = []
        for child in stmt.stmts:
            lowered_child = lower_stmt(child)
            if isinstance(lowered_child, SeqStmt):
                lowered_children.extend(lowered_child.stmts)
            else:
                lowered_children.append(lowered_child)
        return SeqStmt(lowered_children)

    if isinstance(stmt, ForStmt):
        return ForStmt(
            stmt.init,
            stmt.cond,
            stmt.step,
            lower_stmt(stmt.body),
            stmt.inputs,
            stmt.yields,
        )

    if isinstance(stmt, IfStmt):
        else_stmt = lower_stmt(stmt.else_stmt) if stmt.else_stmt is not None else None
        return IfStmt(
            stmt.cond,
            lower_stmt(stmt.then_stmt),
            else_stmt,
        )

    return stmt


def lower_program(program: Program) -> Program:
    return Program(
        program.kernel_vars,
        lower_stmt(program.kernel_stmt),
        program.constants,
        program.launch_code,
    )
