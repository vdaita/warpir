from warpir import (
    AssignStmt,
    BinaryExpr,
    DeclStmt,
    ExprStmt,
    ForStmt,
    Kernel,
    LiteralExpr,
    Module,
    Param,
    SeqStmt,
    TypeRef,
    VarExpr,
    CallExpr,
)
from lowering import LoweringPipeline, ThunderKittensLowerer


def build_demo_module() -> Module:
    i = VarExpr("i")
    n = VarExpr("N")
    body = SeqStmt(
        [
            DeclStmt("acc", TypeRef("float"), LiteralExpr(0.0)),
            ForStmt(
                init=DeclStmt("i", TypeRef("int"), LiteralExpr(0)),
                cond=BinaryExpr(i, "<", n),
                step=AssignStmt(i, BinaryExpr(i, "+", LiteralExpr(1))),
                body=SeqStmt(
                    [
                        ExprStmt(CallExpr("warpgroup::mma_async_wait", [])),
                    ]
                ),
            ),
        ]
    )
    kernel = Kernel(
        name="demo_kernel",
        params=[
            Param("A", TypeRef("const __nv_bfloat16*")),
            Param("B", TypeRef("const __nv_bfloat16*")),
            Param("C", TypeRef("__nv_bfloat16*")),
            Param("N", TypeRef("int")),
        ],
        body=body,
    )
    return Module([kernel])


if __name__ == "__main__":
    module = build_demo_module()
    lowered = ThunderKittensLowerer().lower(LoweringPipeline().lower_module(module))
    print(lowered)
