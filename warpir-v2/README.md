# warpir-v2

Basic v2 IR prototype:

- `warpir_v2/ir.py`: grammar/AST data model
  - `Kernel(name, params, body)`
  - `Stmt` nodes: `SeqStmt`, `DeclStmt`, `AssignStmt`, `IfStmt`, `ForStmt`, `WhileStmt`, `ExprStmt`, `ReturnStmt`
  - `Expr` nodes: `VarExpr`, `LiteralExpr`, `BinaryExpr`, `CallExpr`, `RawExpr`
- `warpir_v2/lowering.py`: starter lowering to ThunderKittens-flavored CUDA text
  - `LoweringPipeline`: placeholder pass pipeline
  - `ThunderKittensLowerer`: emits includes + `__global__` kernels
- `example.py`: tiny end-to-end usage

Run the example:

```bash
python warpir-v2/example.py
```
