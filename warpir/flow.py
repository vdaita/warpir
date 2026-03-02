from __future__ import annotations

from typing import Optional, Sequence, Union
from abc import ABC, abstractmethod
from jinja2 import Environment
from .layouts import Var, SharedTileType, ScalarType
from .ops import TMALoadOp, TMAStoreOp, WaitOp, ExpectBytesOp, ArriveOp, SizeBytesOfTypeOf, ExprLike, BinaryOp, Symbol, FieldRef, ThreadIdx, WarpId, WarpGroupId, BlockIdx, LaneId, OpCallExpr


class Stmt(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

_ENV = Environment(trim_blocks=True, lstrip_blocks=True)

class ForStmt(Stmt):
    def __init__(self, init: ExprLike, cond: ExprLike, step: ExprLike, body: Stmt):
        self.init = init
        self.cond = cond
        self.step = step
        self.body = body

    @classmethod
    def range(cls, var: str, start: Union[int, str], stop: Union[int, str], step: Union[int, str] = 1, body: Optional[Stmt] = None):
        init = f"int {var} = {start}"
        cond = f"{var} < {stop}"
        step_expr = f"{var} += {step}"
        return cls(init, cond, step_expr, body or NoStmt())

    def __str__(self) -> str:
        tmpl = _ENV.from_string(
            """for ({{ init }}; {{ cond }}; {{ step }}) {\n{% if body %}{{ body }}\n{% endif %}}"""
        )
        init = str(self.init).rstrip().rstrip(";")
        step = str(self.step).rstrip().rstrip(";")
        body = str(self.body).rstrip()
        return tmpl.render(init=init, cond=str(self.cond), step=step, body=body)

class WhileStmt(Stmt):
    def __init__(self, cond: ExprLike, body: Stmt):
        self.cond = cond
        self.body = body

    def __str__(self) -> str:
        tmpl = _ENV.from_string(
            """while ({{ cond }}) {\n{% if body %}{{ body }}\n{% endif %}}"""
        )
        body = str(self.body).rstrip()
        return tmpl.render(cond=str(self.cond), body=body)

class NoStmt(Stmt):
    def __init__(self):
        pass
    
    def __str__(self) -> str:
        return ""

class SeqStmt(Stmt):
    def __init__(self, stmts: Sequence[Union[Stmt, ExprLike]]):
        self.stmts = []
        for s in stmts:
            if isinstance(s, Stmt):
                self.stmts.append(s)
            else:
                self.stmts.append(ExprStmt(s))
    
    def __str__(self):
        rendered = [str(stmt).rstrip() for stmt in self.stmts if str(stmt).strip()]
        return "\n".join(rendered)

class RawStmt(Stmt):
    def __init__(self, code: str):
        self.code = code

    def __str__(self) -> str:
        return self.code.rstrip()

class ExprStmt(Stmt):
    def __init__(self, expr: ExprLike):
        self.expr = expr

    def __str__(self) -> str:
        text = str(self.expr).rstrip()
        if not text:
            return ""
        if text.endswith(";"):
            return text
        return f"{text};"

def OpCall(callee: str, *args: ExprLike) -> Stmt:
    return ExprStmt(OpCallExpr(callee, *args))

class AssignStmt(Stmt):
    def __init__(self, lhs: ExprLike, rhs: ExprLike):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        tmpl = _ENV.from_string("{{ lhs }} = {{ rhs }};")
        return tmpl.render(lhs=str(self.lhs), rhs=str(self.rhs))

class DeclStmt(Stmt):
    def __init__(self, var: Var, init: Optional[ExprLike] = None):
        self.var = var
        self.init = init

    def __str__(self) -> str:
        if self.init is None:
            return self.var.define()
        tmpl = _ENV.from_string("{{ type }} {{ name }} = {{ init }};")
        return tmpl.render(type=self.var.var_type, name=self.var.name, init=str(self.init))

class SharedAllocStmt(Stmt):
    def __init__(self, name: str, tile_type, allocator: str = "al", count: Optional[ExprLike] = None):
        self.name = name
        self.tile_type = tile_type
        self.allocator = allocator
        self.count = count

    def __str__(self) -> str:
        if self.count is None:
            tmpl = _ENV.from_string("{{ type }} &{{ name }} = {{ alloc }}.allocate<{{ type }}>();")
            return tmpl.render(type=self.tile_type, name=self.name, alloc=self.allocator)
        tmpl = _ENV.from_string(
            "{{ type }} (&{{ name }})[{{ count }}] = {{ alloc }}.allocate<{{ type }}, {{ count }}>();"
        )
        return tmpl.render(type=self.tile_type, name=self.name, alloc=self.allocator, count=str(self.count))

class IfStmt(Stmt):
    def __init__(self, cond: ExprLike, then_stmt: Stmt, else_stmt: Optional[Stmt] = None):
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

class Warpgroup(Stmt): # TODO: implement a full, functional warpgroup object that allows for splitting tasks
    def __init__(self, warp_id: int, wg_vars: Sequence[Var], stmt: Stmt):
        self.id = warp_id
        self.wg_vars = wg_vars
        self.stmt = stmt

    def __str__(self) -> str:
        tmpl = _ENV.from_string(
            """if (warpgroupid == {{ wid }}) {\n{% if body %}{{ body }}\n{% endif %}}"""
        )
        vars_block = "\n".join([v.define() for v in self.wg_vars]).rstrip()
        body_stmt = self.stmt
        if vars_block:
            body_stmt = SeqStmt([RawStmt(vars_block), self.stmt])
        body = str(body_stmt).rstrip()
        return tmpl.render(wid=self.id, body=body)

class WarpgroupDispatch(Stmt):
    def __init__(self, warpgroupid: ExprLike, cases: Sequence[tuple[int, Stmt]], default: Optional[Stmt] = None):
        self.warpgroupid = warpgroupid
        self.cases = list(cases)
        self.default = default

    def __str__(self) -> str:
        parts: list[str] = []
        for idx, (wid, stmt) in enumerate(self.cases):
            head = "if" if idx == 0 else "else if"
            body = str(stmt).rstrip()
            parts.append(f"{head} ({self.warpgroupid} == {wid}) {{\n{body}\n}}")
        if self.default is not None:
            parts.append(f"else {{\n{str(self.default).rstrip()}\n}}")
        return "\n".join(parts)



def lane0_if(stmt: Stmt) -> Stmt:
    return IfStmt(BinaryOp("==", LaneId, 0), stmt)



class KernelGlobals:
    def __init__(self, name: str = "matmul_globals", **vars_by_name):
        self.name = name
        self._vars = [Var(name, vtype) for name, vtype in vars_by_name.items()]
        self._var_map = {v.name: v for v in self._vars}
        self._sym = Symbol("g")
        
        # Programmatically collect aliases
        self.aliases = []
        seen_aliases = set()
        for v in self._vars:
            vtype = v.var_type
            if hasattr(vtype, 'alias_name') and vtype.alias_name:
                if vtype.alias_name not in seen_aliases:
                    # If it's a GlobalType, it might depend on a sub_tile alias
                    if hasattr(vtype, 'sub_tile') and hasattr(vtype.sub_tile, 'alias_name') and vtype.sub_tile.alias_name:
                        st = vtype.sub_tile
                        if st.alias_name not in seen_aliases:
                            self.aliases.append(f"using {st.alias_name} = {st.emit_type()};")
                            seen_aliases.add(st.alias_name)
                    
                    self.aliases.append(f"using {vtype.alias_name} = {vtype.emit_type()};")
                    seen_aliases.add(vtype.alias_name)

    @property
    def vars(self) -> Sequence[Var]:
        return self._vars

    def __iter__(self):
        return iter(self._vars)

    def __str__(self) -> str:
        return str(self._sym)

    def __getattr__(self, name: str):
        if name in self._var_map:
            return FieldRef(self._sym, name)
        raise AttributeError(name)

    def var(self, name: str) -> Var:
        return self._var_map[name]
        
class Tile:
    def __init__(
        self,
        name: str,
        tile_type,
        use_semaphores: bool = True,
        full_sem: Optional[str] = None,
        empty_sem: Optional[str] = None,
        allocator: str = "al",
    ):
        self.name = name
        self.tile_type = tile_type
        self.use_semaphores = use_semaphores
        self.full_sem = full_sem or f"full_{name}"
        self.empty_sem = empty_sem or f"empty_{name}"
        self.allocator = allocator

    def _is_shared(self) -> bool:
        return isinstance(self.tile_type, SharedTileType)

    def declare(self) -> Stmt:
        if self._is_shared():
            return SharedAllocStmt(self.name, self.tile_type, allocator=self.allocator)
        return DeclStmt(Var(self.name, self.tile_type))

    def def_(self) -> Stmt:
        return self.declare()

    def ref(self):
        return Symbol(self.name)

    def __str__(self) -> str:
        return self.name

    def declare_semaphores(self, shared: bool = True) -> Stmt:
        if not self.use_semaphores:
            return NoStmt()
        prefix = "__shared__ " if shared else ""
        return RawStmt(f"{prefix}semaphore {self.full_sem}, {self.empty_sem};")

    def init_semaphores(self, full_init: Union[str, int], empty_init: Union[str, int]) -> Stmt:
        if not self.use_semaphores:
            return NoStmt()
        return SeqStmt([
            RawStmt(f"init_semaphore({self.full_sem}, {full_init}, 1);"),
            RawStmt(f"init_semaphore({self.empty_sem}, {empty_init}, 0);"),
        ])

    def wait_empty(self) -> Stmt:
        return ExprStmt(WaitOp(self.empty_sem))

    def wait_full(self) -> Stmt:
        return ExprStmt(WaitOp(self.full_sem))

    def arrive_empty(self) -> Stmt:
        return lane0_if(ExprStmt(ArriveOp(self.empty_sem, 1)))

    def arrive_full(self) -> Stmt:
        return lane0_if(ExprStmt(ArriveOp(self.full_sem, 1)))

    def expect_bytes(self) -> Stmt:
        return lane0_if(ExprStmt(ExpectBytesOp(self.full_sem, SizeBytesOfTypeOf(self.name))))


KITTENS_TEMPLATE = """
#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kittens.cuh"

using namespace kittens;

#ifndef cudaLaunchAttributePreferredClusterDimension
#define cudaLaunchAttributePreferredClusterDimension cudaLaunchAttributeClusterDimension
#endif

{{ constants }}

struct {{ globals_name }} {
    {%- for alias in aliases %}
    {{ alias }}
    {%- endfor %}
    {%- for var in kernel_vars %}
    {{ var.define() }}
    {%- endfor %}
};

__global__ void kernel(const __grid_constant__ {{ globals_name }} g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    {{ kernel_stmt }}
}

{{ launch_code }}
"""
class Program:
    def __init__(self, input_vars: Sequence[Var], kernel_vars: KernelGlobals, kernel_stmt: Stmt, 
                 constants: str = "", launch_code: Optional[str] = None,
                 grid_dims: Optional[ExprLike] = None, block_dims: Optional[ExprLike] = None,
                 shared_mem: Optional[ExprLike] = None, launch_name: str = "launch"):
        self.input_vars = input_vars
        self.kernel_vars = kernel_vars
        self.kernel_stmt = kernel_stmt
        self.constants = constants
        self.launch_code = launch_code
        self.grid_dims = grid_dims
        self.block_dims = block_dims
        self.shared_mem = shared_mem
        self.launch_name = launch_name
    
    def __str__(self):
        if isinstance(self.kernel_vars, KernelGlobals):
            kernel_vars = self.kernel_vars.vars
            aliases = self.kernel_vars.aliases
            globals_name = self.kernel_vars.name
        else:
            kernel_vars = self.kernel_vars
            aliases = []
            globals_name = "kernel_globals"
        t = _ENV.from_string(KITTENS_TEMPLATE)
        
        launch_code = self.launch_code
        if launch_code is None:
            # Automate launch code generation
            # We assume a standard signature for now: Tensor A, Tensor B, Tensor C, size_t N
            # But we can generalize by looking at input_vars
            
            torch_args = []
            call_args = []
            check_lines = []
            init_lines = []
            
            for v in self.input_vars:
                if "bf16*" in str(v.var_type):
                    torch_args.append(f"torch::Tensor {v.name}")
                    check_lines.append(f'  TORCH_CHECK({v.name}.is_cuda(), "Tensor {v.name} must be on CUDA");')
                    check_lines.append(f"  TORCH_CHECK({v.name}.dtype() == torch::kBFloat16, \"Tensor {v.name} must be bfloat16\");")
                    call_args.append(f"reinterpret_cast<bf16*>({v.name}.data_ptr<at::BFloat16>())")
                else:
                    torch_args.append(f"{v.var_type} {v.name}")
                    call_args.append(v.name)
            
            # Host-side matmul internal launch wrapper (the one that sets up globals)
            internal_args = ", ".join([f"{v.var_type} {v.name}" for v in self.input_vars])
            
            for v in self.kernel_vars.vars:
                if v.name in [iv.name for iv in self.input_vars]:
                    from .layouts import GlobalType
                    vtype = v.var_type
                    if isinstance(vtype, GlobalType) or (isinstance(vtype, str) and "gl" in vtype) or (hasattr(vtype, 'alias_name') and vtype.alias_name and "gl" in str(vtype)):
                        N_vars = [iv.name for iv in self.input_vars if iv.name == 'N']
                        dim_val = f"(int){N_vars[0]}" if N_vars else "-1"
                        type_str = vtype.alias_name if hasattr(vtype, 'alias_name') and vtype.alias_name else str(vtype)
                        init_lines.append(f"  using {v.name}_t = {globals_name}::{type_str};")
                        init_lines.append(f"  {v.name}_t {v.name}_arg{{{v.name}, nullptr, nullptr, {dim_val}, {dim_val}}};")
                    else:
                        init_lines.append(f"  {v.var_type} {v.name}_arg = {v.name};")

            g_init = ", ".join([f"{v.name}_arg" for v in self.kernel_vars.vars])
            grid = str(self.grid_dims) if self.grid_dims else "1"
            block = str(self.block_dims) if self.block_dims else "1"
            shm = str(self.shared_mem) if self.shared_mem else "0"
            
            launch_code = f"""
void {self.launch_name}_internal({internal_args}) {{
{chr(10).join(init_lines)}
  {globals_name} g{{{g_init}}};
  dim3 grid{{{grid}}};
  dim3 block{{{block}}};
  unsigned long mem_size = {shm};
  if (mem_size > 0) {{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
  }}
  kernel<<<grid, block, mem_size>>>(g);
  CHECK_CUDA_ERROR(cudaGetLastError());
  cudaDeviceSynchronize();
}}

void {self.launch_name}({", ".join(torch_args)}) {{
{chr(10).join(check_lines)}
  {self.launch_name}_internal({", ".join(call_args)});
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("launch", &{self.launch_name}, "PyTorch wrapper for generated kernel");
}}
"""

        return t.render(
            constants=self.constants,
            globals_name=globals_name,
            aliases=aliases,
            kernel_vars=kernel_vars,
            kernel_stmt=self.kernel_stmt,
            launch_code=launch_code
        )
