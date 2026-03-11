from warpir.stmts import *

class Tile(Var):
    def __init__(
        self,
        name: str,
        var_type,
        use_semaphores: bool = False,
        num_consumers: int = 1
    ):
        super().__init__(name, var_type)
        self.use_semaphores = use_semaphores
        self.num_consumers = num_consumers
        self.var = Var(self.name, self.var_type)

        if use_semaphores:
            self._manager = TileGroup(name, [self], num_consumers)

    def __str__(self):
        return str(self.var)

    def __eq__(self, other):
        if not isinstance(other, Tile):
            return False
        return (self.var, self.use_semaphores, self.num_consumers) == (other.var, other.use_semaphores, other.num_consumers)

    def __hash__(self):
        return hash(str(self))

    def declare(self) -> Stmt:
        stmts: List[Stmt] = [self.var.declare()]
        if self.use_semaphores:
            stmts.append(self._manager.initialize())
        return SeqStmt(stmts)

    def load_global(self, src: Var, coord: Coord):
        assert type(src.var_type) == GlobalType
        return SeqStmt([
            ExprStmt(OpCall("kittens::warp::load", [self, src, coord])),
            ExprStmt(OpCall("__syncthreads", []))
        ])

    def warp_store_global(self, dst: Var, coord: Coord):
        assert type(dst.var_type) == GlobalType
        return ExprStmt(OpCall("kittens::warp::store", [self, dst, coord]))

    def warpgroup_store_global(self, dst: Var, coord: Coord):
        assert type(dst.var_type) == GlobalType
        return ExprStmt(OpCall("warpgroup::store", [self, dst, coord]))

    def load_shared(self, src: Var):
        assert type(src.var_type) == SharedTileType or type(self.var_type) == SharedVecType
        return SeqStmt([
            ExprStmt(OpCall("kittens::warp::load", [self, src])),
            ExprStmt(OpCall("__syncthreads", []))
        ])

    def async_load_global(self, src: Var, coord: Coord) -> Stmt:
        assert type(src.var_type) == GlobalType
        return self._manager.async_load_global([MemLoad(src, self, coord)])

    def wait_full(self, level):
        return self._manager.wait_full(level)

    def wait_empty(self, level):
        return self._manager.wait_empty(level)

    def arrive_empty(self):
        return self._manager.arrive_empty()

@dataclass
class MemLoad:
    source: Var
    dest: Tile
    coord: Coord

class TileGroup:
    def __init__(
        self,
        name: str,
        tiles: List[Tile],
        num_consumers: int = 1
    ):
        self.name = name
        self.tiles = tiles
        self.num_consumers = num_consumers

        self.full_sem = Var(f"full_{self.name}", SharedSemaphoreType())
        self.empty_sem = Var(f"empty_{self.name}", SharedSemaphoreType())
        self.full_tic = Var(f"tic_full_{self.name}", ScalarType("int"))
        self.empty_tic = Var(f"tic_empty_{self.name}", ScalarType("int"))

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, TileGroup):
            return False
        return (self.name, self.tiles, self.num_consumers) == (other.name, other.tiles, other.num_consumers)

    def __hash__(self):
        return hash(str(self))

    def initialize(self):
        stmts: List[Stmt] = []
        stmts.append(self.full_sem.declare())
        stmts.append(self.empty_sem.declare())
        stmts.append(self.full_tic.declare())
        stmts.append(self.empty_tic.declare())
        stmts.append(AssignExpr(self.full_tic, zero).to_stmt())
        stmts.append(AssignExpr(self.empty_tic, zero).to_stmt())
        stmts.append(
            thread0_if(ExprStmt(OpCall("init_semaphore", [self.full_sem, zero, one])))
        )
        stmts.append(
            thread0_if(ExprStmt(OpCall("init_semaphore", [self.empty_sem, RawExpr(f"{self.num_consumers}"), RawExpr("0")])))
        )
        stmts.append(
            thread0_if(ExprStmt(OpCall("arrive", [self.empty_sem, RawExpr(self.num_consumers)])))
        )
        return SeqStmt(stmts)
    
    def async_load_global(self, loads: List[MemLoad]) -> Stmt:
        num_bytes = SizeBytesExpr(self.tiles[0])
        for tile_idx in range(1, len(self.tiles)):
            num_bytes = BinaryOp(num_bytes, SizeBytesExpr(self.tiles[tile_idx]), "+")
        stmts: Sequence[Stmt] = [
            ExprStmt(OpCall("tma::expect_bytes", [self.full_sem, num_bytes]))
        ] + [
            ExprStmt(OpCall("tma::load_async", [load.dest, load.source, load.coord, self.full_sem])) for load in loads if load.dest in self.tiles
        ]
        return lane0_if(SeqStmt(stmts))

    def wait_full(self, level):
        stmts = [
            ExprStmt(OpCall("wait", [self.full_sem, self.full_tic])),
            AssignExpr(self.full_tic, BinaryOp(self.full_tic, RawExpr("1"), "^")).to_stmt(),
        ]
        if level == Level.block:
            stmts.append(OpCall("__syncthreads", []).to_stmt())
        return SeqStmt(stmts)
    
    def wait_empty(self, level):
        stmts = [
            ExprStmt(OpCall("wait", [self.empty_sem, self.empty_tic])),
            AssignExpr(self.empty_tic, BinaryOp(self.empty_tic, RawExpr("1"), "^")).to_stmt()
        ]
        if level == Level.block:
            stmts.append(OpCall("__syncthreads", []).to_stmt())
        return SeqStmt(stmts)

    def arrive_empty(self):
        return lane0_if(OpCall("arrive", [self.empty_sem, RawExpr(1)]).to_stmt())
        
KITTENS_TEMPLATE = """
#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include "kittens.cuh"

using namespace kittens;

{{ constants }}

struct global_vars {
    {%- for var in kernel_vars %}
    {{ var.declare() }}
    {%- endfor %}
};

__global__ void kernel(const __grid_constant__ global_vars globals) {
    {{ kernel_stmt }}
}

{{ launch_code }}
"""
class Program:
    def __init__(self, kernel_vars: KernelGlobals, kernel_stmt: Stmt, 
                 constants: str = "", launch_code: str = ""
                ):
        self.kernel_vars = kernel_vars
        self.kernel_stmt = kernel_stmt
        self.constants = constants
        self.launch_code = launch_code
    
    def __str__(self):
        template = Template(KITTENS_TEMPLATE)
        return template.render(
            constants=self.constants,
            kernel_stmt=self.kernel_stmt,
            kernel_vars=self.kernel_vars.vars,
            launch_code=self.launch_code
        )
