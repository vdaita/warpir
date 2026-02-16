from typing import List
from abc import ABC, abstractmethod
from jinja2 import Template, Environment
from layouts import Var

class Stmt(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

class ForStmt(Stmt): # has to accept a lambda expression that takes the iterator state and determines: how to increment 
    ...

class WhileStmt(Stmt):
    ...

class NoStmt(Stmt):
    def __init__():
        pass
    
    def __str__(self) -> str:
        return ""

class SeqStmt(Stmt):
    def __init__(self, stmts: List[Stmt]):
        self.stmts = stmts
    
    def __str__(self):
        return "\n".join([str(stmt) for stmt in self.stmts])

class Warpgroup: # TODO: implement a full, functional warpgroup object that allows for splitting tasks
    def __init__(self, warp_id: int, wg_vars: List[Var], stmt: Stmt):
        self.id = warp_id
        self.wg_vars = wg_vars
        self.stmt = stmt

    ... 

KITTENS_TEMPLATE = """
#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include "kittens.cuh"

using namespace kittens;

struct kernel_globals {
    {%- for var in kernel_vars %}
    {{ var.define() }};
    {%- endfor %}
};

__global__ void kernel(const __grid_constant__ kernel_globals g) {
    {{ kernel_stmt }}
}
"""
class Program:
    def __init__(self, input_vars: List[Var], kernel_vars: List[Var], kernel_stmt: Stmt):
        self.input_vars = input_vars
        self.kernel_vars = kernel_vars
        self.kernel_stmt = kernel_stmt
    
    def __str__(self):
        t = Template(KITTENS_TEMPLATE, trim_blocks=True, lstrip_blocks=True)
        return t.render(
            kernel_vars=self.kernel_vars,
            kernel_stmt=self.kernel_stmt
        )