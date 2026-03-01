from enum import Enum
from typing import Optional, Union
from abc import ABC, abstractmethod

class GPUType(str, Enum):
    bf16 = "bf16"
    fp32 = "fp32"

class RegTileLayout(str, Enum):
    row_major = "ducks::rt_layout::row"
    col_major = "ducks::rt_layout::col"

class SharedTileLayout(str, Enum):
    row_major = "ducks::st_layout::row"
    col_major = "ducks::st_layout::col"

class SharedTileType:
    def __init__(self, data_type: GPUType, tile_w: int, tile_h: int, layout: SharedTileLayout):
        self.data_type = data_type
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.layout = layout
    def __str__(self):
        if self.data_type == GPUType.bf16:
            return f"st_bf<{self.tile_w}, {self.tile_h}, {self.layout}>"
        else:
            return f"st_fl<{self.tile_w}, {self.tile_h}, {self.layout}>"

class SharedVecType:
    def __init__(self, data_type: GPUType, length: int):
        self.data_type = data_type
        self.length = length

    def __str__(self):
        if self.data_type == GPUType.bf16:
            return f"sv_bf<{self.length}>"
        return f"sv_fl<{self.length}>"

class GlobalType:
    def __init__(self, data_type: GPUType, sub_tile: SharedTileType, batch_dim: Optional[int] = -1, depth_dim: Optional[int] = -1, height_dim: Optional[int] = -1, width_dim: Optional[int] = -1):
        self.data_type = data_type
        self.sub_tile = sub_tile
    
        self.batch_dim = batch_dim
        self.depth_dim = depth_dim
        self.height_dim = height_dim
        self.width_dim = width_dim
    def __str__(self):
        return f"gl<{self.data_type}, {self.batch_dim}, {self.depth_dim}, {self.height_dim}, {self.width_dim}, {self.sub_tile}>"

class RegTileType:
    def __init__(self, data_type: GPUType, tile_w: int, tile_h: int, layout: RegTileLayout):
        self.data_type = data_type
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.layout = layout
    def __str__(self):
        if self.data_type == GPUType.bf16:
            return f"rt_bf<{self.tile_w}, {self.tile_h}, {self.layout}>"
        else:
            return f"rt_fl<{self.tile_w}, {self.tile_h}, {self.layout}>"

class RegVecType:
    def __init__(self, data_type: GPUType, length: int):
        self.data_type = data_type
        self.length = length

    def __str__(self):
        if self.data_type == GPUType.bf16:
            return f"rv_bf<{self.length}>"
        return f"rv_fl<{self.length}>"

class ScalarType:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

VarType = Union[
    GlobalType,
    SharedTileType,
    RegTileType,
    SharedVecType,
    RegVecType,
    ScalarType,
    str,
]

class Var():
    def __init__(self, name: str, var_type: VarType):
        self.name = name
        self.var_type = var_type
    
    def define(self):
        return f"{self.var_type} {self.name};"
    
    def use(self):
        return self.name

    def __str__(self):
        return self.name

from .ops import ExprBase
for name in ["__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__div__", "__truediv__", "__mod__", "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"]:
    setattr(Var, name, getattr(ExprBase, name))
