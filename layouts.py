from enum import Enum
from typing import Optional
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
    pass

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
    pass

class Var(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

class GlobalVar(Var):
    def __init__(self, name: str, var_type: GlobalType):
        self.name = name
        self.var_type = var_type
    
    def define(self):
        ...
    
    def use(self):
        ...

class SharedVar(Var):
    def __init__(self, name: str, var_type: SharedTileType):
        ...
    
    def define(self):
        ...
    
    def use(self):
        ...

class RegVar(Var):
    def __init__(self, name: str):
        ...

    def define(self):
        ...
    
    def use(self):
        ...

class IntVar(Var):
    def __init__(self, name: str) -> None:
        ...
    
    def define(self):
        ...
    
    def use(self):
        ...

# class WarpTile:
#     def __init__(self, tile_layout: SharedLayout):
#         ...
#         # initialize semaphores for empty / full (writers have to wait for empty, consumers have to wait for full)
    
#     def consume(self) -> Var:
#         ...
    
#     def produce(self) -> Var:
#         ...