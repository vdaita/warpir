from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


class GPUType(str, Enum):
    bf16 = "bf16"
    fp32 = "fp32"


class TileLayout(str, Enum):
    row_major = "row_major"
    col_major = "col_major"


@dataclass(frozen=True)
class SharedTileType:
    data_type: GPUType
    rows: int
    cols: int
    layout: TileLayout = TileLayout.row_major

    def __str__(self) -> str:
        return f"shared_tile<{self.data_type.value}, {self.rows}x{self.cols}, {self.layout.value}>"


@dataclass(frozen=True)
class RegTileType:
    data_type: GPUType
    rows: int
    cols: int
    layout: TileLayout = TileLayout.row_major

    def __str__(self) -> str:
        return f"reg_tile<{self.data_type.value}, {self.rows}x{self.cols}, {self.layout.value}>"


@dataclass(frozen=True)
class GlobalType:
    data_type: GPUType
    sub_tile_type: SharedTileType
    batch: int = -1
    depth: int = -1
    rows: int = -1
    cols: int = -1

    def __str__(self) -> str:
        return f"global<{self.data_type.value}, {self.batch}, {self.depth}, {self.rows}, {self.cols}>"


@dataclass(frozen=True)
class SharedBufferType:
    """Mutable shared memory buffer array (not an SSA value)."""
    tile_type: SharedTileType
    count: int

    def __str__(self) -> str:
        return f"shared_buffer<{self.tile_type}, count={self.count}>"


@dataclass(frozen=True)
class ColVecType:
    """Register column vector — col_vec<rt<data_type, rows, cols>>."""
    data_type: GPUType
    rows: int
    cols: int

    def __str__(self) -> str:
        return f"col_vec<{self.data_type.value}, {self.rows}x{self.cols}>"


@dataclass(frozen=True)
class ScalarType:
    name: str

    def __str__(self) -> str:
        return self.name


class IntType(ScalarType):
    def __init__(self):
        super().__init__("int")


TypeRef = Union[GlobalType, IntType, SharedTileType, RegTileType, SharedBufferType, ColVecType]
