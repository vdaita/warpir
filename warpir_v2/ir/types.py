from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


class GPUType(str, Enum):
    bf16 = "bf16"
    fp32 = "fp32"


class SharedTileLayout(str, Enum):
    row_major = "ducks::st_layout::row"
    col_major = "ducks::st_layout::col"


class RegTileLayout(str, Enum):
    row_major = "ducks::rt_layout::row"
    col_major = "ducks::rt_layout::col"


@dataclass(frozen=True)
class SharedTileType:
    data_type: GPUType
    rows: int
    cols: int
    layout: SharedTileLayout

    def __str__(self) -> str:
        tile_ctor = "st_bf" if self.data_type == GPUType.bf16 else "st_fl"
        if self.layout == SharedTileLayout.row_major:
            return f"{tile_ctor}<{self.rows}, {self.cols}>"
        return f"{tile_ctor}<{self.rows}, {self.cols}, {self.layout.value}>"


@dataclass(frozen=True)
class RegTileType:
    data_type: GPUType
    rows: int
    cols: int
    layout: RegTileLayout

    def __str__(self) -> str:
        tile_ctor = "rt_bf" if self.data_type == GPUType.bf16 else "rt_fl"
        return f"{tile_ctor}<{self.rows}, {self.cols}, {self.layout.value}>"


@dataclass(frozen=True)
class GlobalType:
    """
      Represents a global (tiled) input tensor

      Attributes
      ----------
    """

    data_type: GPUType
    sub_tile_type: SharedTileType
    batch: int = -1
    depth: int = -1
    rows: int = -1
    cols: int = -1

    def __str__(self) -> str:
        return f"gl<{self.data_type.value}, {self.batch}, {self.depth}, {self.rows}, {self.cols}, {self.sub_tile_type}>"


@dataclass(frozen=True)
class ScalarType:
    name: str

    def __str__(self) -> str:
        return self.name

class IntType(ScalarType):
    def __init__(self):
        super().__init__("int")

    def __str__(self) -> str:
        return "int"


TypeRef = Union[GlobalType, IntType, ScalarType, SharedTileType, RegTileType]