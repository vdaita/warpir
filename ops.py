from typing import Union

from layouts import SharedTileType, RegTileType

class UnaryOp:
    ...

class BinaryOp:
    ...

class ConstantOp:
    ...

class ZeroOp:
    ...

class LoadOp(BinaryOp):
    ...

class TMALoadOp:
    ...

class MMAOp:
    ...

class StoreOp: # going to be just like the LoadOp
    ...

class TMAStoreOp:
    ...