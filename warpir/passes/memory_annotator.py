"""
Take annotations of memory being asynchronous or not and lower it properly.
"""
from warpir.ir import Kernel

# TODO: take the output of a warp-assignment graph and make sure that all async-tagged elements are marked as true or false

def process_memory_annotations(kernel: Kernel):
    # iterate through the body and add wait ops before asynchronous elements are added in 
    new_ops = []
    
    
    for op in kernel.body:
        if is_async(op):
            ...
            # where is the first use of that operation?