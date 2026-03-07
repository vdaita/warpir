from .dataflow_graph import *
from .flow import *
import copy
from typing import Dict

def modulo_schedule_loop(body: DataflowGraph,):    
    # do some kind of ssa things
    new_body = body.subscope() 

    # iterate through each op in ops to get what you want
    ii = 1
    unversioned_graph = body.get_unversioned_graph()
    for node in unversioned_graph.nodes():
        for successor in unversioned_graph.succ[node]:
            distance = unversioned_graph.edges[node, successor].get('distance', 0)
            latency = node.latency
            ii = max(ii, (latency + distance - 1) // max(distance, 1))

    slot_usage = {
        Resource.TMA: [0] * ii,
        Resource.MMA: [0] * ii,
        Resource.CUDA: [0] * ii,
    }
    assigned: Dict[Instruction, int]  = {}

    inst_graph = body.get_instruction_graph()

    for instruction in body.instructions:
        op = instruction.source_op
        earliest = 0
        for pred in inst_graph.pred[instruction]:
            assert type(pred) == Instruction
            earliest = max(earliest, assigned[pred] + pred.source_op.latency)

        cycle = earliest
        while slot_usage[op.resource][cycle % ii]:
            cycle += 1
        
        assigned[instruction] = cycle
        slot_usage[op.resource][cycle % ii] = 1

    max_cycle = max(assigned.values())
    num_stages = max_cycle // ii + 1
    buffers = []
    for producer, consumer in inst_graph.edges():
        p_cycle = assigned[producer]
        c_cycle = assigned[consumer]
        distance = c_cycle - p_cycle