def pipeline(DataflowGraph: DataflowGraph, n: int):
    ...
    # for each for statement, because the statements are formatted to take in i as an input, 
        # generate a version where you have i and i + 1
        # generate if statements to do the first part on it's own
        # and if statements to make sure that you don't do the last part if it goes over

        # get the DAG
        # (ambiguous) how do you figure out which buffers can be split out into using two buffers?
        # split out into n buffers according ot the pipeline instruction

        # eventually, this will have to use a proper cost modek
        
    # with this new DataflowGraph, run get_stmt()

def warp_specialize(DataflowGraph: DataflowGraph):
    ...