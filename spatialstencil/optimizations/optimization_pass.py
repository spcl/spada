from spatialstencil.optimizations.spatial_reduce import ReduceOptimizer



def optimization_pass(program):
    """
    Runs the spatial optimizations on the program.
    """
    reduce_optimizer = ReduceOptimizer(program)
    out = reduce_optimizer.reduce_subroutine()
    return out