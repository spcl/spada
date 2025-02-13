from spatialstencil.optimizations.spatial_reduce import ReduceOptimizer
from spatialstencil.optimizations.spatial_broadcast import BroadcastOptimizer



def optimization_pass(program):
    """
    Runs the spatial optimizations on the program.
    """
    broadcast_optimizer = BroadcastOptimizer(program)
    pass_1 = broadcast_optimizer.broadcast_subroutine()
    reduce_optimizer = ReduceOptimizer(pass_1)
    pass_2 = reduce_optimizer.reduce_subroutine()
    return pass_2