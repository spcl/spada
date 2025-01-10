from spatialstencil.syntax.spatial_ir.irnodes import Kernel, ComputeBlock, ReduceStatement, Expression, SubgridExpression, RangeExpression, ConstantLiteral, ScalarType, DataflowBlock, MulStreamDeclaration, ReduceRoutingDeclaration, RoutingDeclaration, RoutingHop, StreamType, Identifier, TypedIdentifier, ForeachStatement, ArraySlice, BinaryOperator, SendStatement, ReceiveGenerator, AssignmentStatement, RelativeStreamDeclaration, PlaceBlock, Phase, Parameter, KernelArgument, ReceiveStatement, ForStatement
from typing import Union, Tuple, Optional, Literal
import spatialstencil.syntax.spatial_ir.irnodes as spa
from spatialstencil.lowering.versioning import Versioning
from spatialstencil.syntax.common.visitor import ScopedIRNodeVisitor, IRNodeVisitor
# TODO from spatialstencil.syntax.spatial_ir.grid_geometry import Rectangle
# try ScopedIRNodeVisitor / IRNodeVisitor from spatialstencil.syntax.common.visitor to match nodes that have reduce in them


class ReduceOptimizer():
    name: str | None
    parameters: list[Parameter]
    arguments: list[KernelArgument]
    body: list[PlaceBlock | DataflowBlock | ComputeBlock | Phase]
    _communication_patterns: Optional[dict[str, dict[tuple[int, int], list[list[list[int]]]]]] = None
    reduce_operations: dict[str, dict[str, Union[int, Literal['OP_SUM'], list[int]]]] = {}  # needs to be adapted
    grid_streams: dict[str, list[list]] = {}
    snake_streams: dict[str, list[list]] = {}
    pipelined: dict[str, bool] = {}


    def __init__(self, kernel: Kernel) -> None:
        self.name = kernel.name
        self.parameters = kernel.parameters
        self.arguments = kernel.arguments
        self.body = kernel.body
        self.versioning = Versioning[spa.Identifier](spa.Identifier)
        return None

    def reduce_subroutine(self) -> Kernel:
        self.change_data_blocks()
        self.fix_subgrid()
        self.change_compute_blocks()
        return Kernel(name=self.name, parameters=self.parameters, arguments=self.arguments, body=self.body)
    
    

    def create_communication_patterns(self, x_start, x_stop, y_start, y_stop, x, y, name, graph, pipelined) -> None:
        if x < x_start or x >= x_stop or y < y_start or y >= y_stop:
            if x == x_stop or y == y_stop:
                raise ValueError(f"The communication point (x, y) = ({x}, {y}) is not within the subgrid" +
                             f"[x_start, x_stop, y_start, y_stop] = [{x_start}, {x_stop}, {y_start}, {y_stop}] for the operation {name}." +
                             f" Remember that the stop value is exclusive.")
            raise ValueError(f"The communication point (x, y) = ({x}, {y}) is not within the subgrid" +
                             f"[x_start, x_stop, y_start, y_stop] = [{x_start}, {x_stop}, {y_start}, {y_stop}] for the operation {name}.")
        communication = []
        self.pipelined.update({name : False}) # not implemented yet
        mode = graph
        if mode == 'snake':
            if y == y_start:
                if (y_stop - 1 - y_start) % 2 == 0:

                    # horizontal movement
                    if pipelined:
                        # not completely correct for small subgrids
                        if x_stop - x_start > 1: ## this should be handled differently
                            if (x_stop - x_start) % 2 != 0:
                                communication.append([x_start, x_stop - 1, y_start, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop, y_start, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start, x_stop - 1, y_start + 1, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop, y_start + 1, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])
                            else:
                                communication.append([x_start, x_stop, y_start, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop - 1, y_start, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start, x_stop, y_start + 1, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop - 1, y_start + 1, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])
                        else:
                            # effectively not pipelined in x direction as we have a column
                            # still pipelined in y direction
                            # communication.append([x_start, x_stop, y_start, y_stop, -1 if x == x_start else 1, 0, 1, 1])
                            # communication.append([x_start, x_stop, y_start + 1, y_stop - 1, 1 if x == x_start else -1, 0, 1, 1])
                            pass
                    else:
                        if x_stop - x_start > 1:
                            communication.append([x_start, x_stop, y_start, y_stop, -1 if x == x_start else 1, 0, 1, 1])
                            communication.append([x_start, x_stop, y_start + 1, y_stop - 1, 1 if x == x_start else -1, 0, 1, 1])

                    # vertical movement
                    # not dependent on pipelined as if we have a column it's already pipelined
                    if x == x_start:
                        # print('upper left corner odd')
                        if y_stop - y_start > 2:
                            communication.append([x_start, x_start + 1, y_start + 1, y_stop , 0, -1, 1, 1])
                        if y_stop - y_start > 1:
                            communication.append([x_stop - 1, x_stop, y_start, y_stop - 1, 0, -1, 1, 1])
                    if x == x_stop - 1:
                        # print('upper right corner odd')
                        if y_stop - y_start > 2:
                            communication.append([x_stop - 1, x_stop, y_start + 1, y_stop, 0, -1, 1, 1])
                        if y_stop - y_start > 1:
                            communication.append([x_start, x_start + 1, y_start, y_stop - 1, 0, -1, 1, 1])
                else:

                    # horizontal movement
                    if pipelined:
                        if x_stop - x_start > 1:
                            if (x_stop - x_start) % 2 != 0:
                                communication.append([x_start, x_stop - 1, y_start, y_stop - 1, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop, y_start, y_stop - 1, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start, x_stop - 1, y_start + 1, y_stop, 1 if x == x_start else -1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop, y_start + 1, y_stop, 1 if x == x_start else -1, 0, 1, 2])
                            else:
                                communication.append([x_start, x_stop, y_start, y_stop - 1, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop - 1, y_start, y_stop - 1, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start, x_stop, y_start + 1, y_stop, 1 if x == x_start else -1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop - 1, y_start + 1, y_stop, 1 if x == x_start else -1, 0, 1, 2])
                        else:
                            # effectively not pipelined in x direction as we have a column
                            # still pipelined in y direction
                            # communication.append([x_start, x_stop, y_start, y_stop - 1, -1 if x == x_start else 1, 0, 2, 2])
                            # communication.append([x_start, x_stop, y_start + 1, y_stop, 1 if x == x_start else -1, 0, 2, 2])
                            pass
                    else:
                        if x_stop - x_start > 1:
                            communication.append([x_start, x_stop, y_start, y_stop - 1, -1 if x == x_start else 1, 0, 1, 2])
                            communication.append([x_start, x_stop, y_start + 1, y_stop, 1 if x == x_start else -1, 0, 1, 2])

                    # vertical movement
                    # not dependent on pipelined as if we have a column it's already pipelined
                    if x == x_start:
                        # print('upper left corner even')
                        if y_stop - y_start > 2:
                            communication.append([x_start, x_start + 1, y_start + 1, y_stop - 1, 0, -1, 1, 1])
                        if y_stop - y_start > 1:
                            communication.append([x_stop - 1, x_stop, y_start, y_stop, 0, -1, 1, 1])
                    if x == x_stop - 1:
                        # print('upper right corner even')
                        if y_stop - y_start > 2:
                            communication.append([x_stop - 1, x_stop, y_start + 1, y_stop - 1, 0, -1, 1, 1])
                        if y_stop - y_start > 1:
                            communication.append([x_start, x_start + 1, y_start, y_stop, 0, -1, 1, 1])

            elif y == y_stop - 1:
                if (y_stop - 1 - y_start) % 2 == 0:

                    # horizontal movement
                    if pipelined:
                        if x_stop - x_start > 1:
                            if (x_stop - x_start) % 2 != 0:
                                communication.append([x_start, x_stop - 1, y_start, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop, y_start, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start, x_stop - 1, y_start + 1, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop, y_start + 1, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])
                            else:
                                communication.append([x_start, x_stop, y_start, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop - 1, y_start, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start, x_stop, y_start + 1, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop - 1, y_start + 1, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])
                        else:
                            # effectively not pipelined in x direction as we have a column
                            # still pipelined in y direction
                            # communication.append([x_start, x_stop, y_start, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                            # communication.append([x_start, x_stop, y_start + 1, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])
                            pass
                    else:
                        if x_stop - x_start > 1:
                            communication.append([x_start, x_stop, y_start, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                            communication.append([x_start, x_stop, y_start + 1, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])

                    # vertical movement
                    # not dependent on pipelined as if we have a column it's already pipelined
                    if x == x_start:
                        # print('lower left corner odd')
                        if y_stop - y_start > 2:
                            communication.append([x_start, x_start + 1, y_start, y_stop - 1, 0, 1, 1, 1])
                        if y_stop - y_start > 1:
                            communication.append([x_stop - 1, x_stop, y_start + 1, y_stop, 0, 1, 1, 1])
                    if x == x_stop - 1:
                        # print('lower right corner odd')
                        if y_stop - y_start > 2:
                            communication.append([x_stop - 1, x_stop, y_start, y_stop - 1, 0, 1, 1, 1])
                        if y_stop - y_start > 1:
                            communication.append([x_start, x_start + 1, y_start + 1, y_stop, 0, 1, 1, 1])
                else:

                    # horizontal movement
                    if pipelined:
                        if x_stop - x_start > 1:
                            if (x_stop - x_start) % 2 != 0:
                                communication.append([x_start, x_stop - 1, y_start + 1, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop, y_start + 1, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start, x_stop - 1, y_start, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop, y_start, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])
                            else:
                                communication.append([x_start, x_stop, y_start + 1, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop - 1, y_start + 1, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                                communication.append([x_start, x_stop, y_start, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])
                                communication.append([x_start + 1, x_stop - 1, y_start, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])
                        else:
                            # effectively not pipelined in x direction as we have a column
                            # still pipelined in y direction
                            # communication.append([x_start, x_stop, y_start + 1, y_stop, -1 if x == x_start else 1, 0, 2, 2])
                            # communication.append([x_start, x_stop, y_start, y_stop - 1, 1 if x == x_start else -1, 0, 2, 2])
                            pass
                    else: 
                        communication.append([x_start, x_stop, y_start + 1, y_stop, -1 if x == x_start else 1, 0, 1, 2])
                        communication.append([x_start, x_stop, y_start, y_stop - 1, 1 if x == x_start else -1, 0, 1, 2])

                    # vertical movement
                    # not dependent on pipelined as if we have a column it's already pipelined
                    if x == x_start:
                        # print('lower left corner even')
                        if y_stop - y_start > 2:
                            communication.append([x_start, x_start + 1, y_start + 1, y_stop - 1, 0, 1, 1, 1])
                        if y_stop - y_start > 1:
                            communication.append([x_stop - 1, x_stop, y_start, y_stop, 0, 1, 1, 1])
                    if x == x_stop - 1:
                        # print('lower right corner even')
                        if y_stop - y_start > 2:
                            communication.append([x_stop - 1, x_stop, y_start + 1, y_stop - 1, 0, 1, 1, 1])
                        if y_stop - y_start > 1:
                            communication.append([x_start, x_start + 1, y_start, y_stop, 0, 1, 1, 1])
            else:
                raise NotImplementedError("Only the corners are implemented for 'snake'")
            
            self.snake_streams.update({name: communication})

        elif mode == 'grid':
            # TODO add steps for pipelined communication
            if x == x_start:
                # horizontal movement
                if x_start == x_stop - 1:
                    # print('no horizontal movement needed')
                    pass
                elif x_stop - x_start == 2:
                    # print('right to left')
                    communication.append([x_start, x_stop, y_start, y_stop, -1, 0, 1, 1])
                else:
                    if pipelined:
                        if (x_stop - x_start) % 2 == 0:
                            communication.append([x_start, x_stop, y_start, y_stop, -1, 0, 1, 1])
                            communication.append([x_start + 1, x_stop - 1, y_start, y_stop, -1, 0, 1, 1])
                        else:
                            communication.append([x_start, x_stop - 1, y_start, y_stop, -1, 0, 1, 1])
                            communication.append([x_start + 1, x_stop, y_start, y_stop, -1, 0, 1, 1])
                    else:
                        communication.append([x_start, x_stop, y_start, y_stop, -1, 0, 1, 1])
                # TODO add steps for pipelined communication from here

                # vertical movement
                if y_start == y_stop - 1:
                    # print('no vertical movement needed')
                    pass
                elif y == y_start:
                    # print('upper left corner')
                    if not pipelined or y_stop - y_start <= 2:
                        communication.append([x_start, x_start + 1, y_start, y_stop, 0, -1, 1, 1])
                    else:
                        if (y_stop - y_start) % 2 == 0:
                            communication.append([x_start, x_start + 1, y_start, y_stop, 0, -1, 1, 1])
                            communication.append([x_start, x_start + 1, y_start + 1, y_stop - 1, 0, -1, 1, 1])
                        else:
                            communication.append([x_start, x_start + 1, y_start, y_stop - 1, 0, -1, 1, 1])
                            communication.append([x_start, x_start + 1, y_start + 1, y_stop, 0, -1, 1, 1])
                elif y == y_stop - 1:
                    # print('lower left corner')
                    if not pipelined or y_stop - y_start <= 2:
                        communication.append([x_start, x_start + 1, y_start, y_stop, 0, 1, 1, 1])
                    else:
                        if (y_stop - y_start) % 2 == 0:
                            communication.append([x_start, x_start + 1, y_start, y_stop, 0, 1, 1, 1])
                            communication.append([x_start, x_start + 1, y_start + 1, y_stop - 1, 0, 1, 1, 1])
                        else:
                            communication.append([x_start, x_start + 1, y_start, y_stop - 1, 0, 1, 1, 1])
                            communication.append([x_start, x_start + 1, y_start + 1, y_stop, 0, 1, 1, 1])
                else:
                    # print('left edge')
                    if not pipelined:
                        communication.append([x_start, x_start + 1, y_start, y + 1, 0, 1, 1, 1])
                        communication.append([x_start, x_start + 1, y, y_stop, 0, -1, 1, 1])
                    else:
                        # upper part
                        if (y - y_start) >= 2: # y is inclusive while y_stop is exclusive
                            if (y - y_start) % 2 == 0:
                                communication.append([x_start, x_start + 1, y_start, y, 0, 1, 1, 1])
                                communication.append([x_start, x_start + 1, y_start + 1, y + 1, 0, 1, 1, 1])
                            else:
                                communication.append([x_start, x_start + 1, y_start, y + 1, 0, 1, 1, 1])
                                communication.append([x_start, x_start + 1, y_start + 1, y, 0, 1, 1, 1])
                        else:
                            communication.append([x_start, x_start + 1, y_start, y + 1, 0, 1, 1, 1])

                        # lower part
                        if (y_stop - y) > 2:
                            if (y_stop - y) % 2 == 0:
                                communication.append([x_start, x_start + 1, y, y_stop, 0, -1, 1, 1])
                                communication.append([x_start, x_start + 1, y + 1, y_stop - 1, 0, -1, 1, 1])
                            else:
                                communication.append([x_start, x_start + 1, y, y_stop - 1, 0, -1, 1, 1])
                                communication.append([x_start, x_start + 1, y + 1, y_stop, 0, -1, 1, 1])
                        else:
                            communication.append([x_start, x_start + 1, y, y_stop, 0, -1, 1, 1])

                        

            elif x == x_stop - 1:
                # horizontal movement
                if x_start == x_stop - 1:
                    # print('no horizontal movement needed')
                    pass
                elif x_stop - x_start == 2:
                    # print('left to right')
                    communication.append([x_start, x_stop, y_start, y_stop, 1, 0, 1, 1])
                else:
                    # print('left to right')
                    if pipelined:
                        if (x_stop - x_start) % 2 == 0:
                            communication.append([x_start, x_stop, y_start, y_stop, 1, 0, 1, 1])
                            communication.append([x_start + 1, x_stop - 1, y_start, y_stop, 1, 0, 1, 1])
                        else:
                            communication.append([x_start, x_stop - 1, y_start, y_stop, 1, 0, 1, 1])
                            communication.append([x_start + 1, x_stop, y_start, y_stop, 1, 0, 1, 1])
                    else:
                        communication.append([x_start, x_stop, y_start, y_stop, 1, 0, 1, 1])

                # vertical movement
                if y_start == y_stop - 1:
                    # print('no vertical movement needed')
                    pass
                elif y == y_start:
                    # print('upper right corner')
                    if not pipelined or y_stop - y_start <= 2:
                        communication.append([x_stop - 1, x_stop, y_start, y_stop, 0, -1, 1, 1])
                    else:
                        if (y_stop - y_start) % 2 == 0:
                            communication.append([x_stop - 1, x_stop, y_start, y_stop, 0, -1, 1, 1])
                            communication.append([x_stop - 1, x_stop, y_start + 1, y_stop - 1, 0, -1, 1, 1])
                        else:
                            communication.append([x_stop - 1, x_stop, y_start, y_stop - 1, 0, -1, 1, 1])
                            communication.append([x_stop - 1, x_stop, y_start + 1, y_stop, 0, -1, 1, 1])
                elif y == y_stop - 1:
                    # print('lower right corner')
                    if not pipelined or y_stop - y_start <= 2:
                        communication.append([x_stop - 1, x_stop, y_start, y_stop, 0, 1, 1, 1])
                    else:
                        if (y_stop - y_start) % 2 == 0:
                            communication.append([x_stop - 1, x_stop, y_start, y_stop, 0, 1, 1, 1])
                            communication.append([x_stop - 1, x_stop, y_start + 1, y_stop - 1, 0, 1, 1, 1])
                        else:
                            communication.append([x_stop - 1, x_stop, y_start, y_stop - 1, 0, 1, 1, 1])
                            communication.append([x_stop - 1, x_stop, y_start + 1, y_stop, 0, 1, 1, 1])
                else:
                    # print('right edge')
                    if not pipelined:
                        communication.append([x_stop - 1, x_stop, y_start, y + 1, 0, 1, 1, 1])
                        communication.append([x_stop - 1, x_stop, y, y_stop, 0, -1, 1, 1])
                    else:
                        # upper part
                        if (y - y_start) >= 2: # y is inclusive while y_stop is exclusive
                            if (y - y_start) % 2 == 0:
                                communication.append([x_stop - 1, x_stop, y_start, y, 0, 1, 1, 1])
                                communication.append([x_stop - 1, x_stop, y_start + 1, y + 1, 0, 1, 1, 1])
                            else:
                                communication.append([x_stop - 1, x_stop, y_start, y + 1, 0, 1, 1, 1])
                                communication.append([x_stop - 1, x_stop, y_start + 1, y, 0, 1, 1, 1])
                        else:
                            communication.append([x_stop - 1, x_stop, y_start, y + 1, 0, 1, 1, 1])

                        # lower part
                        if (y_stop - y) > 2:
                            if (y_stop - y) % 2 == 0:
                                communication.append([x_stop - 1, x_stop, y, y_stop, 0, -1, 1, 1])
                                communication.append([x_stop - 1, x_stop, y + 1, y_stop - 1, 0, -1, 1, 1])
                            else:
                                communication.append([x_stop - 1, x_stop, y, y_stop - 1, 0, -1, 1, 1])
                                communication.append([x_stop - 1, x_stop, y + 1, y_stop, 0, -1, 1, 1])
                        else:
                            communication.append([x_stop - 1, x_stop, y, y_stop, 0, -1, 1, 1])

            else:
                # horizontal movement
                # print('middle')
                if not pipelined:
                    communication.append([x_start, x + 1, y_start, y_stop, 1, 0, 1, 1]) # left to middle
                    communication.append([x, x_stop, y_start, y_stop, -1, 0, 1, 1]) # right to middle
                else:
                    # left
                    if (x - x_start) >= 2: # x is inclusive while x_stop is exclusive
                        if (x - x_start) % 2 == 0:
                            communication.append([x_start, x, y_start, y_stop, 1, 0, 1, 1])
                            communication.append([x_start + 1, x + 1, y_start, y_stop, 1, 0, 1, 1])
                        else:
                            communication.append([x_start, x + 1, y_start, y_stop, 1, 0, 1, 1])
                            communication.append([x_start + 1, x, y_start, y_stop, 1, 0, 1, 1])
                    else:
                        communication.append([x_start, x + 1, y_start, y_stop, 1, 0, 1, 1])

                    # right
                    if (x_stop - x) > 2:
                        if (x_stop - x) % 2 == 0:
                            communication.append([x, x_stop, y_start, y_stop, -1, 0, 1, 1])
                            communication.append([x + 1, x_stop - 1, y_start, y_stop, -1, 0, 1, 1])
                        else:
                            communication.append([x, x_stop - 1, y_start, y_stop, -1, 0, 1, 1])
                            communication.append([x + 1, x_stop, y_start, y_stop, -1, 0, 1, 1])
                    else:
                        communication.append([x, x_stop, y_start, y_stop, -1, 0, 1, 1])

                # vertical movement
                if y_start == y_stop - 1:
                    # print('no vertical movement needed')
                    pass
                elif y == y_start:
                    # print('upper edge')
                    if not pipelined or y_stop - y_start <= 2:
                        communication.append([x, x + 1, y_start, y_stop, 0, -1, 1, 1])
                    else:
                        if (y_stop - y_start) % 2 == 0:
                            communication.append([x, x + 1, y_start, y_stop, 0, -1, 1, 1])
                            communication.append([x, x + 1, y_start + 1, y_stop - 1, 0, -1, 1, 1])
                        else:
                            communication.append([x, x + 1, y_start, y_stop - 1, 0, -1, 1, 1])
                            communication.append([x, x + 1, y_start + 1, y_stop, 0, -1, 1, 1])
                elif y == y_stop - 1:
                    # print('lower edge')
                    if not pipelined or y_stop - y_start <= 2:
                        communication.append([x, x + 1, y_start, y_stop, 0, 1, 1, 1])
                    else:
                        if (y_stop - y_start) % 2 == 0:
                            communication.append([x, x + 1, y_start, y_stop, 0, 1, 1, 1])
                            communication.append([x, x + 1, y_start + 1, y_stop - 1, 0, 1, 1, 1])
                        else:
                            communication.append([x, x + 1, y_start, y_stop - 1, 0, 1, 1, 1])
                            communication.append([x, x + 1, y_start + 1, y_stop, 0, 1, 1, 1])
                else:
                    # print('center')
                    if not pipelined:
                        communication.append([x, x + 1, y_start, y + 1, 0, 1, 1, 1])
                        communication.append([x, x + 1, y, y_stop, 0, -1, 1, 1])
                    else:
                        # upper part
                        if (y - y_start) >= 2: # y is inclusive while y_stop is exclusive
                            if (y - y_start) % 2 == 0:
                                communication.append([x, x + 1, y_start, y, 0, 1, 1, 1])
                                communication.append([x, x + 1, y_start + 1, y + 1, 0, 1, 1, 1])
                            else:
                                communication.append([x, x + 1, y_start, y + 1, 0, 1, 1, 1])
                                communication.append([x, x + 1, y_start + 1, y, 0, 1, 1, 1])
                        else:
                            communication.append([x, x + 1, y_start, y + 1, 0, 1, 1, 1])

                        # lower part
                        if (y_stop - y) > 2:
                            if (y_stop - y) % 2 == 0:
                                communication.append([x, x + 1, y, y_stop, 0, -1, 1, 1])
                                communication.append([x, x + 1, y + 1, y_stop - 1, 0, -1, 1, 1])
                            else:
                                communication.append([x, x + 1, y, y_stop - 1, 0, -1, 1, 1])
                                communication.append([x, x + 1, y + 1, y_stop, 0, -1, 1, 1])
                        else:
                            communication.append([x, x + 1, y, y_stop, 0, -1, 1, 1])
        
            self.grid_streams.update({name : communication})

        else:
            raise NotImplementedError(f"Communication mode '{mode}' is not implemented.")

        return None
    

    def change_data_blocks(self) -> None:
        newbody = []
        self.reduce_operations = {}
        for elem in self.body:
            if isinstance(elem, DataflowBlock):
                olddataflobblock = []
                newdataflobblocks = []
                for stmt in elem.statements:
                    if isinstance(stmt, MulStreamDeclaration) and isinstance(stmt.routing, ReduceRoutingDeclaration):
                        self.create_communication_patterns(elem.subgrid.x_range.start.value.value, 
                                                           elem.subgrid.x_range.stop.value.value, 
                                                           elem.subgrid.y_range.start.value.value, 
                                                           elem.subgrid.y_range.stop.value.value, 
                                                           stmt.dx.value.value, 
                                                           stmt.dy.value.value,
                                                           stmt.stream_name.name,
                                                           stmt.routing.graph,
                                                           stmt.routing.pipelined)

                        self.reduce_operations.update({stmt.stream_name.name: [{'op': stmt.routing.op}, [stmt.dx.value.value, stmt.dy.value.value], 
                                                                               [elem.subgrid.x_range.start.value.value, elem.subgrid.x_range.stop.value.value],
                                                                               [elem.subgrid.y_range.start.value.value, elem.subgrid.y_range.stop.value.value],
                                                                               [stmt.dx.value.value if (elem.subgrid.y_range.stop.value.value - elem.subgrid.y_range.start.value.value) % 2 == 0 else (elem.subgrid.x_range.stop.value.value - stmt.dx.value.value - 1),
                                                                                elem.subgrid.y_range.stop.value.value - 1 if elem.subgrid.y_range.start.value.value == stmt.dy.value.value else elem.subgrid.y_range.start.value.value]]})
                        new_grid_streams = []
                        new_snake_streams = []
                        
                        if stmt.stream_name.name in self.grid_streams:
                            current_grid_streams = self.grid_streams[stmt.stream_name.name]
                        elif stmt.stream_name.name in self.snake_streams:
                            current_grid_streams = self.snake_streams[stmt.stream_name.name]

                        for com in current_grid_streams:
                            newdataflobblocks.append([[com[0], com[1]], [com[2], com[3]],
                                RelativeStreamDeclaration(
                                    dtype=StreamType(stmt.dtype.dtype),
                                    stream_name=self.versioning.next_version("reduce"),
                                    dx=Expression(ConstantLiteral(com[4], ScalarType.i32)),
                                    dy=Expression(ConstantLiteral(com[5], ScalarType.i32))
                                ),
                                [com[6], com[7]]],
                            )
                            if stmt.stream_name.name in self.grid_streams:
                                if not stmt.routing.pipelined:
                                    if com[4] == -1:
                                        new_grid_streams.append([self.versioning.current_version("reduce"), com, StreamType(stmt.dtype.dtype), 'left', stmt.routing.pipelined])
                                    elif com[4] == 1:
                                        new_grid_streams.append([self.versioning.current_version("reduce"), com, StreamType(stmt.dtype.dtype), 'right', stmt.routing.pipelined])
                                    elif com[5] == -1:
                                        new_grid_streams.append([self.versioning.current_version("reduce"), com, StreamType(stmt.dtype.dtype), 'top', stmt.routing.pipelined])
                                    elif com[5] == 1:
                                        new_grid_streams.append([self.versioning.current_version("reduce"), com, StreamType(stmt.dtype.dtype), 'bottom', stmt.routing.pipelined])
                                else:
                                    if com[4] == -1:
                                        unrolled_com = []
                                        for i in range(com[1], com[0], -1):
                                            if i % 2 == com[1] % 2:
                                                for j in range(com[2], com[3]):
                                                    unrolled_com.append([i-1, i, j, j + 1, com[4], com[5], com[6], com[7], 'sender'])
                                            else:
                                                for j in range(com[2], com[3]):
                                                    unrolled_com.append([i-1, i, j, j + 1, com[4], com[5], com[6], com[7], 'receiver'])
                                        new_grid_streams.append([self.versioning.current_version("reduce"), unrolled_com, StreamType(stmt.dtype.dtype), 'left', stmt.routing.pipelined])
                                    elif com[4] == 1:
                                        unrolled_com = []
                                        for i in range(com[0], com[1]):
                                            if i % 2 == com[0] % 2:
                                                for j in range(com[2], com[3]):
                                                    unrolled_com.append([i, i+1, j, j + 1, com[4], com[5], com[6], com[7], 'sender'])
                                            else:
                                                for j in range(com[2], com[3]):
                                                    unrolled_com.append([i, i+1, j, j + 1, com[4], com[5], com[6], com[7], 'receiver'])
                                        new_grid_streams.append([self.versioning.current_version("reduce"), unrolled_com, StreamType(stmt.dtype.dtype), 'right', stmt.routing.pipelined])
                                    elif com[5] == -1:
                                        unrolled_com = []
                                        for i in range(com[3], com[2], -1):
                                            if i % 2 == com[3] % 2:
                                                unrolled_com.append([com[0], com[1], i-1, i, com[4], com[5], com[6], com[7], 'sender'])
                                            else:
                                                unrolled_com.append([com[0], com[1], i-1, i, com[4], com[5], com[6], com[7], 'receiver'])
                                        new_grid_streams.append([self.versioning.current_version("reduce"), unrolled_com, StreamType(stmt.dtype.dtype), 'top', stmt.routing.pipelined])
                                    elif com[5] == 1:
                                        unrolled_com = []
                                        for i in range(com[2], com[3]):
                                            if i % 2 == com[2] % 2:
                                                unrolled_com.append([com[0], com[1], i, i+1, com[4], com[5], com[6], com[7], 'sender'])
                                            else:
                                                unrolled_com.append([com[0], com[1], i, i+1, com[4], com[5], com[6], com[7], 'receiver'])
                                        new_grid_streams.append([self.versioning.current_version("reduce"), unrolled_com, StreamType(stmt.dtype.dtype), 'bottom', stmt.routing.pipelined])
                            elif stmt.stream_name.name in self.snake_streams:
                                if com[4] == -1:
                                    unrolled_com = []
                                    for i in range(com[2], com[3]):
                                        if (i - com[2]) % com[7] == 0:
                                            if not stmt.routing.pipelined:
                                                unrolled_com.append([com[0], com[1], i, i+1, com[4], com[5], com[6], com[7]])
                                            else:
                                                for j in range(com[1], com[0], -1):
                                                    if j % 2 == com[1] % 2:
                                                        unrolled_com.append([j-1, j, i, i+1, com[4], com[5], com[6], com[7], 'sender'])
                                                    else:
                                                        unrolled_com.append([j-1, j, i, i+1, com[4], com[5], com[6], com[7], 'receiver'])
                                    new_snake_streams.append([self.versioning.current_version("reduce"), com, StreamType(stmt.dtype.dtype), 'left', 'horizontal', unrolled_com, stmt.routing.pipelined])
                                elif com[4] == 1:
                                    unrolled_com = []
                                    for i in range(com[2], com[3]):
                                        if (i - com[2]) % com[7] == 0:
                                            if not stmt.routing.pipelined:
                                                unrolled_com.append([com[0], com[1], i, i+1, com[4], com[5], com[6], com[7]])
                                            else:
                                                for j in range(com[0], com[1]):
                                                    if j % 2 == com[0] % 2:
                                                        unrolled_com.append([j, j+1, i, i+1, com[4], com[5], com[6], com[7], 'sender'])
                                                    else:
                                                        unrolled_com.append([j, j+1, i, i+1, com[4], com[5], com[6], com[7], 'receiver'])
                                    new_snake_streams.append([self.versioning.current_version("reduce"), com, StreamType(stmt.dtype.dtype), 'right', 'horizontal', unrolled_com, stmt.routing.pipelined])
                                elif com[5] == -1:
                                    unrolled_com = []
                                    for i in range(com[3], com[2], -1):
                                        if i % 2 == com[3] % 2:
                                            unrolled_com.append([com[0], com[1], i-1, i, com[4], com[5], com[6], com[7], 'sender'])
                                        else:
                                            unrolled_com.append([com[0], com[1], i-1, i, com[4], com[5], com[6], com[7], 'receiver'])
                                    new_snake_streams.append([self.versioning.current_version("reduce"), com, StreamType(stmt.dtype.dtype), 'top', 'vertical', unrolled_com, stmt.routing.pipelined])
                                elif com[5] == 1:
                                    unrolled_com = []
                                    for i in range(com[2], com[3]):
                                        if i % 2 == com[2] % 2:
                                            unrolled_com.append([com[0], com[1], i, i+1, com[4], com[5], com[6], com[7], 'sender'])
                                        else:
                                            unrolled_com.append([com[0], com[1], i, i+1, com[4], com[5], com[6], com[7], 'receiver'])
                                    new_snake_streams.append([self.versioning.current_version("reduce"), com, StreamType(stmt.dtype.dtype), 'bottom', 'vertical', unrolled_com, stmt.routing.pipelined])
                        
                        if stmt.stream_name.name in self.grid_streams:
                            self.grid_streams.update({stmt.stream_name.name: new_grid_streams})
                        elif stmt.stream_name.name in self.snake_streams:
                            self.snake_streams.update({stmt.stream_name.name: new_snake_streams})

                    else:
                        olddataflobblock.append(stmt)

                if olddataflobblock != []: # not tested
                    newbody.append(DataflowBlock(variables=elem.variables, subgrid=elem.subgrid, statements=olddataflobblock))
                for newdataflobblock in newdataflobblocks:
                    newbody.append(
                        DataflowBlock(
                            variables=elem.variables,
                            subgrid=SubgridExpression(
                                x_range=RangeExpression(
                                    start=Expression(
                                        ConstantLiteral(newdataflobblock[0][0], ScalarType.i32)
                                    ),
                                    stop=Expression(
                                        ConstantLiteral(newdataflobblock[0][1], ScalarType.i32)
                                    ),
                                    step=Expression(
                                        ConstantLiteral(newdataflobblock[3][0], ScalarType.i32)
                                    )
                                ),
                                y_range=RangeExpression(
                                    start=Expression(
                                        ConstantLiteral(newdataflobblock[1][0], ScalarType.i32)
                                    ),
                                    stop=Expression(
                                        ConstantLiteral(newdataflobblock[1][1], ScalarType.i32)
                                    ),
                                    step=Expression(
                                        ConstantLiteral(newdataflobblock[3][1], ScalarType.i32)
                                    )
                                ),
                            ),
                            statements=[newdataflobblock[2]]))
            else:
                newbody.append(elem)
        self.body = newbody


    def fix_subgrid(self) -> None:
        newbody = []

        # change the outer loops to go through everything for each reduce and in that loop change the subgrids for the compute blocks

        for elem in self.body:
            if isinstance(elem, ComputeBlock):
                x_start = elem.subgrid.x_range.start.value.value
                x_stop = elem.subgrid.x_range.stop.value.value
                x_step = elem.subgrid.x_range.step.value.value if elem.subgrid.x_range.step is not None else None
                y_start = elem.subgrid.y_range.start.value.value
                y_stop = elem.subgrid.y_range.stop.value.value
                y_step = elem.subgrid.y_range.step.value.value if elem.subgrid.y_range.step is not None else None
                grid = [[[x_start, x_stop], [y_start, y_stop]]]

                for stmt in elem.statements: # walk operators in baseclass
                    if isinstance(stmt, ReduceStatement):
                        stream_name = stmt.stream_name.name

                        # test if stream_name is in grid_streams
                        if stmt.stream_name.name in self.grid_streams:
                            connections = self.grid_streams[stream_name]
                            
                            if not connections[0][4]:
                                #not pipelined
                                reduce_connections = []
                                send_connections = []
                                for con in connections:
                                    if con[3] == 'left':
                                        send_connections.append([con[1][1] - 1, con[1][1], con[1][2], con[1][3]])
                                    elif con[3] == 'right':
                                        send_connections.append([con[1][0], con[1][0] + 1, con[1][2], con[1][3]])
                                    elif con[3] == 'top':
                                        send_connections.append([con[1][0], con[1][1], con[1][3] - 1, con[1][3]])
                                    elif con[3] == 'bottom':
                                        send_connections.append([con[1][0], con[1][1], con[1][2], con[1][2] + 1])
                                    reduce_connections.append(con[1])
                                root = self.reduce_operations[stmt.stream_name.name][1]
                                for send in send_connections:
                                    reduce_connections.append(send)

                                reduce_connections.append([root[0], root[0] + 1, root[1], root[1] + 1])

                                # needs to be tested properly
                                for com_grid in reduce_connections:
                                    to_remove = []
                                    for sub_grid in grid:
                                        if com_grid[0] > sub_grid[0][0] and com_grid[0] < sub_grid[0][1]:
                                            # print("left")
                                            sub_x_start = sub_grid[0][0]
                                            sub_x_stop = sub_grid[0][1]
                                            sub_y_start = sub_grid[1][0]
                                            sub_y_stop = sub_grid[1][1]
                                            grid.append([[sub_x_start, com_grid[0]], [sub_y_start, sub_y_stop]])
                                            grid.append([[com_grid[0], sub_x_stop], [sub_y_start, sub_y_stop]])
                                            to_remove.append(sub_grid)
                                        elif com_grid[1] > sub_grid[0][0] and com_grid[1] < sub_grid[0][1]:
                                            # print("right")
                                            sub_x_start = sub_grid[0][0]
                                            sub_x_stop = sub_grid[0][1]
                                            sub_y_start = sub_grid[1][0]
                                            sub_y_stop = sub_grid[1][1]
                                            grid.append([[sub_x_start, com_grid[1]], [sub_y_start, sub_y_stop]])
                                            grid.append([[com_grid[1], sub_x_stop], [sub_y_start, sub_y_stop]])
                                            to_remove.append(sub_grid)
                                        elif com_grid[2] > sub_grid[1][0] and com_grid[2] < sub_grid[1][1] and com_grid[0] <= sub_grid[0][0] and com_grid[1] >= sub_grid[0][1]:
                                            # print("top")
                                            sub_x_start = sub_grid[0][0]
                                            sub_x_stop = sub_grid[0][1]
                                            sub_y_start = sub_grid[1][0]
                                            sub_y_stop = sub_grid[1][1]
                                            grid.append([[sub_x_start, sub_x_stop], [sub_y_start, com_grid[2]]])
                                            grid.append([[sub_x_start, sub_x_stop], [com_grid[2], sub_y_stop]])
                                            to_remove.append(sub_grid)
                                        elif com_grid[3] > sub_grid[1][0] and com_grid[3] < sub_grid[1][1] and com_grid[0] <= sub_grid[0][0] and com_grid[1] >= sub_grid[0][1]:
                                            # print("bottom")
                                            sub_x_start = sub_grid[0][0]
                                            sub_x_stop = sub_grid[0][1]
                                            sub_y_start = sub_grid[1][0]
                                            sub_y_stop = sub_grid[1][1]
                                            grid.append([[sub_x_start, sub_x_stop], [sub_y_start, com_grid[3]]])
                                            grid.append([[sub_x_start, sub_x_stop], [com_grid[3], sub_y_stop]])
                                            to_remove.append(sub_grid)
                                    # delete old unused
                                    for rmv in to_remove:
                                        grid.remove(rmv)

                            else:
                                #pipelined
                                new_grid = []
                                for i in range(grid[0][0][0], grid[0][0][1]):
                                    for j in range(grid[0][1][0], grid[0][1][1]):
                                        new_grid.append([[i, i + 1], [j, j + 1]])
                                grid = new_grid



                # needs to be tested in combination with grid_streams
                if self.snake_streams != {}:
                    new_grid = []
                    complete_grid = []
                    pipelined = False

                    for name in self.snake_streams:
                        complete_grid = [self.reduce_operations[name][2], self.reduce_operations[name][3]]
                        pipelined = self.snake_streams[name][0][6]
                        break

                    if not pipelined:

                        list_grid = [[x] for x in grid]

                        for com_grid in list_grid:
                            to_remove = []
                            for com in com_grid:
                                if com[0][0] == complete_grid[0][0] and com[0][1] != complete_grid[0][0] + 1:
                                    # print("left")
                                    com_grid.append([[complete_grid[0][0], complete_grid[0][0] + 1], [com[1][0], com[1][1]]])
                                    com_grid.append([[complete_grid[0][0] + 1, com[0][1]], [com[1][0], com[1][1]]])
                                    to_remove.append(com)
                                elif com[0][1] == complete_grid[0][1] and com[0][0] != complete_grid[0][1] - 1:
                                    # print("right")
                                    com_grid.append([[complete_grid[0][1] - 1, complete_grid[0][1]], [com[1][0], com[1][1]]])
                                    com_grid.append([[com[0][0], complete_grid[0][1] - 1], [com[1][0], com[1][1]]])
                                    to_remove.append(com)
                                elif com[1][1] - com[1][0] != 1:
                                    # print('multiple rows')
                                    for i in range(com[1][0], com[1][1]):
                                        com_grid.append([[com[0][0], com[0][1]], [i, i + 1]])
                                    to_remove.append(com)

                            for rmv in to_remove:
                                com_grid.remove(rmv)

                            for com in com_grid:
                                new_grid.append(com)
                    
                    else:
                        for i in range(grid[0][0][0], grid[0][0][1]):
                            for j in range(grid[0][1][0], grid[0][1][1]):
                                new_grid.append([[i, i + 1], [j, j + 1]])

                    grid = new_grid

                
                for com_grid in grid:
                    newbody.append(
                        ComputeBlock(
                            elem.variables,
                            SubgridExpression(
                                RangeExpression(
                                    start=Expression(ConstantLiteral(com_grid[0][0], ScalarType.i32)),
                                    stop=Expression(ConstantLiteral(com_grid[0][1], ScalarType.i32))
                                ),
                                RangeExpression(
                                    start=Expression(ConstantLiteral(com_grid[1][0], ScalarType.i32)),
                                    stop=Expression(ConstantLiteral(com_grid[1][1], ScalarType.i32))
                                )
                            ),
                            elem.statements
                        )
                    )
            else:
                newbody.append(elem)

        self.body = newbody
        return None


    def change_compute_blocks(self) -> None:
        finalbody = []
        for elem in self.body:
            #print(elem)
            #print('-'*50)
            #for tst in elem.iter_child_nodes(): ### use this to go over nested nodes
            #        print(tst)
            #        print('@'*50)
            #exit()

            if isinstance(elem, ComputeBlock):
                statements = []
                for stmt in elem.statements: # walk operators in baseclass
                    if isinstance(stmt, ReduceStatement):

                        current_position = [elem.subgrid.x_range.start.value.value, 
                                            elem.subgrid.x_range.stop.value.value, 
                                            elem.subgrid.y_range.start.value.value, 
                                            elem.subgrid.y_range.stop.value.value]
                        newstatements = []
                        stream_name = stmt.stream_name.name
                        operation_id = self.reduce_operations[stmt.stream_name.name][0]['op']
                        root = self.reduce_operations[stmt.stream_name.name][1]
                        origin = self.reduce_operations[stmt.stream_name.name][4]
                        complete_grid = [self.reduce_operations[stmt.stream_name.name][2], self.reduce_operations[stmt.stream_name.name][3]]

                        if stream_name in self.grid_streams:
                            connections = self.grid_streams[stream_name]
                        elif stream_name in self.snake_streams:
                            connections = self.snake_streams[stream_name]
                        else:
                            raise ValueError(f"Stream name {stream_name} not found in grid_streams or snake_streams.")
                        
                        if operation_id == "S_SUM":
                            current_op = '+'
                        elif operation_id == "S_PROD":
                            current_op = '*'
                        else:
                            raise NotImplementedError("Currently only S_SUM and S_PROD are supported.")

                        if stream_name in self.grid_streams:
                            pipelined_send = []
                            pipelined_receive = []
                            if not connections[0][4]:
                                # not pipelined
                                for con in connections:
                                    if (current_position[0] >= con[1][0]
                                        and current_position[1] <= con[1][1]
                                        and current_position[2] >= con[1][2]
                                        and current_position[3] <= con[1][3]):

                                        if (con[3] == 'left' and current_position[1] != con[1][1]
                                            or con[3] == 'right' and current_position[0] != con[1][0]
                                            or con[3] == 'top' and current_position[3] != con[1][3]
                                            or con[3] == 'bottom' and current_position[2] != con[1][2]):

                                            newstatements.append(
                                                ForeachStatement(
                                                    variables=[TypedIdentifier(dtype=ScalarType.i32, identifier=self.versioning.next_version("reduce_runner"))],
                                                    parameter_range=[RangeExpression(start=Expression(ConstantLiteral(0, ScalarType.i32)),
                                                                                    stop=Expression(ConstantLiteral(1, ScalarType.i32)),
                                                                                    step=None)],
                                                    stream_variable=TypedIdentifier(dtype=con[2].dtype,
                                                                                    identifier=self.versioning.next_version("reduce_receive")),
                                                    receive_stream=ReceiveGenerator(stream_name=con[0]),
                                                    body=[
                                                        AssignmentStatement(
                                                            destination=ArraySlice(
                                                                array=stmt.local_array,
                                                                indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                            ),
                                                            source=Expression(
                                                                BinaryOperator(
                                                                    left=Expression(
                                                                        value=ArraySlice(
                                                                            array=stmt.local_array,
                                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                        )
                                                                    ),
                                                                    op= current_op,
                                                                    right=Expression(
                                                                        value=self.versioning.current_version("reduce_receive")
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    ],
                                                    completion_name=None
                                                )
                                            )

                                        if (con[3] == 'left' and current_position[0] != con[1][0]
                                            or con[3] == 'right' and current_position[1] != con[1][1]
                                            or con[3] == 'top' and current_position[2] != con[1][2]
                                            or con[3] == 'bottom' and current_position[3] != con[1][3]):
                    
                                            newstatements.append(
                                                SendStatement(
                                                    local_array=stmt.local_array,
                                                    stream_name=con[0],
                                                    completion_name=None
                                                )
                                            )


                            else:
                                print(current_position)
                                print(root)
                                for con_list in connections:
                                    #print(con_list)
                                    for con in con_list[1]:
                                        if (current_position[0] >= con[0] and current_position[1] <= con[1]
                                            and current_position[2] >= con[2] and current_position[3] <= con[3]):
                                            print(con)
                                            if con[8] == 'sender':
                                                pipelined_send.append(con_list[0])
                                            elif con[8] == 'receiver':
                                                pipelined_receive.append(con_list[0])

                                if pipelined_send != [] and pipelined_receive != []:
                                    newstatements.append(
                                        AssignmentStatement(
                                            destination=self.versioning.next_version("pipeline_helper"),
                                            source=Expression(
                                                ConstantLiteral(0, ScalarType.i32)
                                            )
                                        )
                                    )
                                    if len(pipelined_receive) == 1:
                                        newstatements.append(
                                            ForStatement(
                                                variables=[TypedIdentifier(dtype=ScalarType.i32, identifier=self.versioning.next_version("reduce_runner"))],
                                                range_expression=[RangeExpression(start=Expression(ConstantLiteral(0, ScalarType.i32)),
                                                                                stop=Expression(ConstantLiteral(1, ScalarType.i32)),
                                                                                step=None)],
                                                body=[
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[0],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    SendStatement(
                                                        local_array=ArraySlice(
                                                                array=stmt.local_array,
                                                                indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                            ),
                                                        stream_name=pipelined_send[0],
                                                        completion_name=None
                                                    )
                                                ],
                                            )
                                        )
                                    elif len(pipelined_receive) == 2:
                                        newstatements.append(
                                            ForStatement(
                                                variables=[TypedIdentifier(dtype=ScalarType.i32, identifier=self.versioning.next_version("reduce_runner"))],
                                                range_expression=[RangeExpression(start=Expression(ConstantLiteral(0, ScalarType.i32)),
                                                                                stop=Expression(ConstantLiteral(1, ScalarType.i32)),
                                                                                step=None)],
                                                body=[
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[0],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[1],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    SendStatement(
                                                        local_array=ArraySlice(
                                                                array=stmt.local_array,
                                                                indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                            ),
                                                        stream_name=pipelined_send[0],
                                                        completion_name=None
                                                    )
                                                ],
                                            )
                                        )
                                    elif len(pipelined_receive) == 3:
                                        newstatements.append(
                                            ForStatement(
                                                variables=[TypedIdentifier(dtype=ScalarType.i32, identifier=self.versioning.next_version("reduce_runner"))],
                                                range_expression=[RangeExpression(start=Expression(ConstantLiteral(0, ScalarType.i32)),
                                                                                stop=Expression(ConstantLiteral(1, ScalarType.i32)),
                                                                                step=None)],
                                                body=[
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[0],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[1],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[2],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    SendStatement(
                                                        local_array=ArraySlice(
                                                                array=stmt.local_array,
                                                                indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                            ),
                                                        stream_name=pipelined_send[0],
                                                        completion_name=None
                                                    )
                                                ],
                                            )
                                        )
                                    else:
                                        newstatements.append(
                                            ForStatement(
                                                variables=[TypedIdentifier(dtype=ScalarType.i32, identifier=self.versioning.next_version("reduce_runner"))],
                                                range_expression=[RangeExpression(start=Expression(ConstantLiteral(0, ScalarType.i32)),
                                                                                stop=Expression(ConstantLiteral(1, ScalarType.i32)),
                                                                                step=None)],
                                                body=[
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[0],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[1],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[2],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[3],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    SendStatement(
                                                        local_array=ArraySlice(
                                                                array=stmt.local_array,
                                                                indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                            ),
                                                        stream_name=pipelined_send[0],
                                                        completion_name=None
                                                    )
                                                ],
                                            )
                                        )
                                elif pipelined_send == [] and pipelined_receive != []:
                                    newstatements.append(
                                        AssignmentStatement(
                                            destination=self.versioning.next_version("pipeline_helper"),
                                            source=Expression(
                                                ConstantLiteral(0, ScalarType.i32)
                                            )
                                        )
                                    )
                                    if len(pipelined_receive) == 1:
                                        newstatements.append(
                                            ForStatement(
                                                variables=[TypedIdentifier(dtype=ScalarType.i32, identifier=self.versioning.next_version("reduce_runner"))],
                                                range_expression=[RangeExpression(start=Expression(ConstantLiteral(0, ScalarType.i32)),
                                                                                stop=Expression(ConstantLiteral(1, ScalarType.i32)),
                                                                                step=None)],
                                                body=[
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[0],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    )
                                                ],
                                            )
                                        )
                                    elif len(pipelined_receive) == 2:
                                        newstatements.append(
                                            ForStatement(
                                                variables=[TypedIdentifier(dtype=ScalarType.i32, identifier=self.versioning.next_version("reduce_runner"))],
                                                range_expression=[RangeExpression(start=Expression(ConstantLiteral(0, ScalarType.i32)),
                                                                                stop=Expression(ConstantLiteral(1, ScalarType.i32)),
                                                                                step=None)],
                                                body=[
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[0],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[1],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    )
                                                ],
                                            )
                                        )
                                    elif len(pipelined_receive) == 3:
                                        newstatements.append(
                                            ForStatement(
                                                variables=[TypedIdentifier(dtype=ScalarType.i32, identifier=self.versioning.next_version("reduce_runner"))],
                                                range_expression=[RangeExpression(start=Expression(ConstantLiteral(0, ScalarType.i32)),
                                                                                stop=Expression(ConstantLiteral(1, ScalarType.i32)),
                                                                                step=None)],
                                                body=[
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[0],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[1],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[2],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    )
                                                ],
                                            )
                                        )
                                    else:
                                        newstatements.append(
                                            ForStatement(
                                                variables=[TypedIdentifier(dtype=ScalarType.i32, identifier=self.versioning.next_version("reduce_runner"))],
                                                range_expression=[RangeExpression(start=Expression(ConstantLiteral(0, ScalarType.i32)),
                                                                                stop=Expression(ConstantLiteral(1, ScalarType.i32)),
                                                                                step=None)],
                                                body=[
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[0],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[1],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[2],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    ReceiveStatement(
                                                        local_array=self.versioning.current_version("pipeline_helper"),
                                                        stream_name=pipelined_receive[3],
                                                        completion_name=None
                                                    ),
                                                    AssignmentStatement(
                                                        destination=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                        source=Expression(
                                                            BinaryOperator(
                                                                left=Expression(
                                                                    value=ArraySlice(
                                                                        array=stmt.local_array,
                                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                    )
                                                                ),
                                                                op= current_op,
                                                                right=Expression(
                                                                    value=self.versioning.current_version("pipeline_helper")
                                                                )
                                                            )
                                                        )
                                                    )
                                                ],
                                            )
                                        )
                                elif pipelined_send != [] and pipelined_receive == []:
                                    newstatements.append(
                                        ForStatement(
                                            variables=[TypedIdentifier(dtype=ScalarType.i32, identifier=self.versioning.next_version("reduce_runner"))],
                                            range_expression=[RangeExpression(start=Expression(ConstantLiteral(0, ScalarType.i32)),
                                                                            stop=Expression(ConstantLiteral(1, ScalarType.i32)),
                                                                            step=None)],
                                            body=[
                                                SendStatement(
                                                    local_array=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                    stream_name=pipelined_send[0],
                                                    completion_name=None
                                                )
                                            ],
                                        )
                                    )
                                else:
                                    raise ValueError(f"No pipelined send or receive found for position {current_position}.")
                                    
                                

                        elif stream_name in self.snake_streams:
                            if not (current_position[0] == origin[0] and current_position[2] == origin[1]):
                                # everything but the starting point receives first

                                # get receive stream
                                receive_stream = None
                                for con in connections:
                                    for detailed_con in con[5]:
                                        if (current_position[0] >= detailed_con[0] and current_position[1] <= detailed_con[1]
                                            and current_position[2] >= detailed_con[2] and current_position[3] <= detailed_con[3]
                                            and ((detailed_con[4] == -1 and not current_position[1] == detailed_con[1])
                                                 or (detailed_con[4] == 1 and not current_position[0] == detailed_con[0])
                                                 or (detailed_con[4] == 0 and detailed_con[8] == 'receiver')
                                                 or (con[6] == True and detailed_con[8] == 'receiver'))):
                                            receive_stream = con
                                            break
                                        
                                    if not receive_stream == None:
                                        break

                                if operation_id == "S_SUM":
                                    current_op = '+'
                                elif operation_id == "S_PROD":
                                    current_op = '*'
                                else:
                                    raise NotImplementedError("Currently only S_SUM and S_PROD are supported.")

                                # change receive statement

                                # not pipelined
                                if not con[6]:
                                    newstatements.append(
                                        ForeachStatement(
                                            variables=[TypedIdentifier(dtype=ScalarType.i32, identifier=self.versioning.next_version("reduce_runner"))],
                                            parameter_range=[RangeExpression(start=Expression(ConstantLiteral(0, ScalarType.i32)),
                                                                            stop=Expression(ConstantLiteral(1, ScalarType.i32)),
                                                                            step=None)],
                                            stream_variable=TypedIdentifier(dtype=receive_stream[2].dtype,
                                                                            identifier=self.versioning.next_version("reduce_receive")),
                                            receive_stream=ReceiveGenerator(stream_name=receive_stream[0]),
                                            body=[
                                                AssignmentStatement(
                                                    destination=ArraySlice(
                                                        array=stmt.local_array,
                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                    ),
                                                    source=Expression(
                                                        BinaryOperator(
                                                            left=Expression(
                                                                value=ArraySlice(
                                                                    array=stmt.local_array,
                                                                    indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                )
                                                            ),
                                                            op= current_op,
                                                            right=Expression(
                                                                value=self.versioning.current_version("reduce_receive")
                                                            )
                                                        )
                                                    )
                                                )
                                            ],
                                            completion_name=None
                                        )
                                    )

                                # pipelined root
                                elif (current_position[0] == root[0] and current_position[2] == root[1]):
                                    newstatements.append(
                                        AssignmentStatement(
                                            destination=self.versioning.next_version("pipeline_helper"),
                                            source=Expression(
                                                ConstantLiteral(0, ScalarType.i32)
                                            )
                                        )
                                    )
                                    newstatements.append(
                                        ForStatement(
                                            variables=[TypedIdentifier(dtype=ScalarType.i32, identifier=self.versioning.next_version("reduce_runner"))],
                                            range_expression=[RangeExpression(start=Expression(ConstantLiteral(0, ScalarType.i32)),
                                                                            stop=Expression(ConstantLiteral(1, ScalarType.i32)),
                                                                            step=None)],
                                            body=[
                                                ReceiveStatement(
                                                    local_array=self.versioning.current_version("pipeline_helper"),
                                                    stream_name=receive_stream[0],
                                                    completion_name=None
                                                ),
                                                AssignmentStatement(
                                                    destination=ArraySlice(
                                                        array=stmt.local_array,
                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                    ),
                                                    source=Expression(
                                                        BinaryOperator(
                                                            left=Expression(
                                                                value=ArraySlice(
                                                                    array=stmt.local_array,
                                                                    indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                )
                                                            ),
                                                            op= current_op,
                                                            right=Expression(
                                                                value=self.versioning.current_version("pipeline_helper")
                                                            )
                                                        )
                                                    )
                                                )
                                            ],
                                        )
                                    )

                            if not (current_position[0] == root[0] and current_position[2] == root[1]):
                                # only root does not send

                                # get send stream
                                send_stream = None
                                for con in connections:
                                    for detailed_con in con[5]:
                                        if (current_position[0] >= detailed_con[0] and current_position[1] <= detailed_con[1]
                                            and current_position[2] >= detailed_con[2] and current_position[3] <= detailed_con[3]
                                            and ((detailed_con[4] == 1 and not current_position[1] == detailed_con[1])
                                                 or (detailed_con[4] == -1 and not current_position[0] == detailed_con[0])
                                                 or (detailed_con[4] == 0 and detailed_con[8] == 'sender')
                                                 or (con[6] == True and detailed_con[8] == 'sender'))):
                                            send_stream = con
                                            break
                                        
                                    if not send_stream == None:
                                        break

                                # not pipelined
                                if not con[6]:
                                    newstatements.append(
                                        SendStatement(
                                            local_array=stmt.local_array,
                                            stream_name=send_stream[0],
                                            completion_name=None
                                        )
                                    )

                                # pipelined origin
                                elif (current_position[0] == origin[0] and current_position[2] == origin[1]):
                                    newstatements.append(
                                        ForStatement(
                                            variables=[TypedIdentifier(dtype=ScalarType.i32, identifier=self.versioning.next_version("reduce_runner"))],
                                            range_expression=[RangeExpression(start=Expression(ConstantLiteral(0, ScalarType.i32)),
                                                                            stop=Expression(ConstantLiteral(1, ScalarType.i32)),
                                                                            step=None)],
                                            body=[
                                                SendStatement(
                                                    local_array=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                    stream_name=send_stream[0],
                                                    completion_name=None
                                                )
                                            ],
                                        )
                                    )


                                # pipelined
                                else:
                                    newstatements.append(
                                        AssignmentStatement(
                                            destination=self.versioning.next_version("pipeline_helper"),
                                            source=Expression(
                                                ConstantLiteral(0, ScalarType.i32)
                                            )
                                        )
                                    )
                                    newstatements.append(
                                        ForStatement(
                                            variables=[TypedIdentifier(dtype=ScalarType.i32, identifier=self.versioning.next_version("reduce_runner"))],
                                            range_expression=[RangeExpression(start=Expression(ConstantLiteral(0, ScalarType.i32)),
                                                                            stop=Expression(ConstantLiteral(1, ScalarType.i32)),
                                                                            step=None)],
                                            body=[
                                                ReceiveStatement(
                                                    local_array=self.versioning.current_version("pipeline_helper"),
                                                    stream_name=receive_stream[0],
                                                    completion_name=None
                                                ),
                                                AssignmentStatement(
                                                    destination=ArraySlice(
                                                        array=stmt.local_array,
                                                        indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                    ),
                                                    source=Expression(
                                                        BinaryOperator(
                                                            left=Expression(
                                                                value=ArraySlice(
                                                                    array=stmt.local_array,
                                                                    indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                                )
                                                            ),
                                                            op= current_op,
                                                            right=Expression(
                                                                value=self.versioning.current_version("pipeline_helper")
                                                            )
                                                        )
                                                    )
                                                ),
                                                SendStatement(
                                                    local_array=ArraySlice(
                                                            array=stmt.local_array,
                                                            indices=[Expression(value=self.versioning.current_version("reduce_runner"))]
                                                        ),
                                                    stream_name=send_stream[0],
                                                    completion_name=None
                                                )
                                            ],
                                        )
                                    )

                        # add receive + calculation + send here
                        for new_statement in newstatements:
                            statements.append(new_statement)

                    else:
                        statements.append(stmt)
                
                finalbody.append(ComputeBlock(elem.variables, elem.subgrid, statements))        
            else:
                finalbody.append(elem)   
        
        self.body = finalbody
        #exit()

        return None