from spatialstencil.syntax.spatial_ir.irnodes import (Kernel, DataflowBlock, MulStreamDeclaration, BroadcastRoutingDeclaration, ComputeBlock, ConstantLiteral,
                                                      SubgridExpression, RangeExpression, Expression, ScalarType, ForeachStatement, ForStatement, MapStatement,
                                                      AsyncBlock, BroadcastStatement, SendStatement, ReceiveStatement, Identifier)
from spatialstencil.lowering.versioning import Versioning

class BroadcastOptimizer():
    roots: list[list[int]] = []
    broadcast_operations: dict[str, list[int]] = {}

    def __init__(self, kernel: Kernel) -> None:
        self.name = kernel.name
        self.parameters = kernel.parameters
        self.arguments = kernel.arguments
        self.body = kernel.body
        self.versioning = Versioning[Identifier](Identifier)
        self.roots = []
        self.broadcast_operations = {}
        return None

    ##
    # Replace the broadcast statements in the compute blocks and change the tiling of the compute statemtents accordingly
    # Entry Function
    ##
    def broadcast_subroutine(self) -> Kernel:
        self.find_roots()
        if self.roots != []:
            self.fix_subgrid()
            self.replace_broadcast()
        return Kernel(name=self.name, parameters=self.parameters, arguments=self.arguments, body=self.body)
    

    ##
    # Function to aggregate the roots of all broadcast operations
    ##
    def find_roots(self) -> None:
        for elem in self.body:
            if isinstance(elem, DataflowBlock):
                for stmt in elem.statements:
                    if isinstance(stmt, MulStreamDeclaration) and isinstance(stmt.routing, BroadcastRoutingDeclaration):
                        self.roots.append([stmt.x.value.value, stmt.x.value.value + 1, stmt.y.value.value, stmt.y.value.value + 1])
                        self.broadcast_operations[stmt.stream_name.name] = [stmt.x.value.value, stmt.y.value.value]
        return None


    ##
    # Updates the compute block tiling for the new communication patterns
    ##
    def fix_subgrid(self) -> None:
        newbody = []
        for elem in self.body:
            if isinstance(elem, ComputeBlock):
                x_start = elem.subgrid.x_range.start.value.value
                x_stop = elem.subgrid.x_range.stop.value.value if isinstance(elem.subgrid.x_range.stop.value, ConstantLiteral) else None
                y_start = elem.subgrid.y_range.start.value.value
                y_stop = elem.subgrid.y_range.stop.value.value if isinstance(elem.subgrid.y_range.stop.value, ConstantLiteral) else None

                ## fix to deal with parameters
                x_literal = True if x_stop is None else False
                y_literal = True if y_stop is None else False
                if x_literal:
                    x_stop = 9999999999999
                if y_literal:
                    y_stop = 9999999999999
                grid = [[[x_start, x_stop], [y_start, y_stop]]]

                for com_grid in self.roots:
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

                for sub_grid in grid:
                    if x_literal or y_literal:
                        if sub_grid[0][1] == 9999999999999:
                            sub_grid[0][1] = elem.subgrid.x_range.stop.value
                        else:
                            sub_grid[0][1] = ConstantLiteral(value=sub_grid[0][1], dtype=elem.subgrid.x_range.start.value.dtype)
                        if sub_grid[1][1] == 9999999999999:
                            sub_grid[1][1] = elem.subgrid.y_range.stop.value
                        else:
                            sub_grid[1][1] = ConstantLiteral(value=sub_grid[1][1], dtype=elem.subgrid.y_range.start.value.dtype)

                if not x_literal and not y_literal:
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
                    for com_grid in grid:
                        newbody.append(
                            ComputeBlock(
                                elem.variables,
                                SubgridExpression(
                                    RangeExpression(
                                        start=Expression(ConstantLiteral(com_grid[0][0], ScalarType.i32)),
                                        stop=Expression(com_grid[0][1])
                                    ),
                                    RangeExpression(
                                        start=Expression(ConstantLiteral(com_grid[1][0], ScalarType.i32)),
                                        stop=Expression(com_grid[1][1])
                                    )
                                ),
                                elem.statements
                            )
                        )
                
            else:
                newbody.append(elem)

        self.body = newbody
        return None
    


    ##
    # Replace the broadcast statements with send and receive statements
    ##
    def _replace_broadcast(self, stmt, elem) -> list[Expression]:
        x_start = elem.subgrid.x_range.start.value.value
        x_stop = elem.subgrid.x_range.stop.value.value if isinstance(elem.subgrid.x_range.stop.value, ConstantLiteral) else None
        y_start = elem.subgrid.y_range.start.value.value
        y_stop = elem.subgrid.y_range.stop.value.value if isinstance(elem.subgrid.y_range.stop.value, ConstantLiteral) else None
        
        ## fix to deal with parameters
        x_literal = True if x_stop is None else False
        y_literal = True if y_stop is None else False
        if x_literal:
            x_stop = 9999999999999
        if y_literal:
            y_stop = 9999999999999

        name = stmt.stream_name.name
        root = self.broadcast_operations[name]

        if x_start == root[0] and y_start == root[1] and x_stop == root[0] + 1 and y_stop == root[1] + 1:
            send = SendStatement(
                local_array=stmt.local_array,
                stream_name=stmt.stream_name,
                completion_name=None
            )
            return [send]
        else:
            receive = ReceiveStatement(
                local_array=stmt.local_array,
                stream_name=stmt.stream_name,
                completion_name=None
            )
            return [receive]


    ##
    # Function to recursively go through the compute blocks and find all the occurernces of the broadcast statements
    ##
    def replace_stmt(self, stmt, elem, to_replace) -> list[Expression]:
        input_stmt = stmt
        if isinstance(stmt, to_replace):
            if to_replace == BroadcastStatement:
                return self._replace_broadcast(stmt, elem)
        
        # all of these use body
        elif isinstance(stmt, ForeachStatement) or isinstance(stmt, ForStatement) or isinstance(stmt, MapStatement) or isinstance(stmt, AsyncBlock):
            new_body = []
            for body_stmt in stmt.body:
                replaced_stmts = self.replace_stmt(body_stmt, elem, to_replace)
                for replaced_stmt in replaced_stmts:
                    new_body.append(replaced_stmt)
            input_stmt.body = new_body
        return [input_stmt]


    ##
    # Changes the occurences of broadcast statements in the compute blocks
    ##
    def replace_broadcast(self) -> None:
        finalbody = []
        for elem in self.body:
            if isinstance(elem, ComputeBlock):
                statements = []
                for stmt in elem.statements:
                    new_stmts = self.replace_stmt(stmt, elem, BroadcastStatement)
                    for nstmt in new_stmts:
                        statements.append(nstmt)
                finalbody.append(ComputeBlock(elem.variables, elem.subgrid, statements))
            else:
                finalbody.append(elem) 
        
        self.body = finalbody
        return None