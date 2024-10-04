import copy
from dataclasses import dataclass
from typing import TypeVar, Generic

from spatialstencil.lowering.stencil_to_spatial_dataflow import ProgramDataflow
from spatialstencil.lowering.stencil_to_spatial_place import ProgramPlacement
from spatialstencil.lowering.versioning import Versioning
from spatialstencil.syntax.common.basenode import Wildcard
from spatialstencil.syntax.common.tree_matching import PatternMatcher, PatternTransformer
from spatialstencil.syntax.common.types import ScalarType
from spatialstencil.syntax.common.visitor import IRNodeVisitor
from spatialstencil.syntax.spatial_ir.grid_geometry import Rectangle, group_rectangles_by_domain, split_rectangles
from spatialstencil.syntax.stencil_ir.domain_collector import DomainCollector
import spatialstencil.syntax.spatial_ir.irnodes as spa
import spatialstencil.syntax.stencil_ir.irnodes as sast

AbstractStatement = Rectangle[tuple[int, spa.Statement]]


class ProgramCompute:

    def __init__(self,
                 domains: DomainCollector,
                 versioning: Versioning[spa.Identifier],
                 dataflow: ProgramDataflow,
                 placement: ProgramPlacement,
                 subgrid_var_type: ScalarType = ScalarType.u16):
        self.domains = domains
        self.versioning = versioning
        self.dataflow = dataflow
        self.placement = placement
        self.offset_domain = domains.get_shift()[0:2]
        self.visitor = ComputeVisitor(placement, versioning, dataflow)
        self.grid_var_t = subgrid_var_type

    def generate_computation(self, comp: sast.ComputationBlock) -> list[spa.ComputeBlock]:
        """
        Generate a computation block.

        :param comp:
        :return:
        """
        assert comp.schedule == sast.ComputationType.PARALLEL

        # Generate the compute block
        self.visitor.visit(comp)
        body = self.visitor.stmts

        # Merge all statements into a compute blocks

        split = split_rectangles(body)
        merged = group_rectangles_by_domain(split)

        # Convert to Compute blocks
        compute_blocks = []
        for block in merged:
            compute_blocks.append(self._convert_to_compute_block(block))

        return compute_blocks

    def _convert_to_compute_block(self, block: list[AbstractStatement]) -> spa.ComputeBlock:
        var_i = self.versioning.next_version('i')
        var_j = self.versioning.next_version('j')

        variables = [spa.TypedIdentifier(self.grid_var_t, var_i),
                     spa.TypedIdentifier(self.grid_var_t, var_j)]

        subgrid = spa.SubgridExpression.from_tuple(
            block[0].x_range, block[0].y_range
        )

        stmts = sorted(block, key=lambda x: x.metadata[0])

        block = spa.ComputeBlock(
            variables,
            subgrid,
            [stmt.metadata[1] for stmt in stmts]
        )

        return block


class ComputeVisitor(sast.ScopedNodeVisitor):

    def __init__(self, placement: ProgramPlacement,
                 versioning: Versioning[spa.Identifier],
                 dataflow: ProgramDataflow):
        super().__init__()
        self.placement = placement
        self.versioning = versioning
        self.dataflow = dataflow
        self.stmts = []

        self.statement_transformers = [UnaryMapTransformer(placement, versioning),
                                       MapTransformer(placement, versioning),
                                       HorizontalStencilTransformer(placement, versioning, dataflow)]

    def visit_ReturnOp(self, op: sast.ReturnOp):
        comp = self.get_scope()
        assert isinstance(comp, sast.ComputationBlock)

        shift = self.placement.get_shift()
        assert shift[2] == 0
        for value, value_t, out, out_t in zip(op.values, op.operation_type.source, comp.outputs,
                                              comp.operation_type.destination):
            value = value.value
            assert isinstance(value, sast.Identifier)

            x_range = _shift(out_t.domain.x, shift[0])
            y_range = _shift(out_t.domain.y, shift[1])

            var_k = self.versioning.next_version('k')

            dst_range = (value_t.domain.z[0], value_t.domain.z[1])
            src_range = (out_t.domain.z[0], out_t.domain.z[1])

            translation = src_range[0] - dst_range[0]

            src_id, src_dtype = self.placement.get_storage(value)

            src_e = spa.Expression(
                spa.ArraySlice(
                    src_id,
                    [spa.Expression(var_k)]
                )
            )

            dst_id, dst_dtype = self.placement.get_storage(out)

            if translation > 0:

                dst_e = spa.ArraySlice(
                    dst_id,
                    [spa.RangeExpression(spa.Expression(spa.BinaryOperator(spa.Expression(var_k),
                                                                           '+',
                                                                           spa.Expression(
                                                                               spa.ConstantLiteral(translation,
                                                                                                   ScalarType.i32)))))]
                )
            else:
                dst_e = spa.ArraySlice(
                    dst_id,
                    [spa.Expression(var_k)]
                )

            stmt = spa.MapStatement(
                variables=[spa.TypedIdentifier(ScalarType.i32, var_k)],
                range_expression=[spa.RangeExpression.from_args(0, out_t.domain.z[1])],
                body=[
                    spa.AssignmentStatement(
                        dst_e,
                        src_e,
                    )
                ]
            )

            line_nr = self.versioning.next_version("___line___").version
            self.stmts.append(AbstractStatement(x_range, y_range, (line_nr, stmt)))

    def visit_StatementBlock(self, op: sast.StatementBlock):
        comp = self.get_scope()
        assert isinstance(comp, sast.ComputationBlock)

        for transformer in self.statement_transformers:
            transformer.set_context((comp, op))

        for stmt in op.body:
            statements = self._apply_statement_transformers(stmt)
            assert len(statements) > 0, f"Could not match statement {stmt.as_ir()}"
            self.stmts.extend(statements)

    def _apply_statement_transformers(self, op: sast.AssignOp) -> list[AbstractStatement]:
        blocks = []
        for transformer in self.statement_transformers:
            res = transformer.first(op)
            if len(res):
                blocks.extend(res)
                break
        return blocks

    def visit_MaterializeOp(self, op: sast.MaterializeOp):

        # The materialize operation creates data movement for each offset in its output offsets
        # that is not zero
        dst = op.result
        src = op.value
        result = []
        for extent in op.operation_type.destination[0].extent.extents:
            if extent != sast.Offset.zero():
                dst_buf, dst_dtype = self.placement.get_storage(dst, extent)

                # Approach: Communicate the remote values and aggregate them into the local value
                # For this, we need:

                # (2) local buffer
                src_buf, src_dtype = self.placement.get_storage(src)

                # (3) remote buffer
                # Determine if its an input type or an intermediate type
                out_t = op.operation_type.destination[0]
                shift = self.placement.get_shift()
                xy_range = _get_range(out_t, shift)

                # (4) stream used to communicate the remote buffer
                stream = self.dataflow.get_stream(src, dst, extent)
                assert stream

                # Loop variables
                var_k = self.versioning.next_version('k')
                var_x = self.versioning.next_version('x')

                recv = spa.ReceiveGenerator(
                    stream
                )

                src_expr = spa.Expression(var_x)

                assign_stmt = spa.AssignmentStatement(
                    source=src_expr,
                    destination=spa.ArraySlice(
                        dst_buf,
                        [spa.Expression(var_k)]
                    )
                )

                body = [assign_stmt]

                recv_comp_id = self.versioning.next_version('_recv_comp')
                recv_completion = spa.Completion(recv_comp_id)
                recv_foreach = spa.ForeachStatement(
                    variables=[spa.TypedIdentifier(ScalarType.i32, var_k)],
                    parameter_range=[spa.RangeExpression.from_args(0, dst_dtype.shape[0])],
                    stream_variable=spa.TypedIdentifier(src_dtype.base_type, var_x),
                    receive_stream=recv,
                    body=body,
                    completion_name=recv_completion,
                )

                line_nr = self.versioning.next_version("___line___").version
                receive = AbstractStatement(xy_range[0], xy_range[1], (line_nr, recv_foreach))

                send_comp_id = self.versioning.next_version('_send_comp')
                send_completion = spa.Completion(send_comp_id)
                send = spa.SendStatement(
                    src_buf,
                    stream,
                    send_completion
                )

                send_domain = out_t.domain.union(out_t.domain.add(extent.values))

                send_x_range = _shift(send_domain.x, shift[0])
                send_y_range = _shift(send_domain.y, shift[1])

                line_nr = self.versioning.next_version("___line___").version
                send_stmt = AbstractStatement(send_x_range, send_y_range, (line_nr, send))

                line_nr = self.versioning.next_version("___line___").version
                await_send = AbstractStatement(send_x_range, send_y_range,
                                               (line_nr, spa.AwaitCompletionStatement(send_comp_id)))

                line_nr = self.versioning.next_version("___line___").version
                await_recv = AbstractStatement(xy_range[0], xy_range[1],
                                               (line_nr, spa.AwaitCompletionStatement(recv_comp_id)))

                self.stmts.extend([receive, send_stmt, await_send, await_recv])


class MapTransformer(PatternTransformer[sast.AssignOp, AbstractStatement, tuple[sast.ComputationBlock, sast.StatementBlock]]):

    def __init__(self, placement: ProgramPlacement, versioning: Versioning[spa.Identifier]):
        self.placement = placement
        self.versioning = versioning
        # x (op) a[0, 0, 0]
        e = sast.Expression(
            sast.BinaryOperator(
                sast.Expression(Wildcard[float]("value")()),
                Wildcard("op")(),
                sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0])),
            ))
        assignment_f = sast.AssignOp(Wildcard("dst")(), e, Wildcard()())

        e = sast.Expression(
            sast.BinaryOperator(
                sast.Expression(Wildcard[int]("value")()),
                Wildcard("op")(),
                sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0])),
            ))
        assignment_i = sast.AssignOp(Wildcard("dst")(), e, Wildcard()())

        super().__init__([assignment_i, assignment_f])

    def transform(self,
                  root: sast.AssignOp,
                  op: str = None,
                  value=None,
                  src: sast.Identifier = None,
                  dst: sast.Identifier = None,
                  **wildcards) -> list[AbstractStatement]:
        assert op is not None
        assert src is not None
        assert dst is not None

        res_id, res_dtype = self.placement.get_storage(dst)
        var_k = self.versioning.next_version('k')

        # so we can easily extract the correct operation from the expression
        src_id, src_dtype = self.placement.get_storage(src)

        src_e = spa.Expression(
            spa.BinaryOperator(
                spa.Expression(spa.ConstantLiteral(value, src_dtype.base_type)),
                op,
                spa.Expression(spa.ArraySlice(
                    src_id,
                    [spa.Expression(var_k)]
                )),
            )
        )

        stmt = spa.MapStatement(
            variables=[spa.TypedIdentifier(ScalarType.i32, var_k)],
            range_expression=[spa.RangeExpression.from_args(0, res_dtype.shape[0])],
            body=[
                spa.AssignmentStatement(
                    spa.ArraySlice(
                        res_id,
                        [spa.Expression(var_k)]
                    ),
                    src_e)
            ]
        )

        stmt_block = self.get_context()[1]
        out_t = stmt_block.operation_type.destination[0]
        xy_range = _get_range(out_t, self.placement.get_shift())

        return [AbstractStatement(xy_range[0], xy_range[1], stmt)]


class UnaryMapTransformer(PatternTransformer[sast.AssignOp, AbstractStatement, tuple[sast.ComputationBlock, sast.StatementBlock]]):

    def __init__(self, placement: ProgramPlacement, versioning: Versioning[spa.Identifier]):
        self.placement = placement
        self.versioning = versioning
        # (u_op) x (op) a[0, 0, 0]
        e = sast.Expression(
            sast.BinaryOperator(
                sast.Expression(sast.UnaryOperator(Wildcard("u_op")(), sast.Expression(Wildcard[float]("value")()))),
                Wildcard("op")(),
                sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0])),
            ))
        assignment_f = sast.AssignOp(Wildcard("dst")(), e, Wildcard()())
        e = sast.Expression(
            sast.BinaryOperator(
                sast.Expression(sast.UnaryOperator(Wildcard("u_op")(), sast.Expression(Wildcard[int]("value")()))),
                Wildcard("op")(),
                sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0])),
            ))
        assignment_i = sast.AssignOp(Wildcard("dst")(), e, Wildcard()())

        super().__init__([assignment_i, assignment_f])

    def transform(self,
                  root: sast.AssignOp,
                  u_op: str = None,
                  op: str = None,
                  value=None,
                  src: sast.Identifier = None,
                  dst: sast.Identifier = None,
                  **wildcards) -> list[AbstractStatement]:
        assert op is not None
        assert src is not None
        assert dst is not None

        res_id, res_dtype = self.placement.get_storage(dst)
        var_k = self.versioning.next_version('k')

        # so we can easily extract the correct operation from the expression
        src_id, src_dtype = self.placement.get_storage(src)

        src_e = spa.Expression(
            spa.BinaryOperator(
                spa.Expression(spa.UnaryOperator(u_op, spa.Expression(spa.ConstantLiteral(value, src_dtype.base_type)))),
                op,
                spa.Expression(spa.ArraySlice(
                    src_id,
                    [spa.Expression(var_k)]
                )),
            )
        )

        stmt = spa.MapStatement(
            variables=[spa.TypedIdentifier(ScalarType.i32, var_k)],
            range_expression=[spa.RangeExpression.from_args(0, res_dtype.shape[0])],
            body=[
                spa.AssignmentStatement(
                    spa.ArraySlice(
                        res_id,
                        [spa.Expression(var_k)]
                    ),
                    src_e,
                )
            ]
        )

        stmt_block = self.get_context()[1]
        out_t = stmt_block.operation_type.destination[0]
        xy_range = _get_range(out_t, self.placement.get_shift())

        line_nr = self.versioning.next_version("___line___").version

        return [AbstractStatement(xy_range[0], xy_range[1], (line_nr, stmt))]


class HorizontalStencilTransformer(
    PatternTransformer[sast.AssignOp, AbstractStatement, tuple[sast.ComputationBlock, sast.StatementBlock]]):

    def __init__(self,
                 placement: ProgramPlacement,
                 versioning: Versioning[spa.Identifier],
                 dataflow: ProgramDataflow):
        self.placement = placement
        self.versioning = versioning
        self.dataflow = dataflow
        # %c = (%a[0, 0, 0] + %b[dx, dy, 0]) : f32
        # %c = (%b[dx, dy, 0] + %a[0, 0, 0]) : f32
        # %c = (%b[dx, dy, 0] : f32

        e_1 = sast.Expression(
            value=sast.BinaryOperator(left=sast.Expression(value=sast.Subscript(
                Wildcard[sast.Identifier]('local')(), [0, 0, 0])),
                op=Wildcard("op")(),
                right=sast.Expression(
                    sast.Subscript(Wildcard('remote')(),
                                   [Wildcard[int]('dx')(), Wildcard[int]('dy')(), 0]))))

        e = sast.Expression(
            value=sast.BinaryOperator(right=sast.Expression(value=sast.Subscript(
                Wildcard[sast.Identifier]('local')(), [0, 0, 0])),
                op=Wildcard("op")(),
                left=sast.Expression(
                    sast.Subscript(Wildcard('remote')(),
                                   [Wildcard[int]('dx')(), Wildcard[int]('dy')(), 0]))))

        e_2 = sast.Expression(sast.Subscript(Wildcard('remote')(),
                                            [Wildcard[int]('dx')(), Wildcard[int]('dy')(), 0]))

        assignment_0 = sast.AssignOp(Wildcard[sast.Identifier]("dst")(), e, Wildcard()())
        assignment_1 = sast.AssignOp(Wildcard[sast.Identifier]("dst")(), e_1, Wildcard()())
        assignment_2 = sast.AssignOp(Wildcard[sast.Identifier]("dst")(), e_2, Wildcard()())

        return_0 = sast.ReturnOp([e], Wildcard()())
        return_1 = sast.ReturnOp([e_1], Wildcard()())
        return_2 = sast.ReturnOp([e_2], Wildcard()())

        super().__init__([assignment_0, return_0, assignment_1, return_1, assignment_2, return_2])

    def transform(self,
                  root: sast.AssignOp,
                  op: str = None,
                  local: sast.Identifier = None,
                  remote: sast.Identifier = None,
                  dst: sast.Identifier = None,
                  dx: int = None,
                  dy: int = None,
                  **wildcards) -> list[AbstractStatement]:
        assert remote is not None
        assert dx is not None
        assert dy is not None

        compute_block, stmt_block = self.get_context()
        out_id = stmt_block.outputs[0]

        if dst is None:
            dst = out_id

        # (1) dst buffer
        res_id, res_dtype = self.placement.get_storage(dst)

        # Approach: Communicate the remote values and aggregate them into the local value
        # For this, we need:

        # (2) local buffer
        if local is not None:
            assert op is not None
            local_id, local_dtype = self.placement.get_storage(local)

        # (3) remote buffer
        # Determine if its an input type or an intermediate type

        out_t = stmt_block.operation_type.destination[0]
        shift = self.placement.get_shift()
        xy_range = _get_range(out_t, shift)

        if any([remote == inp for inp in compute_block.inputs]):
            remote_id, remote_dtype = self.placement.get_storage(remote)

            # (4) stream used to communicate the remote buffer
            stream = self.dataflow.get_stream(remote, out_id, sast.Offset((dx, dy, 0)))
            assert stream

            # Loop variables
            var_k = self.versioning.next_version('k')
            var_x = self.versioning.next_version('x')

            recv = spa.ReceiveGenerator(
                stream
            )

            if local is not None:
                src_expr = spa.Expression(
                    spa.BinaryOperator(
                        spa.Expression(local_id),
                        op,
                        spa.Expression(var_x),
                    ))
            else:
                src_expr = spa.Expression(var_x)

            assign_stmt = spa.AssignmentStatement(
                source=src_expr,
                destination=spa.ArraySlice(
                    res_id,
                    [spa.Expression(var_k)]
                )
            )

            body = [assign_stmt]

            recv_comp_id = self.versioning.next_version('_recv_comp')
            recv_completion = spa.Completion(recv_comp_id)
            recv_foreach = spa.ForeachStatement(
                variables=[spa.TypedIdentifier(ScalarType.i32, var_k)],
                parameter_range=[spa.RangeExpression.from_args(0, res_dtype.shape[0])],
                stream_variable=spa.TypedIdentifier(remote_dtype.base_type, var_x),
                receive_stream=recv,
                body=body,
                completion_name=recv_completion,
            )

            line_nr = self.versioning.next_version("___line___").version

            receive = AbstractStatement(xy_range[0], xy_range[1], (line_nr, recv_foreach))

            send_comp_id = self.versioning.next_version('_send_comp')
            send_completion = spa.Completion(send_comp_id)
            send = spa.SendStatement(
                remote_id,
                stream,
                send_completion
            )

            send_domain = out_t.domain.union(out_t.domain.add((dx, dy, 0)))

            send_x_range = _shift(send_domain.x, shift[0])
            send_y_range = _shift(send_domain.y, shift[1])

            line_nr = self.versioning.next_version("___line___").version
            send_stmt = AbstractStatement(send_x_range, send_y_range, (line_nr, send))

            line_nr = self.versioning.next_version("___line___").version
            await_send = AbstractStatement(send_x_range, send_y_range,
                                           (line_nr, spa.AwaitCompletionStatement(send_comp_id)))

            line_nr = self.versioning.next_version("___line___").version
            await_recv = AbstractStatement(xy_range[0], xy_range[1],
                                           (line_nr, spa.AwaitCompletionStatement(recv_comp_id)))

            return [receive, send_stmt, await_send, await_recv]

        elif local is not None:
            # (4) materialized buffer (already computed)
            # Only local computation is needed
            remote_id, remote_dtype = self.placement.get_storage(remote, sast.Offset((dx, dy, 0)))

            var_k = self.versioning.next_version('k')

            src_e = spa.Expression(
                spa.BinaryOperator(
                    spa.Expression(spa.ArraySlice(
                        local_id,
                        [spa.Expression(var_k)]
                    )),
                    op,
                    spa.Expression(spa.ArraySlice(
                        remote_id,
                        [spa.Expression(var_k)]
                    )),
                )
            )

            stmt = spa.MapStatement(
                variables=[spa.TypedIdentifier(ScalarType.i32, var_k)],
                range_expression=[spa.RangeExpression.from_args(0, res_dtype.shape[0])],
                body=[
                    spa.AssignmentStatement(
                        spa.ArraySlice(
                            res_id,
                            [spa.Expression(var_k)]
                        ),
                        src_e,
                    )
                ]
            )

            line_nr = self.versioning.next_version("___line___").version
            return [AbstractStatement(xy_range[0], xy_range[1], (line_nr, stmt))]


def _get_range(access_type: sast.ViewType | sast.FieldType, shift: tuple) -> tuple:
    assert isinstance(access_type, (sast.ViewType, sast.FieldType))
    assert isinstance(access_type.domain, sast.Cartesian)
    x_range = _shift(access_type.domain.x, shift[0])
    y_range = _shift(access_type.domain.y, shift[1])
    return x_range, y_range


def _shift(_range: tuple | sast.Interval, shift: int) -> tuple:
    return _range[0] + shift, _range[1] + shift
