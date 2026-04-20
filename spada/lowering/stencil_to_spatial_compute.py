import copy
from dataclasses import dataclass

from spada.lowering.stencil_to_spatial_compute_fwbw import ForwardBackwardComputeVisitor
from spada.lowering.stencil_to_spatial_dataflow import ProgramDataflow
from spada.lowering.stencil_to_spatial_place import ProgramPlacement
from spada.lowering.versioning import Versioning
from spada.syntax.common.basenode import Wildcard
from spada.syntax.common.tree_matching import PatternTransformer
from spada.syntax.common.types import ScalarType
from spada.syntax.spatial_ir.grid_geometry import Rectangle, group_rectangles_by_domain, split_rectangles
from spada.syntax.stencil_ir.domain_collector import DomainCollector
import spada.syntax.spatial_ir.irnodes as spa
import spada.syntax.stencil_ir.irnodes as sast

AbstractStatement = Rectangle[tuple[int, spa.Statement]]


class ProgramCompute:

    def __init__(self,
                 versioning: Versioning[spa.Identifier],
                 dataflow: ProgramDataflow,
                 placement: ProgramPlacement,
                 subgrid_var_type: ScalarType = ScalarType.u16):
        self.versioning = versioning
        self.dataflow = dataflow
        self.placement = placement
        self.visitor = ParallelComputeVisitor(placement, versioning, dataflow)
        self.vertical_visitor = ForwardBackwardComputeVisitor(placement, versioning, dataflow)
        self.grid_var_t = subgrid_var_type

    def generate_computation(self, comp: sast.ComputationBlock) -> list[spa.ComputeBlock]:
        """
        Generate a computation block.

        """
        # Generate the compute block
        if comp.schedule == sast.ComputationType.PARALLEL:
            self.visitor.visit(comp)
            body = self.visitor.stmts
        else:
            self.vertical_visitor.visit(comp)
            body = self.vertical_visitor.stmts

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

        variables = [spa.TypedIdentifier(self.grid_var_t, var_i), spa.TypedIdentifier(self.grid_var_t, var_j)]

        subgrid = spa.SubgridExpression.from_tuple(block[0].x_range, block[0].y_range)

        stmts = sorted(block, key=lambda x: x.metadata[0])
        
        block = spa.ComputeBlock(variables, subgrid, [stmt.metadata[1] for stmt in stmts if stmt.metadata[1]])

        return block


class DummyReceiveTransformer(spa.NodeTransformer):
    """Cleans up dummy receive foreach loops.
    These have to be inserted to avoid deadlocks at the edges of computations, but should not actually perform
    any computations themselves.
    This transformer deletes the body of these loops.
    """
    _dataflow: ProgramDataflow
    _current_block: spa.ComputeBlock | None
    
    def __init__(self, dataflow: ProgramDataflow):
        super().__init__()
        self._dataflow = dataflow
        self._current_block = None
        
        
    def visit_ComputeBlock(self, blk: spa.ComputeBlock):
        self._current_block = blk
        self.generic_visit(blk)
        self._current_block = None
        return blk
    
    def visit_ForeachStatement(self, stmt: spa.ForeachStatement):
        rcv = stmt.receive_stream
        
        block_domain = self._current_block.get_grid_rect()
        block_domain_stride = self._current_block.get_grid_stride()
        
        block_rect = Rectangle[int]((block_domain[0], block_domain[1], block_domain_stride[0]), (block_domain[2], block_domain[3], block_domain_stride[1]), 0)
        send_domain_x, send_domain_y = self._dataflow.stream_send_range_map[rcv.stream_name]
        send_rect = Rectangle[int](send_domain_x, send_domain_y, 1)
        
        if not block_rect.intersects(send_rect):
            stmt = copy.deepcopy(stmt)
            assign = stmt.body[0]
            if isinstance(assign, spa.AssignmentStatement):
                assign.source = spa.Expression(stmt.stream_variable.identifier)
            #stmt.body = []

        return stmt

@dataclass(frozen=True)
class TransformerContext:
    comp: sast.ComputationBlock
    stmt: sast.StatementBlock
    # Indicates that we are operating on the index-th result of the statement
    index: int = 0


class ParallelComputeVisitor(sast.ScopedNodeVisitor):

    def __init__(self, placement: ProgramPlacement, versioning: Versioning[spa.Identifier], dataflow: ProgramDataflow):
        super().__init__()
        self.placement = placement
        self.versioning = versioning
        self.dataflow = dataflow
        self.stmts = []

        self.statement_transformers = [
            UnaryMapTransformer(placement, versioning, dataflow),
            MapTransformer(placement, versioning, dataflow),
            TernaryMapTransformer(placement, versioning, dataflow),
            HorizontalStencilTransformer(placement, versioning, dataflow)
        ]

    def visit_ReturnOp(self, op: sast.ReturnOp):
        comp = self.get_scope()
        assert isinstance(comp, sast.ComputationBlock)

        for value, value_t, out, out_t in zip(op.values, op.operation_type.source, comp.outputs,
                                              comp.operation_type.destination):
            value = value.value
            assert isinstance(value, sast.Identifier)

            x_range, y_range = self.dataflow.get_x_y_range(out_t, 0, 0)

            var_k = self.versioning.next_version('k')

            dst_range = (value_t.domain.z[0], value_t.domain.z[1])
            src_range = (out_t.domain.z[0], out_t.domain.z[1])

            translation = src_range[0] - dst_range[0]

            src_id, src_dtype = self.placement.get_storage(value)

            if isinstance(src_dtype, spa.ArrayType):
                src_e = spa.Expression(spa.ArraySlice(src_id, [spa.Expression(var_k)]))
            else:
                src_e = spa.Expression(src_id)

            dst_id, dst_dtype = self.placement.get_storage(out)

            if translation > 0:

                dst_e = spa.ArraySlice(dst_id, [
                    spa.RangeExpression(
                        spa.Expression(
                            spa.BinaryOperator(
                                spa.Expression(var_k), '+',
                                spa.Expression(spa.ConstantLiteral(translation, ScalarType.i32)))))
                ])
            else:
                dst_e = spa.ArraySlice(dst_id, [spa.Expression(var_k)])

            stmt = spa.MapStatement(
                variables=[spa.TypedIdentifier(ScalarType.i32, var_k)],
                range_expression=[spa.RangeExpression.from_args(0, out_t.domain.z[1])],
                body=[spa.AssignmentStatement(
                    dst_e,
                    src_e,
                )])

            line_nr = self.versioning.next_version("___line___").version
            self.stmts.append(AbstractStatement(x_range, y_range, (line_nr, stmt)))

    def visit_StatementBlock(self, op: sast.StatementBlock):
        comp = self.get_scope()
        assert isinstance(comp, sast.ComputationBlock)

        for transformer in self.statement_transformers:
            transformer.set_context(TransformerContext(comp, op, 0))

        for stmt in op.body:
            statements = self._apply_statement_transformers(stmt)
            assert len(statements) > 0, f"Could not match statement {stmt.as_ir()} \n {stmt}"
            self.stmts.extend(statements)

    def _apply_statement_transformers(self, op: sast.AssignOp | sast.ReturnOp) -> list[AbstractStatement]:
        blocks = []
        for transformer in self.statement_transformers:
            res = []
            if isinstance(op, sast.AssignOp):
                res = transformer.first(op)
            else:
                assert isinstance(op, sast.ReturnOp)
                ctxt: TransformerContext = transformer.get_context()
                for i, r in enumerate(op.values):
                    transformer.set_context(TransformerContext(ctxt.comp, ctxt.stmt, i))
                    # We handle multiple return types by applying the transformer to each return value
                    # in sequence
                    optype = sast.OperationType(source=[op.operation_type.source[i]], destination=None)
                    synthetic_return = sast.ReturnOp([r], optype)
                    res.extend(transformer.first(synthetic_return))

            if len(res):
                blocks.extend(res)
                break

        for stmt in blocks:
            assert isinstance(stmt, Rectangle)
            assert isinstance(stmt.metadata, tuple)
        return blocks

    def visit_MaterializeOp(self, op: sast.MaterializeOp):

        # The materialize operation creates data movement for each offset in its output offsets
        # that is not zero
        dst = op.result
        src = op.value
        out_t = op.operation_type.destination[0]
        for extent in op.operation_type.destination[0].extent.extents:
            if extent != sast.Offset.zero():
                dst_buf, dst_dtype = self.placement.get_storage(dst, extent)
                
                dx, dy, dz = extent.values

                # Approach: Communicate the remote values and aggregate them into the local value
                # For this, we need:

                # (2) local buffer
                src_buf, src_dtype = self.placement.get_storage(src)

                # (3) remote buffer
                # Determine if its an input type or an intermediate type
                dst_t = out_t
                xy_range = self.dataflow.get_x_y_receive_range(dst_t, dx, dy)

                # (4) stream used to communicate the remote buffer
                stream = self.dataflow.get_stream(src, dst, extent)
                assert stream

                # Loop variables
                var_k = self.versioning.next_version('k')
                var_x = self.versioning.next_version('x')

                recv = spa.ReceiveGenerator(stream)

                src_expr = spa.Expression(var_x)

                assign_stmt = spa.AssignmentStatement(
                    source=src_expr, destination=spa.ArraySlice(dst_buf, [spa.Expression(var_k)]))

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
                send = spa.SendStatement(src_buf, stream, send_completion)

                send_x_range, send_y_range = self.dataflow.get_x_y_send_range(dst_t, dx, dy)

                line_nr = self.versioning.next_version("___line___").version
                send_stmt = AbstractStatement(send_x_range, send_y_range, (line_nr, send))

                line_nr = self.versioning.next_version("___line___").version
                await_send = AbstractStatement(send_x_range, send_y_range,
                                               (line_nr, spa.AwaitCompletionStatement(send_comp_id)))

                line_nr = self.versioning.next_version("___line___").version
                await_recv = AbstractStatement(xy_range[0], xy_range[1],
                                               (line_nr, spa.AwaitCompletionStatement(recv_comp_id)))

                self.stmts.extend([receive, send_stmt, await_send, await_recv])


class MapTransformer(PatternTransformer[sast.AssignOp | sast.ReturnOp, AbstractStatement, TransformerContext]):

    def __init__(self, placement: ProgramPlacement, versioning: Versioning[spa.Identifier], dataflow: ProgramDataflow):
        self.placement = placement
        self.versioning = versioning
        self.dataflow = dataflow
        # x (op) a[0, 0, 0]
        e_0 = sast.Expression(
            sast.BinaryOperator(
                sast.Expression(Wildcard[float]("value")()),
                Wildcard("op")(),
                sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0])),
            ))

        e_1 = sast.Expression(
            sast.BinaryOperator(
                sast.Expression(Wildcard[int]("value")()),
                Wildcard("op")(),
                sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0])),
            ))

        e_2 = sast.Expression(
            sast.BinaryOperator(
                sast.Expression(sast.Subscript(Wildcard("src2")(), [0, 0, 0])),
                Wildcard("op")(),
                sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0])),
            ))

        # Copy operator (op and value are none)
        e_3 = sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0]))

        e_0_neg = sast.Expression(
            sast.BinaryOperator(
                sast.Expression(
                    sast.UnaryOperator(Wildcard[str]("unary_op")(), sast.Expression(Wildcard[float]("value")()))),
                Wildcard("op")(),
                sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0])),
            ))

        e_1_neg = sast.Expression(
            sast.BinaryOperator(
                sast.Expression(
                    sast.UnaryOperator(Wildcard[str]("unary_op")(), sast.Expression(Wildcard[int]("value")()))),
                Wildcard("op")(),
                sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0])),
            ))

        assignments = [
            sast.AssignOp(Wildcard[sast.Identifier]("dst")(), e,
                          Wildcard()()) for e in [e_0, e_1, e_2, e_3, e_1_neg, e_0_neg]
        ]
        returns = [sast.ReturnOp([e], Wildcard()()) for e in [e_0, e_1, e_2, e_3, e_1_neg, e_0_neg]]

        super().__init__(assignments + returns)

    def transform(self,
                  root: sast.AssignOp,
                  op: str = None,
                  unary_op: str = None,
                  value=None,
                  src: sast.Identifier = None,
                  src2: sast.Identifier = None,
                  dst: sast.Identifier = None,
                  **wildcards) -> list[AbstractStatement]:
        assert src is not None

        src_id, src_dtype = self.placement.get_storage(src)

        context = self.get_context()
        compute_block, stmt_block = context.comp, context.stmt

        if dst is None:
            assert isinstance(root, sast.ReturnOp)
            # Return statement has an implicit destination to the i-th output of the statement block
            dst = stmt_block.outputs[context.index]
        res_id, res_dtype = self.placement.get_storage(dst)
        
        var_k = self.versioning.next_version('k')

        if isinstance(src_dtype, spa.ArrayType):
            src_1_expr = spa.Expression(spa.ArraySlice(src_id, [spa.Expression(var_k)]))
        else:
            src_1_expr = spa.Expression(src_id)

        if src2 is None:

            if value is None and op is None:
                src_e = src_1_expr
            else:
                assert value is not None and op is not None

                if unary_op is None:
                    const_expr = spa.ConstantLiteral(value, src_dtype.base_type)
                else:
                    const_expr = spa.UnaryOperator(unary_op,
                                                   spa.Expression(spa.ConstantLiteral(value, src_dtype.base_type)))

                src_e = spa.Expression(
                    spa.BinaryOperator(
                        spa.Expression(const_expr),
                        op,
                        src_1_expr,
                    ))
        else:
            src2_id, src2_dtype = self.placement.get_storage(src2)
            
            if isinstance(src2_dtype, spa.ScalarType):
                src2_expr = spa.Expression(src2_id)
            else:
                src2_expr = spa.Expression(spa.ArraySlice(src2_id, [spa.Expression(var_k)]))
                
            src_e = spa.Expression(
                spa.BinaryOperator(
                    src2_expr,
                    op,
                    src_1_expr,
                ))

        stmt = spa.MapStatement(
            variables=[spa.TypedIdentifier(ScalarType.i32, var_k)],
            range_expression=[spa.RangeExpression.from_args(0, res_dtype.shape[0])],
            body=[spa.AssignmentStatement(spa.ArraySlice(res_id, [spa.Expression(var_k)]), src_e)])

        stmt_block = self.get_context().stmt
        out_t = stmt_block.operation_type.destination[0]
        xy_range = self.dataflow.get_x_y_range(out_t, 0, 0)

        line_nr = self.versioning.next_version("___line___").version

        return [AbstractStatement(xy_range[0], xy_range[1], (line_nr, stmt))]


class UnaryMapTransformer(PatternTransformer[sast.AssignOp | sast.ReturnOp, AbstractStatement, TransformerContext]):

    def __init__(self, placement: ProgramPlacement, versioning: Versioning[spa.Identifier], dataflow: ProgramDataflow):
        self.placement = placement
        self.versioning = versioning
        self.dataflow = dataflow
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

        if isinstance(src_dtype, spa.ScalarType):
            src_expr = spa.Expression(src_id)
        else:
            src_expr = spa.Expression(spa.ArraySlice(src_id, [spa.Expression(var_k)]))

        src_e = spa.Expression(
            spa.BinaryOperator(
                spa.Expression(
                    spa.UnaryOperator(u_op, spa.Expression(spa.ConstantLiteral(value, src_dtype.base_type)))),
                op,
                src_expr,
            ))

        stmt = spa.MapStatement(
            variables=[spa.TypedIdentifier(ScalarType.i32, var_k)],
            range_expression=[spa.RangeExpression.from_args(0, res_dtype.shape[0])],
            body=[spa.AssignmentStatement(
                spa.ArraySlice(res_id, [spa.Expression(var_k)]),
                src_e,
            )])

        stmt_block = self.get_context().stmt
        out_t = stmt_block.operation_type.destination[0]
        xy_range = self.dataflow.get_x_y_range(out_t, 0, 0)

        line_nr = self.versioning.next_version("___line___").version

        return [AbstractStatement(xy_range[0], xy_range[1], (line_nr, stmt))]


class TernaryMapTransformer(PatternTransformer[sast.AssignOp | sast.ReturnOp, AbstractStatement, TransformerContext]):

    def __init__(self, placement: ProgramPlacement, versioning: Versioning[spa.Identifier], dataflow: ProgramDataflow):
        self.placement = placement
        self.versioning = versioning
        self.dataflow = dataflow

        patterns = []

        for const_type in (int, float):
            for const2_type in (int, float):
                # const if (b[0,0,0] op const_cond) else a[0,0,0]
                patterns.append(
                    sast.Expression(
                        sast.TernaryOperator(
                            sast.Expression(Wildcard[const_type]("value")()),
                            sast.Expression(
                                sast.BinaryOperator(
                                    sast.Expression(sast.Subscript(Wildcard("src_cond")(), [0, 0, 0])),
                                    Wildcard("op")(),
                                    sast.Expression(Wildcard[const2_type]("value_cond")()),
                                )),
                            sast.Expression(sast.Subscript(Wildcard("src_false")(), [0, 0, 0])),
                        )))

                # a[0,0,0] if (b[0,0,0] op const_cond) else const
                patterns.append(
                    sast.Expression(
                        sast.TernaryOperator(
                            sast.Expression(sast.Subscript(Wildcard("src_true")(), [0, 0, 0])),
                            sast.Expression(
                                sast.BinaryOperator(
                                    sast.Expression(sast.Subscript(Wildcard("src_cond")(), [0, 0, 0])),
                                    Wildcard("op")(),
                                    sast.Expression(Wildcard[const2_type]("value_cond")()),
                                )),
                            sast.Expression(Wildcard[const_type]("value")()),
                        )))

        assignments = [sast.AssignOp(Wildcard[sast.Identifier]("dst")(), e, Wildcard()()) for e in patterns]
        returns = [sast.ReturnOp([e], Wildcard()()) for e in patterns]

        super().__init__(assignments + returns)

    def transform(self,
                  root: sast.AssignOp | sast.ReturnOp,
                  op: str = None,
                  value=None,
                  value_cond=None,
                  src_false: sast.Identifier = None,
                  src_true: sast.Identifier = None,
                  src_cond: sast.Identifier = None,
                  dst: sast.Identifier = None,
                  **wildcards) -> list[AbstractStatement]:
        assert src_false is not None or src_true is not None
        assert src_cond is not None
        assert src_false is None or src_true is None, "Only one of src_false or src_true should be set"

        src_id, src_dtype = self.placement.get_storage(src_false or src_true)
        src_cond_id, src_cond_dtype = self.placement.get_storage(src_cond)

        context = self.get_context()
        compute_block, stmt_block = context.comp, context.stmt

        if dst is None:
            assert isinstance(root, sast.ReturnOp)
            # Return statement has an implicit destination to the i-th output of the statement block
            dst = stmt_block.outputs[context.index]
        res_id, res_dtype = self.placement.get_storage(dst)

        var_k = self.versioning.next_version('k')

        arr_operand = spa.Expression(spa.ArraySlice(src_id, [spa.Expression(var_k)]))
        const_operand = spa.Expression(spa.ConstantLiteral(value, src_dtype.base_type))

        src_e = spa.Expression(
            spa.TernaryOperator(
                arr_operand if src_true is not None else const_operand,
                spa.Expression(
                    spa.BinaryOperator(
                        spa.Expression(spa.ArraySlice(src_cond_id, [spa.Expression(var_k)])), op,
                        spa.Expression(spa.ConstantLiteral(value_cond, src_cond_dtype.base_type)))),
                const_operand if src_true is not None else arr_operand,
            ))

        stmt = spa.MapStatement(
            variables=[spa.TypedIdentifier(ScalarType.i32, var_k)],
            range_expression=[spa.RangeExpression.from_args(0, res_dtype.shape[0])],
            body=[spa.AssignmentStatement(spa.ArraySlice(res_id, [spa.Expression(var_k)]), src_e)])

        stmt_block = self.get_context().stmt
        out_t = stmt_block.operation_type.destination[0]
        xy_range = self.dataflow.get_x_y_range(out_t, 0, 0)

        line_nr = self.versioning.next_version("___line___").version

        return [AbstractStatement(xy_range[0], xy_range[1], (line_nr, stmt))]


class HorizontalStencilTransformer(PatternTransformer[sast.AssignOp | sast.ReturnOp, AbstractStatement,
                                                      TransformerContext]):

    def __init__(self, placement: ProgramPlacement, versioning: Versioning[spa.Identifier], dataflow: ProgramDataflow):
        self.placement = placement
        self.versioning = versioning
        self.dataflow = dataflow
        # %c = (%a[0, 0, 0] + %b[dx, dy, 0]) : f32
        # %c = (%b[dx, dy, 0] + %a[0, 0, 0]) : f32
        # %c = (%b[dx, dy, 0] : f32
        # %d = %factor * %b[dx, dy, 0] : f32

        e_1 = sast.Expression(
            value=sast.BinaryOperator(
                left=sast.Expression(value=sast.Subscript(Wildcard[sast.Identifier]('local')(), [0, 0, 0])),
                op=Wildcard("op")(),
                right=sast.Expression(
                    sast.Subscript(Wildcard('remote')(), [Wildcard[int]('dx')(), Wildcard[int]('dy')(), 0]))))

        e = sast.Expression(
            value=sast.BinaryOperator(
                right=sast.Expression(value=sast.Subscript(Wildcard[sast.Identifier]('local')(), [0, 0, 0])),
                op=Wildcard("op")(),
                left=sast.Expression(
                    sast.Subscript(Wildcard('remote')(), [Wildcard[int]('dx')(), Wildcard[int]('dy')(), 0]))))

        e_2 = sast.Expression(sast.Subscript(Wildcard('remote')(), [Wildcard[int]('dx')(), Wildcard[int]('dy')(), 0]))

        e_3 = sast.Expression(
            sast.BinaryOperator(
                sast.Expression(Wildcard[int]('factor')()),
                Wildcard('op')(),
                sast.Expression(
                    sast.Subscript(Wildcard('remote')(), [Wildcard[int]('dx')(), Wildcard[int]('dy')(), 0]))))

        e_4 = sast.Expression(
            sast.BinaryOperator(
                sast.Expression(
                    sast.Subscript(Wildcard('remote')(), [Wildcard[int]('dx')(), Wildcard[int]('dy')(), 0])),
                Wildcard('op')(), sast.Expression(Wildcard[int]('factor')())))

        e_5 = sast.Expression(
            sast.BinaryOperator(
                sast.Expression(Wildcard[float]('factor')()),
                Wildcard('op')(),
                sast.Expression(
                    sast.Subscript(Wildcard('remote')(), [Wildcard[int]('dx')(), Wildcard[int]('dy')(), 0]))))

        e_6 = sast.Expression(
            sast.BinaryOperator(
                sast.Expression(
                    sast.Subscript(Wildcard('remote')(), [Wildcard[int]('dx')(), Wildcard[int]('dy')(), 0])),
                Wildcard('op')(), sast.Expression(Wildcard[float]('factor')())))

        exprs = [e, e_1, e_2, e_3, e_4, e_5, e_6]

        patterns: list = [sast.AssignOp(Wildcard[sast.Identifier]("dst")(), exp, Wildcard()()) for exp in exprs]
        patterns.extend([sast.ReturnOp([exp], Wildcard()()) for exp in exprs])

        super().__init__(patterns)

    def transform(self,
                  root: sast.AssignOp | sast.ReturnOp,
                  op: str = None,
                  local: sast.Identifier = None,
                  remote: sast.Identifier = None,
                  dst: sast.Identifier = None,
                  dx: int = None,
                  dy: int = None,
                  factor: int = None,
                  **wildcards) -> list[AbstractStatement]:
        assert remote is not None
        assert dx is not None
        assert dy is not None

        context = self.get_context()
        compute_block, stmt_block = context.comp, context.stmt
        out_id = stmt_block.outputs[0]

        if dst is None:
            assert isinstance(root, sast.ReturnOp)
            # Return statement has an implicit destination to the i-th output of the statement block
            dst = stmt_block.outputs[context.index]

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
        out_t = stmt_block.operation_type.destination[context.index]

        if any([remote == inp for inp in compute_block.inputs]):
            if dx == 0 and dy == 0:
                # Horizontal stencil with no offset -> does not match
                return []
            # (4) stream used to communicate the remote buffer
            stream = self.dataflow.get_stream(remote, out_id, sast.Offset((dx, dy, 0)))
            assert stream, f"Stream not found for {remote.as_ir()} -> {out_id.as_ir()} with offset ({dx, dy}, 0)"

            # Loop variables
            var_k = self.versioning.next_version('k')
            var_x = self.versioning.next_version('x')

            # Build the source expression
            if local is not None:
                
                local_array_expr = spa.ArraySlice(
                    local_id,
                    [spa.Expression(var_k)]
                )
                
                src_expr = spa.Expression(spa.BinaryOperator(
                    spa.Expression(local_array_expr),
                    op,
                    spa.Expression(var_x),
                ))
                
            elif factor is not None:
                # %d = %factor * %b[dx, dy, 0] : f32
                if isinstance(factor, int):
                    factor = spa.ConstantLiteral(factor, ScalarType.i32)
                elif isinstance(factor, float):
                    factor = spa.ConstantLiteral(factor, ScalarType.f32)
                src_expr = spa.Expression(spa.BinaryOperator(spa.Expression(factor), op, spa.Expression(var_x)))
            else:
                # %c = (%b[dx, dy, 0] : f32
                src_expr = spa.Expression(var_x)

            return _send_receive_statement(self.dataflow, self.versioning, self.placement, remote, out_t, dx, dy,
                                           res_id, res_dtype, out_id, var_x, var_k, src_expr)
        else:
            xy_range = self.dataflow.get_x_y_range(out_t, 0, 0)

            # (4) materialized buffer (already computed)
            # Only local computation is needed
            remote_id, remote_dtype = self.placement.get_storage(remote, sast.Offset((dx, dy, 0)))

            var_k = self.versioning.next_version('k')

            # Build the source expression
            if local is not None:
                src_e = spa.Expression(
                    spa.BinaryOperator(
                        spa.Expression(spa.ArraySlice(local_id, [spa.Expression(var_k)])),
                        op,
                        spa.Expression(spa.ArraySlice(remote_id, [spa.Expression(var_k)])),
                    ))
            elif factor is not None:
                src_e = spa.Expression(
                    spa.BinaryOperator(
                        spa.Expression(spa.ConstantLiteral(factor, ScalarType.i32)),
                        op,
                        spa.Expression(spa.ArraySlice(remote_id, [spa.Expression(var_k)])),
                    ))
            else:
                src_e = spa.Expression(spa.ArraySlice(remote_id, [spa.Expression(var_k)]))

            stmt = spa.MapStatement(
                variables=[spa.TypedIdentifier(ScalarType.i32, var_k)],
                range_expression=[spa.RangeExpression.from_args(0, res_dtype.shape[0])],
                body=[spa.AssignmentStatement(
                    spa.ArraySlice(res_id, [spa.Expression(var_k)]),
                    src_e,
                )])

            line_nr = self.versioning.next_version("___line___").version
            return [AbstractStatement(xy_range[0], xy_range[1], (line_nr, stmt))]


def _send_receive_statement(dataflow: ProgramDataflow, versioning: Versioning[spa.Identifier],
                            placement: ProgramPlacement, remote: sast.Identifier, out_t: sast.DataType, dx: int,
                            dy: int, res_id: spa.Identifier, res_dtype: spa.ArrayType, out_id: spa.Identifier,
                            var_x: spa.Identifier, var_k: spa.Identifier,
                            src_expr: spa.Expression) -> list[AbstractStatement]:
    # stream used to communicate the remote buffer
    remote_id, remote_dtype = placement.get_storage(remote)
    stream = dataflow.get_stream(remote, out_id, sast.Offset((dx, dy, 0)))
    assert stream

    send_range_x, send_range_y = dataflow.get_x_y_send_range(out_t, dx, dy)
    receive_range_x, receive_range_y = dataflow.get_x_y_receive_range(out_t, dx, dy)

    recv = spa.ReceiveGenerator(stream)

    assign_stmt = spa.AssignmentStatement(source=src_expr, destination=spa.ArraySlice(res_id, [spa.Expression(var_k)]))

    body = [assign_stmt]

    recv_comp_id = versioning.next_version('_recv_comp')
    recv_completion = spa.Completion(recv_comp_id)
    recv_foreach = spa.ForeachStatement(
        variables=[spa.TypedIdentifier(ScalarType.i32, var_k)],
        parameter_range=[spa.RangeExpression.from_args(0, res_dtype.shape[0])],
        stream_variable=spa.TypedIdentifier(remote_dtype.base_type, var_x),
        receive_stream=recv,
        body=body,
        completion_name=recv_completion,
    )

    line_nr = versioning.next_version("___line___").version

    receive = AbstractStatement(receive_range_x, receive_range_y, (line_nr, recv_foreach))

    send_comp_id = versioning.next_version('_send_comp')
    send_completion = spa.Completion(send_comp_id)
    send = spa.SendStatement(remote_id, stream, send_completion)

    line_nr = versioning.next_version("___line___").version
    send_stmt = AbstractStatement(send_range_x, send_range_y, (line_nr, send))

    line_nr = versioning.next_version("___line___").version
    await_send = AbstractStatement(send_range_x, send_range_y, (line_nr, spa.AwaitCompletionStatement(send_comp_id)))

    line_nr = versioning.next_version("___line___").version
    await_recv = AbstractStatement(receive_range_x, receive_range_y,
                                   (line_nr, spa.AwaitCompletionStatement(recv_comp_id)))

    return [receive, send_stmt, await_send, await_recv]
