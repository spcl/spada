"""
Parses DSD operations from IR nodes.
"""
import copy
from dataclasses import dataclass
from typing import Literal, Optional
from spada.syntax.spatial_ir import irnodes as spir
from spada.syntax.csl import structures as cslstruct

UniqueDSDDict = dict[str, list[tuple[str, cslstruct.DataStructureDescriptor]]]


@dataclass
class AsyncTarget:
    target_task: str
    inter_task_edge: Literal["activate", "unblock"]


class DSDOp:
    """ Class representing a DSD operation that can be lowered to CSL. """

    def _append_async_suffix(self, base: str, dsd_objects: list[cslstruct.DataStructureDescriptor],
                             async_target: Optional[AsyncTarget]) -> str:
        if async_target is None:
            return base
        if any(isinstance(dsd, cslstruct.FabricDSD) for dsd in dsd_objects):
            return f'{base[:-2]}, .{{ .async = true, .{async_target.inter_task_edge} = {async_target.target_task} }});'
        else:
            # Pure Memory DSD operations are synchronous
            return f'{base}\n@{async_target.inter_task_edge}({async_target.target_task});'

    def as_csl(self,
               statement: spir.Statement,
               dtypes: dict[spir.Identifier, spir.IRType],
               dsds: UniqueDSDDict,
               async_target: Optional[AsyncTarget] = None) -> str:
        """ Returns the CSL representation of the DSD operation, including asynchronous activation if requested. """
        if isinstance(statement, spir.ForeachStatement):
            # Add generator to DSDs
            dsds = copy.copy(dsds)
            dsds[statement.stream_variable.identifier.as_ir()] = dsds[_ident(
                statement.receive_stream.stream_name).as_ir()]
        normalized_statement: Optional[spir.AssignmentStatement | spir.SendStatement]
        if isinstance(statement, (spir.AssignmentStatement, spir.SendStatement)):
            normalized_statement = statement
        else:
            normalized_statement = get_dsd_statement(dtypes, statement)
        if normalized_statement is None:
            raise ValueError('Expected a statement that can be lowered to a DSD operation')
        dsd_objects = self.used_dsd_objects(normalized_statement, dsds)
        return self._append_async_suffix(self._as_csl(normalized_statement, dtypes, dsds), dsd_objects, async_target)

    def _as_csl(self, statement: spir.Statement, dtypes: dict[spir.Identifier, spir.IRType],
                dsds: UniqueDSDDict) -> str:
        """
        Internal method that needs to be implemented by every DSD operation for specific Spatial IR statement
        parsing.

        :param dtypes: A mapping of identifiers to their data types.
        :param statement: The Spatial IR statement to convert.
        :return: The CSL representation of the DSD operation.
        """
        raise NotImplementedError

    def used_dsd_objects(self, statement: spir.AssignmentStatement,
                         dsds: UniqueDSDDict) -> list[cslstruct.DataStructureDescriptor]:
        """
        Identifies the DSD objects used in the given assignment statement.

        :param statement: The assignment statement to analyze.
        :param dsds: The mapping of DSD names to their descriptors.
        :return: A list of DSD objects used in the statement.
        """
        raise NotImplementedError


def _ident(expr: spir.Identifier | spir.ArraySlice | spir.TypedIdentifier) -> spir.Identifier:
    if isinstance(expr, spir.Identifier):
        return expr
    elif isinstance(expr, spir.ArraySlice):
        return expr.array
    elif isinstance(expr, spir.TypedIdentifier):
        return expr.identifier
    elif isinstance(expr, spir.Expression):
        return _ident(expr.value)
    raise TypeError(f"Unsupported expression type: {type(expr)}")


def _ident_or_const(expr: spir.SpatialNode) -> spir.Identifier | spir.ConstantLiteral:
    if isinstance(expr, spir.Expression):
        return _ident_or_const(expr.value)
    if isinstance(expr, spir.ConstantLiteral):
        return expr
    else:
        return _ident(expr)


def _dsd(dsds: UniqueDSDDict, expr: spir.SpatialNode, output: bool = False) -> str:
    from spada.syntax.csl.statements import name_to_csl
    if isinstance(expr, spir.Identifier):
        if expr.as_ir() not in dsds:
            return name_to_csl(expr)
        if output:
            # Find fabout DSD, if exists
            for dsd in dsds[expr.as_ir()]:
                if isinstance(dsd[1], cslstruct.FabricDSD) and dsd[1].dsd_type == cslstruct.DSDType.fabout:
                    return dsd[0]
        else:
            # Find fabin DSD, if exists
            for dsd in dsds[expr.as_ir()]:
                if isinstance(dsd[1], cslstruct.FabricDSD) and dsd[1].dsd_type == cslstruct.DSDType.fabin:
                    return dsd[0]

        # If no fabin/fabout, return memory DSD
        for dsd in dsds[expr.as_ir()]:
            if isinstance(dsd[1], cslstruct.MemoryDSD):
                return dsd[0]

        # If all else fails, return first DSD
        return dsds[expr.as_ir()][0][0]
    elif isinstance(expr, spir.ConstantLiteral):
        return str(float(expr.value))
    raise TypeError(f"Unsupported expression type: {type(expr)}")


def _dsd_object(dsds: UniqueDSDDict, expr: spir.SpatialNode, output: bool = False) -> str:
    from spada.syntax.csl.statements import name_to_csl
    if isinstance(expr, spir.Identifier):
        if expr.as_ir() not in dsds:
            return name_to_csl(expr)
        if output:
            # Find fabout DSD, if exists
            for dsd in dsds[expr.as_ir()]:
                if isinstance(dsd[1], cslstruct.FabricDSD) and dsd[1].dsd_type == cslstruct.DSDType.fabout:
                    return dsd[1]
        return dsds[expr.as_ir()][0][1]
    elif isinstance(expr, spir.ConstantLiteral):
        return str(float(expr.value))
    raise TypeError(f"Unsupported expression type: {type(expr)}")


class UnaryDSDOp(DSDOp):

    def used_dsd_objects(self, statement: spir.AssignmentStatement,
                         dsds: UniqueDSDDict) -> list[cslstruct.DataStructureDescriptor]:
        assert isinstance(statement.source.value, spir.UnaryOperator)
        arg = _ident_or_const(statement.source.value.value.value)
        dest = _ident(statement.destination)

        return [_dsd_object(dsds, dest, output=True), _dsd_object(dsds, arg)]


class BinaryDSDOp(DSDOp):

    def _csl_op(self, a_dtype: spir.ScalarType, b_dtype: spir.ScalarType, dest_dtype: spir.ScalarType) -> str:
        raise NotImplementedError

    def _as_csl(self, statement: spir.AssignmentStatement, dtypes: dict[spir.Identifier, spir.IRType],
                dsds: UniqueDSDDict) -> str:
        assert isinstance(statement.source.value, spir.BinaryOperator)
        a = _ident_or_const(statement.source.value.left.value)
        b = _ident_or_const(statement.source.value.right.value)
        dest = _ident(statement.destination)

        a_dtype = _get_base_dtype(dtypes, a)
        b_dtype = _get_base_dtype(dtypes, b)
        dest_dtype = _get_base_dtype(dtypes, dest)
        if a_dtype == spir.ScalarType.UNKNOWN:
            a_dtype = b_dtype
        if b_dtype == spir.ScalarType.UNKNOWN:
            b_dtype = a_dtype

        op = self._csl_op(a_dtype, b_dtype, dest_dtype)
        return f"{op}({_dsd(dsds, dest, output=True)}, {_dsd(dsds, a)}, {_dsd(dsds, b)});"

    def used_dsd_objects(self, statement: spir.AssignmentStatement,
                         dsds: UniqueDSDDict) -> list[cslstruct.DataStructureDescriptor]:
        assert isinstance(statement.source.value, spir.BinaryOperator)
        a = _ident_or_const(statement.source.value.left.value)
        b = _ident_or_const(statement.source.value.right.value)
        dest = _ident(statement.destination)

        return [_dsd_object(dsds, dest, output=True), _dsd_object(dsds, a), _dsd_object(dsds, b)]


class NegDSDOp(UnaryDSDOp):

    def _as_csl(self, statement: spir.AssignmentStatement, dtypes: dict[spir.Identifier, spir.IRType],
                dsds: UniqueDSDDict) -> str:
        assert isinstance(statement.source.value, spir.UnaryOperator)
        arg = _ident_or_const(statement.source.value.value.value)
        dest = _ident(statement.destination)

        if _get_base_dtype(dtypes, arg) == spir.ScalarType.f32:
            return f'@fnegs({_dsd(dsds, dest, output=True)}, {_dsd(dsds, arg)});'
        return f'@fnegh({_dsd(dsds, dest, output=True)}, {_dsd(dsds, arg)});'


class AddDSDOp(BinaryDSDOp):

    def _csl_op(self, a_dtype: spir.ScalarType, b_dtype: spir.ScalarType, dest_dtype: spir.ScalarType) -> str:
        if a_dtype == spir.ScalarType.f16 and b_dtype == spir.ScalarType.f16:
            return '@faddh'
        elif a_dtype == spir.ScalarType.f32 and b_dtype == spir.ScalarType.f32:
            return '@fadds'
        elif a_dtype == spir.ScalarType.f16 and b_dtype == spir.ScalarType.f32:
            return '@faddhs'
        elif a_dtype == spir.ScalarType.f32 and b_dtype == spir.ScalarType.f16:
            return '@faddhs'
        else:
            return '@add16'


class SubDSDOp(BinaryDSDOp):

    def _csl_op(self, a_dtype: spir.ScalarType, b_dtype: spir.ScalarType, dest_dtype: spir.ScalarType) -> str:
        if a_dtype == spir.ScalarType.f16 and b_dtype == spir.ScalarType.f16:
            return '@fsubh'
        elif a_dtype == spir.ScalarType.f32 and b_dtype == spir.ScalarType.f32:
            return '@fsubs'
        else:
            return '@sub16'


class MulDSDOp(BinaryDSDOp):

    def _csl_op(self, a_dtype: spir.ScalarType, b_dtype: spir.ScalarType, dest_dtype: spir.ScalarType) -> str:
        if a_dtype == spir.ScalarType.f16 and b_dtype == spir.ScalarType.f16:
            return '@fmulh'
        elif a_dtype == spir.ScalarType.f32 and b_dtype == spir.ScalarType.f32:
            return '@fmuls'
        raise TypeError(f"Unsupported types for multiplication: {a_dtype}, {b_dtype}")


class FMADSDOp(DSDOp):

    def _as_csl(self, statement: spir.AssignmentStatement, dtypes: dict[spir.Identifier, spir.IRType],
                dsds: UniqueDSDDict) -> str:
        assert isinstance(statement.source.value, spir.MultiplyAccumulateOperator)
        a = _ident_or_const(statement.source.value.a)
        b = _ident_or_const(statement.source.value.b)
        c = _ident_or_const(statement.source.value.c)
        dest = _ident(statement.destination)
        a_dtype = _get_base_dtype(dtypes, a)
        b_dtype = _get_base_dtype(dtypes, b)
        c_dtype = _get_base_dtype(dtypes, c)
        if c_dtype == spir.ScalarType.UNKNOWN:
            c_dtype = a_dtype

        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f16 and c_dtype == spir.ScalarType.f16:
            return f'@fmach({_dsd(dsds, dest, output=True)}, {_dsd(dsds, a)}, {_dsd(dsds, b)}, {_dsd(dsds, c)});'
        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f32 and c_dtype == spir.ScalarType.f16:
            return f'@fmachs({_dsd(dsds, dest, output=True)}, {_dsd(dsds, a)}, {_dsd(dsds, b)}, {_dsd(dsds, c)});'  # 16-bit multiplication, 32-bit addition
        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f32 and c_dtype == spir.ScalarType.f32:
            return f'@fmacs({_dsd(dsds, dest, output=True)}, {_dsd(dsds, a)}, {_dsd(dsds, b)}, {_dsd(dsds, c)});'
        raise TypeError(f"Unsupported types for FMA: {a_dtype}, {b_dtype}, {c_dtype}")

    def used_dsd_objects(self, statement: spir.AssignmentStatement,
                         dsds: UniqueDSDDict) -> list[cslstruct.DataStructureDescriptor]:
        assert isinstance(statement.source.value, spir.MultiplyAccumulateOperator)
        a = _ident_or_const(statement.source.value.a)
        b = _ident_or_const(statement.source.value.b)
        c = _ident_or_const(statement.source.value.c)
        dest = _ident(statement.destination)

        return [_dsd_object(dsds, dest, output=True), _dsd_object(dsds, a), _dsd_object(dsds, b), _dsd_object(dsds, c)]


class CopyDSDOp(DSDOp):

    def __init__(self, scalar_input: bool = False):
        super().__init__()
        self.scalar_input = scalar_input

    def _as_csl(self, statement: spir.AssignmentStatement | spir.SendStatement,
                dtypes: dict[spir.Identifier, spir.IRType], dsds: UniqueDSDDict) -> str:
        if isinstance(statement, spir.SendStatement):
            src = _ident_or_const(statement.local_array)
            dest = _ident(statement.stream_name)
        else:
            assert isinstance(statement.source.value, (spir.ArraySlice, spir.Identifier, spir.ConstantLiteral))
            src = _ident_or_const(statement.source.value)
            dest = _ident(statement.destination)

        src_dtype = _get_base_dtype(dtypes, src)
        dtype = _get_base_dtype(dtypes, dest)
        if src_dtype == dtype or src_dtype == spir.ScalarType.UNKNOWN:
            if dtype in (spir.ScalarType.i16, spir.ScalarType.u16):
                op = '@mov16'
            elif dtype in (spir.ScalarType.i32, spir.ScalarType.u32):
                op = '@mov32'
            elif dtype == spir.ScalarType.f16:
                op = '@fmovh'
            elif dtype == spir.ScalarType.f32:
                op = '@fmovs'
            else:
                raise TypeError(f"Unsupported types for copy operation: {dtype}")
        else:
            if dtype == spir.ScalarType.f16 and src_dtype == spir.ScalarType.f32:
                op = '@fs2h'
            elif dtype == spir.ScalarType.f32 and src_dtype == spir.ScalarType.f16:
                op = '@fh2s'
            elif dtype == spir.ScalarType.f16 and src_dtype in (spir.ScalarType.i16, spir.ScalarType.u16):
                op = '@xp162fh'
            elif dtype == spir.ScalarType.f32 and src_dtype in (spir.ScalarType.i16, spir.ScalarType.u16):
                op = '@xp162fs'
            elif dtype in (spir.ScalarType.i16, spir.ScalarType.u16) and src_dtype == spir.ScalarType.f16:
                op = '@fh2xp16'
            elif dtype in (spir.ScalarType.i16, spir.ScalarType.u16) and src_dtype == spir.ScalarType.f32:
                op = '@fs2xp16'
            else:
                raise TypeError(f"Unsupported types for cast operation: {src_dtype}, {dtype}")

        if self.scalar_input:
            from spada.syntax.csl.statements import emit_expression
            if isinstance(statement, spir.SendStatement):
                # local_array may be an ArraySlice (e.g. a[k]) or a plain Identifier (e.g. x)
                src_expr = emit_expression(spir.Expression(statement.local_array), dsds, dtypes)
            else:
                src_expr = emit_expression(statement.source, dsds, dtypes)
            return f'{op}({_dsd(dsds, dest, output=True)}, {src_expr});'

        return f'{op}({_dsd(dsds, dest, output=True)}, {_dsd(dsds, src)});'

    def used_dsd_objects(self, statement: spir.AssignmentStatement,
                         dsds: UniqueDSDDict) -> list[cslstruct.DataStructureDescriptor]:
        if isinstance(statement, spir.SendStatement):
            src = _ident_or_const(statement.local_array)
            dest = _ident(statement.stream_name)
        else:
            assert isinstance(statement.source.value, (spir.ArraySlice, spir.Identifier, spir.ConstantLiteral))
            src = _ident_or_const(statement.source.value)
            dest = _ident(statement.destination)

        if self.scalar_input:
            return [_dsd_object(dsds, dest, output=True)]

        return [_dsd_object(dsds, dest, output=True), _dsd_object(dsds, src)]


DSD_ASSIGNMENT_MAPPING: dict[str, type[DSDOp]] = {
    # Unary operations
    '@fnegh': NegDSDOp,
    '@fnegs': NegDSDOp,
    # Binary operations
    '@faddh': AddDSDOp,
    '@fadds': AddDSDOp,
    '@faddhs': AddDSDOp,
    '@add16': AddDSDOp,
    '@fsubh': SubDSDOp,
    '@fsubs': SubDSDOp,
    '@sub16': SubDSDOp,
    '@fmulh': MulDSDOp,
    '@fmuls': MulDSDOp,
    # Fused multiply-add operations
    '@fmach': FMADSDOp,
    '@fmachs': FMADSDOp,
    '@fmacs': FMADSDOp,
    # Copy operations
    '@mov16': CopyDSDOp,
    '@mov32': CopyDSDOp,
    '@fmovh': CopyDSDOp,
    '@fmovs': CopyDSDOp,
    # Other operations
    '@fs2h': CopyDSDOp,
    '@fh2s': CopyDSDOp,
    '@xp162fh': CopyDSDOp,
    '@xp162fs': CopyDSDOp,
    '@fh2xp16': CopyDSDOp,
    '@fs2xp16': CopyDSDOp,

    # HACK: Skip activation if these statements are found in the code
    '@activate': None,
    '@unblock': None,
}


def _get_id(value: spir.ArraySlice | spir.Identifier) -> spir.Identifier:
    if isinstance(value, spir.Expression):
        return _get_id(value.value)
    if isinstance(value, spir.ArraySlice):
        return value.array
    return value


def _get_dtype(dtypes: dict[spir.Identifier, spir.IRType],
               value: spir.Identifier | spir.ArraySlice | spir.ConstantLiteral) -> spir.IRType:
    if isinstance(value, spir.Expression):
        return _get_dtype(dtypes, value.value)
    if isinstance(value, spir.ConstantLiteral):
        return value.dtype
    if isinstance(value, (spir.UnaryOperator, spir.BinaryOperator, spir.TernaryOperator)):
        return None
    return dtypes[_get_id(value)]


def _get_base_dtype(dtypes: dict[str, spir.IRType],
                    value: spir.Identifier | spir.ArraySlice | spir.ConstantLiteral) -> spir.ScalarType:
    dtype = _get_dtype(dtypes, value)
    if dtype is None:
        return dtype
    while not isinstance(dtype, spir.ScalarType):
        dtype = dtype.element_type
    return dtype


def _is_stream_backed_dtype(dtype: spir.IRType) -> bool:
    if isinstance(dtype, spir.StreamType):
        return True
    if isinstance(dtype, spir.ArrayType):
        return _is_stream_backed_dtype(dtype.base_type)
    return False


def _make_relay_dsd_statement(dtypes: dict[spir.Identifier, spir.IRType],
                              stmt: spir.ForeachStatement) -> Optional[spir.AssignmentStatement]:
    if len(stmt.body) != 2:
        return None

    first_stmt, second_stmt = stmt.body
    if not isinstance(first_stmt, spir.AssignmentStatement) or not isinstance(second_stmt, spir.SendStatement):
        return None

    receive_dtype = _get_dtype(dtypes, stmt.receive_stream.stream_name)
    if not _is_stream_backed_dtype(receive_dtype):
        return None

    if first_stmt.destination.as_ir() != second_stmt.local_array.as_ir():
        return None

    relay_stmt = spir.AssignmentStatement(copy.deepcopy(second_stmt.stream_name), copy.deepcopy(first_stmt.source))
    return relay_stmt


def get_dsd_statement(dtypes: dict[spir.Identifier, spir.IRType],
                      stmt: spir.Statement) -> Optional[spir.AssignmentStatement]:
    if isinstance(stmt, spir.AssignmentStatement):
        return stmt

    if not hasattr(stmt, 'body'):
        return None

    if len(stmt.body) == 0:
        return None
    if len(stmt.body) == 1 and isinstance(stmt.body[0], spir.AssignmentStatement):
        return stmt.body[0]
    if isinstance(stmt, spir.ForeachStatement):
        return _make_relay_dsd_statement(dtypes, stmt)
    return None


DISABLE_DSD = False


def get_dsd_op(dtypes: dict[spir.Identifier, spir.IRType],
               stmt: spir.ForeachStatement | spir.MapStatement | spir.AssignmentStatement) -> Optional[str]:
    """
    Returns a DSD op name if a foreach or map statement can be represented by a single DSD operation 
    (@mov, @fadd*, etc.), or None if the body cannot be expressed as a single DSD operation.
    This is used in lowering to CSL to determine whether a DSD operation can be used directly vs. creating
    a data task.
    """
    if DISABLE_DSD:
        return None
    if not isinstance(stmt, spir.AssignmentStatement) and len(stmt.body) == 0:
        # No-op
        return ''

    inner_stmt = get_dsd_statement(dtypes, stmt)
    if inner_stmt is None:
        return None

    dst = _get_id(inner_stmt.destination)
    if dst not in dtypes:
        raise NameError(f'"{dst.as_ir()}" not in recognized data types')
    dtype = _get_base_dtype(dtypes, dst)
    if not isinstance(dtypes[dst], (spir.ArrayType, spir.StreamType)):
        return None

    inner_stmt = inner_stmt.source.value

    if isinstance(inner_stmt, spir.UnaryOperator):  # @fneg*
        # NOTE: There is no negation DSD operation for integer types
        if dtype == spir.ScalarType.f16:
            return '@fnegh'
        if dtype == spir.ScalarType.f32:
            return '@fnegs'

    elif isinstance(inner_stmt, spir.BinaryOperator):
        # @add*, @fadd*, @fmul*, @sub*, @fsub*
        source_types = (_get_base_dtype(dtypes, inner_stmt.left), _get_base_dtype(dtypes, inner_stmt.right))
        if source_types[0] == spir.ScalarType.UNKNOWN:
            source_types = (source_types[1], source_types[1])
        if source_types[1] == spir.ScalarType.UNKNOWN:
            source_types = (source_types[0], source_types[0])

        if inner_stmt.op == '+':
            if (dtype == spir.ScalarType.f16 and source_types[0] == spir.ScalarType.f16 and
                    source_types[1] == spir.ScalarType.f16):
                return '@faddh'
            if (dtype == spir.ScalarType.f32 and source_types[0] == spir.ScalarType.f32 and
                    source_types[1] == spir.ScalarType.f32):
                return '@fadds'
            if (dtype == spir.ScalarType.f32 and
                ((source_types[0] == spir.ScalarType.f16 and source_types[1] == spir.ScalarType.f32) or
                 (source_types[0] == spir.ScalarType.f32 and source_types[1] == spir.ScalarType.f16))):
                return '@faddhs'
            if (dtype in (spir.ScalarType.i16, spir.ScalarType.u16) and
                    source_types[0] in (spir.ScalarType.i16, spir.ScalarType.u16) and
                    source_types[1] in (spir.ScalarType.i16, spir.ScalarType.u16)):
                return '@add16'

        elif inner_stmt.op == '-':
            if (dtype == spir.ScalarType.f16 and source_types[0] == spir.ScalarType.f16 and
                    source_types[1] == spir.ScalarType.f16):
                return '@fsubh'
            if (dtype == spir.ScalarType.f32 and source_types[0] == spir.ScalarType.f32 and
                    source_types[1] == spir.ScalarType.f32):
                return '@fsubs'
            if (dtype in (spir.ScalarType.i16, spir.ScalarType.u16) and
                    source_types[0] in (spir.ScalarType.i16, spir.ScalarType.u16) and
                    source_types[1] in (spir.ScalarType.i16, spir.ScalarType.u16)):
                return '@sub16'

        elif inner_stmt.op == '*':
            # NOTE: There is no @mul*
            if (dtype == spir.ScalarType.f16 and source_types[0] == spir.ScalarType.f16 and
                    source_types[1] == spir.ScalarType.f16):
                return '@fmulh'
            if (dtype == spir.ScalarType.f32 and source_types[0] == spir.ScalarType.f32 and
                    source_types[1] == spir.ScalarType.f32):
                return '@fmuls'

    elif isinstance(inner_stmt, spir.MultiplyAccumulateOperator):  # @fmac*
        # @fmac* only works with scalar/constant values of ``c``
        c_type = _get_dtype(dtypes, inner_stmt.c.value)
        if not isinstance(c_type, spir.ScalarType):
            return None
        a_dtype, b_dtype, c_dtype = (_get_base_dtype(dtypes, inner_stmt.a), _get_base_dtype(dtypes, inner_stmt.b),
                                     _get_base_dtype(dtypes, inner_stmt.c))
        if c_dtype == spir.ScalarType.UNKNOWN:
            c_dtype = a_dtype
        if dtype != a_dtype or dtype != b_dtype:
            # NOTE: Destination type semantics are unclear, supporting only same src/dst dtype for now
            return None
        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f16 and c_dtype == spir.ScalarType.f16:
            return '@fmach'
        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f32 and c_dtype == spir.ScalarType.f16:
            return '@fmachs'  # 16-bit multiplication, 32-bit addition
        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f32 and c_dtype == spir.ScalarType.f32:
            return '@fmacs'

    elif isinstance(inner_stmt, (spir.Identifier, spir.ConstantLiteral, spir.ArraySlice)):  # @fmov*, @mov*
        src_dtype = _get_base_dtype(dtypes, inner_stmt)
        # Move statements are valid for operands of the same type
        if src_dtype == dtype or src_dtype == spir.ScalarType.UNKNOWN:
            if dtype in (spir.ScalarType.i16, spir.ScalarType.u16):
                return '@mov16'
            if dtype in (spir.ScalarType.i32, spir.ScalarType.u32):
                return '@mov32'
            if dtype == spir.ScalarType.f16:
                return '@fmovh'
            if dtype == spir.ScalarType.f32:
                return '@fmovs'
        else:
            if dtype == spir.ScalarType.f16 and src_dtype == spir.ScalarType.f32:
                return '@fs2h'
            if dtype == spir.ScalarType.f32 and src_dtype == spir.ScalarType.f16:
                return '@fh2s'
            if dtype == spir.ScalarType.f16 and src_dtype in (spir.ScalarType.i16, spir.ScalarType.u16):
                return '@xp162fh'
            if dtype == spir.ScalarType.f32 and src_dtype in (spir.ScalarType.i16, spir.ScalarType.u16):
                return '@xp162fs'
            if dtype in (spir.ScalarType.i16, spir.ScalarType.u16) and src_dtype == spir.ScalarType.f16:
                return '@fh2xp16'
            if dtype in (spir.ScalarType.i16, spir.ScalarType.u16) and src_dtype == spir.ScalarType.f32:
                return '@fs2xp16'

    return None
