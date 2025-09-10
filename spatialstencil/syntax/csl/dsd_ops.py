"""
Parses DSD operations from IR nodes.
"""
import copy
from dataclasses import dataclass
from typing import Literal, Optional
from spatialstencil.syntax.spatial_ir import irnodes as spir
from spatialstencil.syntax.csl import structures as cslstruct

UniqueDSDDict = dict[str, list[tuple[str, cslstruct.DataStructureDescriptor]]]


@dataclass
class AsyncTarget:
    target_task: str
    inter_task_edge: Literal["activate", "unblock"]


class DSDOp:
    """ Class representing a DSD operation that can be lowered to CSL. """

    def _append_async_suffix(self, base: str, async_target: Optional[AsyncTarget]) -> str:
        if async_target is None:
            return base
        return f'{base[:-2]}, .{{ .async = true, .{async_target.inter_task_edge} = {async_target.target_task} }});'

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
            statement = statement.body[0]
        elif hasattr(statement, 'body'):
            statement = statement.body[0]
        return self._append_async_suffix(self._as_csl(statement, dtypes, dsds), async_target)

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


def _ident(expr: spir.Identifier | spir.ArraySlice | spir.TypedIdentifier) -> spir.Identifier:
    if isinstance(expr, spir.Identifier):
        return expr
    elif isinstance(expr, spir.ArraySlice):
        return expr.array
    elif isinstance(expr, spir.TypedIdentifier):
        return expr.identifier
    raise TypeError(f"Unsupported expression type: {type(expr)}")


def _ident_or_const(expr: spir.SpatialNode) -> spir.Identifier | spir.ConstantLiteral:
    if isinstance(expr, spir.ConstantLiteral):
        return expr
    else:
        return _ident(expr)


def _dsd(dsds: UniqueDSDDict, expr: spir.SpatialNode) -> str:
    if isinstance(expr, spir.Identifier):
        return dsds[expr.as_ir()][0][0]
    elif isinstance(expr, spir.ConstantLiteral):
        return str(expr.value)
    raise TypeError(f"Unsupported expression type: {type(expr)}")


class UnaryDSDOp(DSDOp):
    pass


class BinaryDSDOp(DSDOp):
    pass


class NegDSDOp(UnaryDSDOp):

    def _as_csl(self, statement: spir.AssignmentStatement, dtypes: dict[spir.Identifier, spir.IRType],
                dsds: UniqueDSDDict) -> str:
        assert isinstance(statement.source.value, spir.UnaryOperator)
        arg = _ident_or_const(statement.source.value.value.value)
        dest = _ident(statement.destination)

        if _get_base_dtype(dtypes, arg) == spir.ScalarType.f32:
            return f'@fnegs({_dsd(dsds, dest)}, {_dsd(dsds, arg)});'
        return f'@fnegh({_dsd(dsds, dest)}, {_dsd(dsds, arg)});'


class AddDSDOp(BinaryDSDOp):

    def _as_csl(self, statement: spir.AssignmentStatement, dtypes: dict[spir.Identifier, spir.IRType],
                dsds: UniqueDSDDict) -> str:
        assert isinstance(statement.source.value, spir.BinaryOperator)
        a = _ident_or_const(statement.source.value.left.value)
        b = _ident_or_const(statement.source.value.right.value)
        dest = _ident(statement.destination)

        if _get_base_dtype(dtypes, a) == spir.ScalarType.f16 and _get_base_dtype(dtypes, b) == spir.ScalarType.f16:
            return f'@faddh({_dsd(dsds, dest)}, {_dsd(dsds, a)}, {_dsd(dsds, b)});'
        elif _get_base_dtype(dtypes, a) == spir.ScalarType.f32 and _get_base_dtype(dtypes, b) == spir.ScalarType.f32:
            return f'@fadds({_dsd(dsds, dest)}, {_dsd(dsds, a)}, {_dsd(dsds, b)});'
        elif _get_base_dtype(dtypes, a) == spir.ScalarType.f16 and _get_base_dtype(dtypes, b) == spir.ScalarType.f32:
            return f'@faddhs({_dsd(dsds, dest)}, {_dsd(dsds, a)}, {_dsd(dsds, b)});'
        elif _get_base_dtype(dtypes, a) == spir.ScalarType.f32 and _get_base_dtype(dtypes, b) == spir.ScalarType.f16:
            return f'@faddhs({_dsd(dsds, dest)}, {_dsd(dsds, a)}, {_dsd(dsds, b)});'
        else:
            return f'@add16({_dsd(dsds, dest)}, {_dsd(dsds, a)}, {_dsd(dsds, b)});'


class SubDSDOp(BinaryDSDOp):

    def _as_csl(self, statement: spir.AssignmentStatement, dtypes: dict[spir.Identifier, spir.IRType],
                dsds: UniqueDSDDict) -> str:
        assert isinstance(statement.source.value, spir.BinaryOperator)
        a = _ident_or_const(statement.source.value.left.value)
        b = _ident_or_const(statement.source.value.right.value)
        dest = _ident(statement.destination)

        if _get_base_dtype(dtypes, a) == spir.ScalarType.f16 and _get_base_dtype(dtypes, b) == spir.ScalarType.f16:
            return f'@fsubh({_dsd(dsds, dest)}, {_dsd(dsds, a)}, {_dsd(dsds, b)});'
        elif _get_base_dtype(dtypes, a) == spir.ScalarType.f32 and _get_base_dtype(dtypes, b) == spir.ScalarType.f32:
            return f'@fsubs({_dsd(dsds, dest)}, {_dsd(dsds, a)}, {_dsd(dsds, b)});'
        else:
            return f'@sub16({_dsd(dsds, dest)}, {_dsd(dsds, a)}, {_dsd(dsds, b)});'


class MulDSDOp(BinaryDSDOp):

    def _as_csl(self, statement: spir.AssignmentStatement, dtypes: dict[spir.Identifier, spir.IRType],
                dsds: UniqueDSDDict) -> str:
        assert isinstance(statement.source.value, spir.BinaryOperator)
        a = _ident_or_const(statement.source.value.left.value)
        b = _ident_or_const(statement.source.value.right.value)
        dest = _ident(statement.destination)

        if _get_base_dtype(dtypes, a) == spir.ScalarType.f16 and _get_base_dtype(dtypes, b) == spir.ScalarType.f16:
            return f'@fmulh({_dsd(dsds, dest)}, {_dsd(dsds, a)}, {_dsd(dsds, b)});'
        elif _get_base_dtype(dtypes, a) == spir.ScalarType.f32 and _get_base_dtype(dtypes, b) == spir.ScalarType.f32:
            return f'@fmuls({_dsd(dsds, dest)}, {_dsd(dsds, a)}, {_dsd(dsds, b)});'
        raise TypeError(
            f"Unsupported types for multiplication: {_get_base_dtype(dtypes, a)}, {_get_base_dtype(dtypes, b)}")


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
        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f16 and c_dtype == spir.ScalarType.f16:
            return f'@fmach({_dsd(dsds, dest)}, {_dsd(dsds, a)}, {_dsd(dsds, b)}, {_dsd(dsds, c)});'
        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f32 and c_dtype == spir.ScalarType.f16:
            return f'@fmachs({_dsd(dsds, dest)}, {_dsd(dsds, a)}, {_dsd(dsds, b)}, {_dsd(dsds, c)});'  # 16-bit multiplication, 32-bit addition
        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f32 and c_dtype == spir.ScalarType.f32:
            return f'@fmacs({_dsd(dsds, dest)}, {_dsd(dsds, a)}, {_dsd(dsds, b)}, {_dsd(dsds, c)});'
        raise TypeError(f"Unsupported types for FMA: {a_dtype}, {b_dtype}, {c_dtype}")


class CopyDSDOp(DSDOp):

    def _as_csl(self, statement: spir.AssignmentStatement | spir.SendStatement,
                dtypes: dict[spir.Identifier, spir.IRType], dsds: UniqueDSDDict) -> str:
        if isinstance(statement, spir.SendStatement):
            src = _ident(statement.local_array)
            dest = _ident(statement.stream_name)
        else:
            assert isinstance(statement.source.value, (spir.ArraySlice, spir.Identifier, spir.ConstantLiteral))
            src = _ident_or_const(statement.source.value)
            dest = _ident(statement.destination)

        src_dtype = _get_base_dtype(dtypes, src)
        dtype = _get_base_dtype(dtypes, dest)
        if src_dtype == dtype:
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
        return f'{op}({_dsd(dsds, dest)}, {_dsd(dsds, src)});'


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


def get_dsd_op(dtypes: dict[spir.Identifier, spir.IRType],
               stmt: spir.ForeachStatement | spir.MapStatement | spir.AssignmentStatement) -> Optional[str]:
    """
    Returns a DSD op name if a foreach or map statement can be represented by a single DSD operation 
    (@mov, @fadd*, etc.), or None if the body cannot be expressed as a single DSD operation.
    This is used in lowering to CSL to determine whether a DSD operation can be used directly vs. creating
    a data task.
    """
    if isinstance(stmt, spir.AssignmentStatement):
        inner_stmt = stmt
    else:
        if len(stmt.body) == 0:
            # No-op
            return ''
        if len(stmt.body) > 1:
            return None
        inner_stmt = stmt.body[0]
        if not isinstance(inner_stmt, spir.AssignmentStatement):
            return None

    dst = _get_id(inner_stmt.destination)
    if dst not in dtypes:
        raise NameError(f'"{dst.as_ir()}" not in recognized data types')
    dtype = _get_base_dtype(dtypes, dst)

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
        if dtype != a_dtype or dtype != b_dtype:
            # NOTE: Destination type semantics are unclear, supporting only same src/dst dtype for now
            return None
        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f16 and c_dtype == spir.ScalarType.f16:
            return '@fmach'
        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f32 and c_dtype == spir.ScalarType.f16:
            return '@fmachs'  # 16-bit multiplication, 32-bit addition
        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f32 and c_dtype == spir.ScalarType.f32:
            return '@fmacs'

    elif isinstance(inner_stmt, (spir.Identifier, spir.ConstantLiteral)):  # @fmov*, @mov*
        src_dtype = _get_base_dtype(dtypes, inner_stmt)
        # Move statements are valid for operands of the same type
        if src_dtype == dtype:
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
