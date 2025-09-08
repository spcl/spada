from spatialstencil.syntax.csl.structures import DataStructureDescriptor
from spatialstencil.syntax.csl.tasks import get_dsd_op
from spatialstencil.syntax.spatial_ir import irnodes as spir
from spatialstencil.syntax.common.types import BIT_WIDTH


def generate_csl_statement(statement: spir.Statement, dsds: dict[spir.Identifier, DataStructureDescriptor],
                           dtypes: dict[spir.Identifier, spir.IRType]) -> str:
    """
    Generates a CSL statement from a Spatial IR statement.

    :param statement: The Spatial IR statement to convert.
    :return: The generated CSL statement.
    """
    if isinstance(statement, spir.ReceiveStatement):
        return emit_copy(statement.stream_name, statement.local_array, dsds, dtypes)
    elif isinstance(statement, spir.SendStatement):
        return emit_copy(statement.local_array, statement.stream_name, dsds, dtypes)
    elif isinstance(statement, spir.ForeachStatement):
        dsd_op = get_dsd_op(dtypes, statement)
        if dsd_op is not None:
            if isinstance(statement.receive_stream.stream_name, spir.ArraySlice):
                src = f'{statement.receive_stream.stream_name.array.as_ir()}_in_dsd'
            else:
                src = f'{statement.receive_stream.stream_name.as_ir()}_in_dsd'
            identifiers = [ident for ident in statement.body[0].walk() if isinstance(ident, spir.ArraySlice)]
            # Check if array slice matches foreach iterate
            # TODO(later): Multidimensional loops
            filtered_identifiers = [
                ident for ident in identifiers if ident.indices[0].value == statement.variables[0].identifier
            ]
            args = [f'{arg.array.as_ir()}_dsd' for arg in filtered_identifiers] + [src]
            return f'{dsd_op}({", ".join(args)});'
        else:
            raise ValueError('Operation was supposed to be lowered to a data task.\n'
                             f'  In line {statement.lineinfo}')
        # return emit_foreach(statement, dsds, dtypes)
    elif isinstance(statement, spir.MapStatement):
        pass
        # return emit_map(statement, dsds, dtypes)
    elif isinstance(statement, spir.ForStatement):
        pass
        # return emit_for(statement, dsds, dtypes)
    elif isinstance(statement, spir.AsyncBlock):
        # In the beginning, activate the next sequential-dependency task
        # In the end, unblock the completion waiters
        pass
        # return emit_async_block(statement, dsds, dtypes)
    elif isinstance(statement, spir.AssignmentStatement):
        return emit_assignment(statement, dsds, dtypes)
    return f'// TODO: Convert {statement} to CSL'


def emit_copy(source: spir.Identifier | spir.ArraySlice, destination: spir.Identifier | spir.ArraySlice,
              dsds: dict[spir.Identifier, DataStructureDescriptor], dtypes: dict[spir.Identifier, spir.IRType]) -> str:
    """
    Generates a CSL copy statement from source to destination.

    :param source: The source of the copy operation.
    :param destination: The destination of the copy operation.
    :param dsds: A dictionary of DSDs for the source and destination.
    :return: The generated CSL copy statement.
    """
    src_identifier: spir.Identifier
    dst_identifier: spir.Identifier
    if isinstance(source, spir.ArraySlice):
        src_identifier = source.array
    else:
        src_identifier = source

    if isinstance(destination, spir.ArraySlice):
        dst_identifier = destination.array
    else:
        dst_identifier = destination

    # One element copy
    if src_identifier.name not in dsds or dst_identifier.name not in dsds:
        if isinstance(dtypes[src_identifier], spir.ArrayType):
            src_expr = src_identifier.as_ir() + '[0]'
        else:
            src_expr = src_identifier.as_ir()
        if isinstance(dtypes[dst_identifier], spir.ArrayType):
            dst_expr = dst_identifier.as_ir() + '[0]'
        else:
            dst_expr = dst_identifier.as_ir()
        return f"{dst_expr} = {src_expr};"

    # Get the DSD operation for the copy operation
    if dtypes[src_identifier] != dtypes[dst_identifier]:
        raise ValueError(
            f"Source and destination types do not match: {dtypes[src_identifier]} != {dtypes[dst_identifier]}")
    if dtypes[src_identifier] in (spir.ScalarType.i16, spir.ScalarType.u16):
        op = '@mov16'
    elif dtypes[src_identifier] in (spir.ScalarType.i32, spir.ScalarType.u32):
        op = '@mov32'
    elif dtypes[src_identifier] == spir.ScalarType.f16:
        op = '@fmovh'
    elif dtypes[src_identifier] == spir.ScalarType.f32:
        op = '@fmovs'
    else:
        raise ValueError(f"Unsupported source type for copy operation: {dtypes[src_identifier]}")

    # If both source and destination are DSDs, use the copy operation
    return f"{op}({dsds[dst_identifier]}, {dsds[src_identifier]});"


# Dictionary mapping DSD operations to their corresponding argument conversion functions
# These functions are used to convert the source argument to the appropriate type for the DSD operation
_UNOP = lambda x: f"{x.value.as_ir()}"
_BINOP = lambda x: f"{x.left.as_ir()}, {x.right.as_ir()}"
_FMAOP = lambda x: f"{x.a.as_ir()}, {x.b.as_ir()}, {x.c.as_ir()}"
_DSD_ASSIGNMENT_MAPPING = {
    # Unary operations
    '@fnegh': _UNOP,
    '@fnegs': _UNOP,
    # Binary operations
    '@faddh': _BINOP,
    '@fadds': _BINOP,
    '@faddhs': _BINOP,
    '@add16': _BINOP,
    '@fsubh': _BINOP,
    '@fsubs': _BINOP,
    '@sub16': _BINOP,
    '@fmulh': _BINOP,
    '@fmuls': _BINOP,
    # Fused multiply-add operations
    '@fmach': _FMAOP,
    '@fmachs': _FMAOP,
    '@fmacs': _FMAOP,
    # Copy operations
    '@mov16': _UNOP,
    '@mov32': _UNOP,
    '@fmovh': _UNOP,
    '@fmovs': _UNOP,
    # Other operations
    '@fs2h': _UNOP,
    '@fh2s': _UNOP,
    '@xp162fh': _UNOP,
    '@xp162fs': _UNOP,
    '@fh2xp16': _UNOP,
    '@fs2xp16': _UNOP,
}


def emit_assignment(statement: spir.AssignmentStatement, dsds: dict[spir.Identifier, DataStructureDescriptor],
                    dtypes: dict[spir.Identifier, spir.IRType]) -> str:
    """
    Generates a CSL assignment statement from a Spatial IR assignment statement.

    :param statement: The Spatial IR assignment statement to convert.
    :return: The generated CSL assignment statement.
    """
    dst_identifier: spir.Identifier

    if isinstance(statement.destination, spir.ArraySlice):
        dst_identifier = statement.destination.array
    else:
        dst_identifier = statement.destination

    # One element assignment
    if dst_identifier.name not in dsds:
        if isinstance(dtypes[dst_identifier], spir.ArrayType):
            dst_expr = dst_identifier.as_ir() + '[0]'
        else:
            dst_expr = dst_identifier.as_ir()
        return f"{dst_expr} = {statement.source.as_ir()};"

    # DSD assignment
    dsd_op = get_dsd_op(dtypes, statement.source)
    if dsd_op is None:
        # TODO(later): Use a map / for loop?
        raise NotImplementedError(
            f"Assignment operation for type {dtypes[statement.source]} is not implemented as a DSD op.")
    return f"{dsd_op}({dsds[dst_identifier]}, {_DSD_ASSIGNMENT_MAPPING[dsd_op](statement.source)});"
