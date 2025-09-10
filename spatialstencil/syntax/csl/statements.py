from typing import Optional
from spatialstencil.syntax.csl.structures import DataStructureDescriptor
from spatialstencil.syntax.csl import dsd_ops
from spatialstencil.syntax.spatial_ir import irnodes as spir
from spatialstencil.syntax.common.types import BIT_WIDTH

UniqueDSDDict = dict[str, list[tuple[str, DataStructureDescriptor]]]


def generate_csl_statement(statement: spir.Statement, dsds: UniqueDSDDict, dtypes: dict[spir.Identifier,
                                                                                        spir.IRType]) -> str:
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
              dsds: UniqueDSDDict, dtypes: dict[spir.Identifier, spir.IRType]) -> str:
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
    src_dtype = dtypes[src_identifier]
    dst_dtype = dtypes[dst_identifier]
    if isinstance(src_dtype, spir.ArrayType) and isinstance(src_dtype.base_type, spir.StreamType):
        if src_dtype.base_type.buffer_size is None:
            src_dtype = (src_dtype.base_type.element_type, [])
        else:
            src_dtype = (src_dtype.base_type.element_type, [src_dtype.base_type.buffer_size.eval()])
    else:
        src_dtype = (src_dtype.element_type, [s.eval() for s in src_dtype.shape])

    if isinstance(dst_dtype, spir.ArrayType) and isinstance(dst_dtype.base_type, spir.StreamType):
        if dst_dtype.base_type.buffer_size is None:
            dst_dtype = (dst_dtype.base_type.element_type, [])
        else:
            dst_dtype = (dst_dtype.base_type.element_type, [dst_dtype.base_type.buffer_size.eval()])
    else:
        dst_dtype = (dst_dtype.element_type, [s.eval() for s in dst_dtype.shape])

    if src_dtype[0] != dst_dtype[0]:
        raise ValueError(
            f"Source and destination types do not match: {dtypes[src_identifier]} != {dtypes[dst_identifier]}")
    if src_dtype[0] in (spir.ScalarType.i16, spir.ScalarType.u16):
        op = '@mov16'
    elif src_dtype[0] in (spir.ScalarType.i32, spir.ScalarType.u32):
        op = '@mov32'
    elif src_dtype[0] == spir.ScalarType.f16:
        op = '@fmovh'
    elif src_dtype[0] == spir.ScalarType.f32:
        op = '@fmovs'
    else:
        raise ValueError(f"Unsupported source type for copy operation: {src_dtype}")

    # If both source and destination are DSDs, use the copy operation
    return f"{op}({dsds[dst_identifier.as_ir()][0][0]}, {dsds[src_identifier.as_ir()][0][0]});"


def emit_assignment(statement: spir.AssignmentStatement, dsds: UniqueDSDDict, dtypes: dict[spir.Identifier,
                                                                                           spir.IRType]) -> str:
    """
    Generates a CSL assignment statement from a Spatial IR assignment statement.

    :param statement: The Spatial IR assignment statement to convert.
    :return: The generated CSL assignment statement.
    """
    dst_identifier: spir.Identifier

    if isinstance(statement.destination, spir.ArraySlice):
        dst_identifier = statement.destination.array
        indices = [idx.eval() for idx in statement.destination.indices]
    else:
        dst_identifier = statement.destination
        indices = [0]

    # One element assignment
    if isinstance(statement.destination, spir.ArraySlice) or dst_identifier.name not in dsds:
        if isinstance(dtypes[dst_identifier], spir.ArrayType):
            dst_expr = dst_identifier.as_ir() + f'[{", ".join(map(str, indices))}]'
        else:
            dst_expr = dst_identifier.as_ir()
        return f"{dst_expr} = {statement.source.as_ir()};"

    # DSD assignment
    dsd_op = dsd_ops.get_dsd_op(dtypes, statement)
    if dsd_op is None:
        # TODO(later): Use a map / for loop?
        raise NotImplementedError(f"Assignment operation for {statement.source.as_ir()} is not implemented as a DSD op."
                                  f"\n  In line {statement.lineinfo}")
    return f"{dsd_op}({dsds[dst_identifier.as_ir()][0][0]}, {DSD_ASSIGNMENT_MAPPING[dsd_op](dsds, statement.source.value)});"
