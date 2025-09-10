from typing import Optional
from spatialstencil.syntax.csl.structures import DataStructureDescriptor
from spatialstencil.syntax.csl import dsd_ops
from spatialstencil.syntax.spatial_ir import irnodes as spir
from spatialstencil.syntax.common.types import BIT_WIDTH

UniqueDSDDict = dict[str, list[tuple[str, DataStructureDescriptor]]]


def generate_csl_statement(statement: spir.Statement, dsds: UniqueDSDDict, dtypes: dict[spir.Identifier, spir.IRType],
                           async_target: Optional[dsd_ops.AsyncTarget]) -> str:
    """
    Generates a CSL statement from a Spatial IR statement.

    :param statement: The Spatial IR statement to convert.
    :return: The generated CSL statement.
    """
    op: str | dsd_ops.DSDOp | None = None
    if isinstance(statement, spir.ReceiveStatement):
        op = emit_copy(statement.stream_name, statement.local_array, dsds, dtypes)
    elif isinstance(statement, spir.SendStatement):
        op = emit_copy(statement.local_array, statement.stream_name, dsds, dtypes)
    elif isinstance(statement, spir.ForeachStatement):
        op = _try_emit_dsd_op(statement, dsds, dtypes)
        if op is None:
            raise ValueError('Operation was supposed to be lowered to a data task.\n'
                             f'  In line {statement.lineinfo}')
    elif isinstance(statement, spir.MapStatement):
        op = _try_emit_dsd_op(statement, dsds, dtypes)
        if op is None:
            op = emit_map(statement, dsds, dtypes)
    elif isinstance(statement, spir.ForStatement):
        pass
        # op = emit_for(statement, dsds, dtypes)
    elif isinstance(statement, spir.AsyncBlock):
        # In the beginning, activate the next sequential-dependency task
        # In the end, unblock the completion waiters
        # op = emit_async_block(statement, dsds, dtypes)
        pass
    elif isinstance(statement, spir.AssignmentStatement):
        op = emit_assignment(statement, dsds, dtypes)

    if op is None:
        return f'// TODO: Convert {statement} to CSL'

    return op.as_csl(statement, dtypes, dsds, async_target) if isinstance(op, dsd_ops.DSDOp) else op


def _try_emit_dsd_op(statement: spir.MapStatement | spir.ForeachStatement, dsds: UniqueDSDDict,
                     dtypes: dict[spir.Identifier, spir.IRType]) -> dsd_ops.DSDOp | None:
    """
    Tries to emit a DSD operation for the given statement, or return None if not applicable.

    :param statement: The Spatial IR statement to convert.
    :param dsds: A dictionary of DSDs for the statement.
    :param dtypes: A dictionary of data types for the statement.
    :return: The generated DSD operation or None if not applicable.
    """
    dsd_op = dsd_ops.get_dsd_op(dtypes, statement)
    if dsd_op is None:
        return None

    dsd_op = dsd_ops.DSD_ASSIGNMENT_MAPPING[dsd_op]()
    return dsd_op


def emit_copy(source: spir.Identifier | spir.ArraySlice, destination: spir.Identifier | spir.ArraySlice,
              dsds: UniqueDSDDict, dtypes: dict[spir.Identifier, spir.IRType]) -> str | dsd_ops.CopyDSDOp:
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

    # If both source and destination are DSDs, use the copy operation
    return dsd_ops.CopyDSDOp()


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
    return dsd_ops.DSD_ASSIGNMENT_MAPPING[dsd_op]()
