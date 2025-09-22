from io import StringIO
from typing import Optional
from spatialstencil.syntax.csl.structures import DataStructureDescriptor
from spatialstencil.syntax.csl import dsd_ops
from spatialstencil.syntax.spatial_ir import irnodes as spir
from spatialstencil.syntax.common.types import BIT_WIDTH

UniqueDSDDict = dict[str, list[tuple[str, DataStructureDescriptor]]]


def generate_csl_statement(statement: spir.Statement, dsds: UniqueDSDDict, dtypes: dict[spir.Identifier, spir.IRType],
                           async_target: Optional[dsd_ops.AsyncTarget], header_code: StringIO) -> str:
    """
    Generates a CSL statement from a Spatial IR statement.

    :param statement: The Spatial IR statement to convert.
    :param dsds: A dictionary of data structure descriptors.
    :param dtypes: A dictionary of data types.
    :param async_target: The asynchronous target for the statement, or None if synchronous.
    :param header_code: The header code to include in the generated statement.
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
            op = emit_map(statement, dsds, dtypes, header_code)
    elif isinstance(statement, spir.ForStatement):
        op = emit_for(statement, dsds, dtypes, header_code)
    elif isinstance(statement, spir.AsyncBlock):
        # In the beginning, activate the next sequential-dependency task
        # In the end, unblock the completion waiters
        op = emit_async_block(statement, dsds, dtypes, async_target, header_code)
    elif isinstance(statement, spir.AssignmentStatement):
        op = emit_assignment(statement, dsds, dtypes)
    elif isinstance(statement, (spir.AwaitCompletionStatement, spir.AwaitAllStatement)):
        # Skip (taken care of when tasks are defined)
        return ""

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
    if src_identifier.as_ir() not in dsds or dst_identifier.as_ir() not in dsds:
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


def emit_expression(expr: spir.Expression, dsds: UniqueDSDDict, dtypes: dict[spir.Identifier, spir.IRType]) -> str:
    """
    Generates a CSL expression from a Spatial IR expression.

    :param expr: The Spatial IR expression to convert.
    :return: The generated CSL expression.
    """
    val = expr.value
    if isinstance(val, spir.BinaryOperator):
        return f"({emit_expression(val.left, dsds, dtypes)} {val.op} {emit_expression(val.right, dsds, dtypes)})"
    elif isinstance(val, spir.UnaryOperator):
        return f"({val.op}{emit_expression(val.value, dsds, dtypes)})"
    elif isinstance(val, spir.TernaryOperator):
        return f"(if ({emit_expression(val.cond, dsds, dtypes)}) {emit_expression(val.if_true, dsds, dtypes)} else {emit_expression(val.if_false, dsds, dtypes)})"
    elif isinstance(val, spir.MultiplyAccumulateOperator):
        return f"({emit_expression(val.a, dsds, dtypes)} + {emit_expression(val.b, dsds, dtypes)} * {emit_expression(val.c, dsds, dtypes)})"
    elif isinstance(val, spir.Identifier):
        return name_to_csl(val)
    elif isinstance(val, spir.ConstantLiteral):
        return str(val.value)
    elif isinstance(val, spir.ArraySlice):
        return f"{name_to_csl(val.array)}[{', '.join(map(str, val.indices))}]"
    else:
        raise NotImplementedError(f"Expression type {type(val)} is not implemented.")


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
        return f"{dst_expr} = {emit_expression(statement.source, dsds, dtypes)};"

    # DSD assignment
    dsd_op = dsd_ops.get_dsd_op(dtypes, statement)
    if dsd_op is None:
        # TODO(later): Use a map / for loop?
        raise NotImplementedError(f"Assignment operation for {statement.source.as_ir()} is not implemented as a DSD op."
                                  f"\n  In line {statement.lineinfo}")
    return dsd_ops.DSD_ASSIGNMENT_MAPPING[dsd_op]()


def emit_for(statement: spir.ForStatement, dsds: UniqueDSDDict, dtypes: dict[spir.Identifier, spir.IRType],
             header_code: StringIO) -> str:
    """
    Generates a CSL for loop statement from a Spatial IR for loop statement.

    :param statement: The Spatial IR for loop statement to convert.
    :param dsds: The unique DSD dictionary.
    :param dtypes: The data types dictionary.
    :param header_code: The header code to include.
    :return: The generated CSL for loop statement.
    """
    ranges = statement.range_expression
    vars_ = statement.variables

    result = ""
    # Open nested loops
    for depth, (rng, var) in enumerate(zip(ranges, vars_)):
        # Expect rng to have start, end and optionally step
        start = rng.start.eval() if rng.start is not None else 0
        end = rng.stop.eval() if rng.stop is not None else 0
        step = rng.step.eval() if rng.step is not None else 1

        var_name = name_to_csl(var.identifier)
        var_type = dtype_as_csl(var.dtype)
        indent = "    " * depth
        result += f"{indent}for (@range({var_type}, {start}, {end}, {step})) |{var_name}| {{\n"

    # Body (indent one level deeper than the deepest loop)
    body_indent = "    " * len(ranges)
    for stmt in statement.body:
        sub_op = generate_csl_statement(stmt, dsds, dtypes, None, header_code)
        # If the generated sub_op already contains newlines, indent each line
        sub_lines = str(sub_op).splitlines()
        for line in sub_lines:
            result += f"{body_indent}{line}\n"

    # Close nested loops
    for depth in range(len(ranges) - 1, -1, -1):
        indent = "    " * depth
        result += f"{indent}}}\n"

    return result


def emit_async_block(statement: spir.AsyncBlock, dsds: UniqueDSDDict, dtypes: dict[spir.Identifier, spir.IRType],
                     async_target: Optional[dsd_ops.AsyncTarget], header_code: StringIO) -> str:
    """
    Generates a CSL async block statement from a Spatial IR async block statement.

    :param statement: The Spatial IR async block statement to convert.
    :return: The generated CSL async block statement.
    """
    result = "{\n"

    # If async target exists, activate it first
    if async_target:
        result += f"    @{async_target.inter_task_edge}({async_target.target_task});\n"

    # Generate the rest of the body
    for stmt in statement.body:
        # The async block body is not asynchronous
        sub_op = generate_csl_statement(stmt, dsds, dtypes, None, header_code)
        # If the generated sub_op already contains newlines, indent each line
        sub_lines = sub_op.splitlines()
        for line in sub_lines:
            result += f"    {line}\n"

    result += "}\n"
    return result


def name_to_csl(name: spir.Identifier) -> str:
    """
    Returns a CSL syntactic equivalent to a Spatial IR identifier.

    :param name: Spatial IR identifier.
    :return: Compilable CSL string representing the identifier.
    """
    if name.version == 0:
        return name.name
    else:
        return f'{name.name}__{name.version}'


def dtype_as_csl(dtype: spir.ScalarType | spir.StreamType | spir.ArrayType, export: bool = False) -> str:
    """
    Returns a CSL syntactic equivalent to a Spatial IR data type.

    :param dtype: Spatial IR data type.
    :param export: If True, the type is exported as a symbol.
    :return: CSL string representing the given data type.
    """
    if isinstance(dtype, spir.ScalarType):
        return dtype.as_ir()
    if isinstance(dtype, spir.StreamType):
        return dtype.element_type.as_ir()
    if isinstance(dtype, spir.ArrayType):
        if export:
            shape = '[*]'
        else:
            shape = f'[{", ".join(str(s) if isinstance(s, int) else s.as_ir() for s in dtype.shape)}]' if len(
                dtype.shape) > 0 else ''
        return shape + dtype_as_csl(dtype.base_type, export=export)
