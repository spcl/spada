from io import StringIO
from typing import Optional
from spatialstencil.syntax.csl.structures import DataStructureDescriptor
from spatialstencil.syntax.csl import dsd_ops
from spatialstencil.syntax.spatial_ir import irnodes as spir

UniqueDSDDict = dict[str, list[tuple[str, DataStructureDescriptor]]]


def generate_csl_statement(statement: spir.Statement,
                           dsds: UniqueDSDDict,
                           dtypes: dict[spir.Identifier, spir.IRType],
                           async_target: Optional[dsd_ops.AsyncTarget],
                           header_code: StringIO,
                           in_foreach_or_map: bool = False) -> str:
    """
    Generates a CSL statement from a Spatial IR statement.

    :param statement: The Spatial IR statement to convert.
    :param dsds: A dictionary of data structure descriptors.
    :param dtypes: A dictionary of data types.
    :param async_target: The asynchronous target for the statement, or None if synchronous.
    :param header_code: The header code to include in the generated statement.
    :param in_foreach_or_map: Whether the statement is inside a foreach or map block.
    :return: The generated CSL statement.
    """
    op: str | dsd_ops.DSDOp | None = None
    if isinstance(statement, spir.ReceiveStatement):
        op = emit_copy(statement.stream_name, statement.local_array, dsds, dtypes, in_foreach_or_map)
    elif isinstance(statement, spir.SendStatement):
        op = emit_copy(statement.local_array, statement.stream_name, dsds, dtypes, in_foreach_or_map)
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


def emit_copy(source: spir.Identifier | spir.ArraySlice | spir.ConstantLiteral,
              destination: spir.Identifier | spir.ArraySlice,
              dsds: UniqueDSDDict,
              dtypes: dict[spir.Identifier, spir.IRType],
              in_foreach_or_map: bool = False) -> str | dsd_ops.CopyDSDOp:
    """
    Generates a CSL copy statement from source to destination.

    :param source: The source of the copy operation.
    :param destination: The destination of the copy operation.
    :param dsds: A dictionary of DSDs for the source and destination.
    :param dtypes: A dictionary of data types for the source and destination.
    :param in_foreach_or_map: Whether the copy is inside a foreach or map block.
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

    # Plain assignment: destination is not a DSD at all.
    # When destination IS a DSD but source is a plain scalar (Identifier not in dsds),
    # fall through to CopyDSDOp(scalar_input=True) below so the correct @fmovs call
    # is emitted instead of a direct variable assignment.
    src_is_plain_scalar = (isinstance(src_identifier, spir.Identifier) and
                           src_identifier.as_ir() not in dsds)
    if dst_identifier.as_ir() not in dsds:

        def _format_indexed_access(value: spir.Identifier | spir.ArraySlice, identifier: spir.Identifier) -> str:
            if isinstance(identifier, spir.TypedIdentifier):
                dtype = identifier.dtype
            else:
                dtype = dtypes.get(identifier)
            if isinstance(value, spir.ArraySlice):
                indices: list[str] = []
                for idx in value.indices:
                    if isinstance(idx, spir.Expression):
                        indices.append(emit_expression(idx, dsds, dtypes))
                    elif isinstance(idx, spir.RangeExpression):
                        indices.append(idx.as_ir())
                    else:
                        indices.append(str(idx))
                if not indices and isinstance(dtype, spir.ArrayType):
                    indices = ['0']
                if indices:
                    return f"{name_to_csl(identifier)}[{', '.join(indices)}]"
                return name_to_csl(identifier)

            if isinstance(dtype, spir.ArrayType):
                return f"{name_to_csl(identifier)}[0]"
            return name_to_csl(identifier)

        src_expr = _format_indexed_access(source, src_identifier)
        dst_expr = _format_indexed_access(destination, dst_identifier)
        return f"{dst_expr} = {src_expr};"

    # Get the DSD operation for the copy operation.
    # scalar_input is set when inside a foreach/map OR when the source is a plain
    # scalar variable (Identifier not in dsds) being written to a fabric DSD.
    dst_dtype = dtypes[dst_identifier]
    src_dtype = dtypes.get(src_identifier, dst_dtype)
    if isinstance(src_dtype, spir.ArrayType) and isinstance(src_dtype.base_type, spir.StreamType):
        if src_dtype.base_type.buffer_size is None:
            src_dtype = (src_dtype.base_type.element_type, [])
        else:
            src_dtype = (src_dtype.base_type.element_type, [src_dtype.base_type.buffer_size.eval()])
    elif isinstance(src_dtype, spir.ScalarType):
        src_dtype = (src_dtype, [])
    else:
        src_dtype = (src_dtype.element_type, [s.eval() for s in src_dtype.shape])

    if isinstance(dst_dtype, spir.ArrayType) and isinstance(dst_dtype.base_type, spir.StreamType):
        if dst_dtype.base_type.buffer_size is None:
            dst_dtype = (dst_dtype.base_type.element_type, [])
        else:
            dst_dtype = (dst_dtype.base_type.element_type, [dst_dtype.base_type.buffer_size.eval()])
    elif isinstance(dst_dtype, spir.ScalarType):
        dst_dtype = (dst_dtype, [])
    else:
        dst_dtype = (dst_dtype.element_type, [s.eval() for s in dst_dtype.shape])

    if src_dtype[0] != dst_dtype[0]:
        raise ValueError(
            f"Source and destination types do not match: {dtypes[src_identifier]} != {dtypes[dst_identifier]}")

    return dsd_ops.CopyDSDOp(scalar_input=in_foreach_or_map or src_is_plain_scalar)


def emit_expression(expr: spir.Expression,
                    dsds: UniqueDSDDict,
                    dtypes: dict[spir.Identifier, spir.IRType],
                    other: Optional[spir.Expression] = None) -> str:
    """
    Generates a CSL expression from a Spatial IR expression.

    :param expr: The Spatial IR expression to convert.
    :return: The generated CSL expression.
    """
    val = expr.value
    if isinstance(val, spir.BinaryOperator):
        return f"({emit_expression(val.left, dsds, dtypes, other=val.right)} {val.op} {emit_expression(val.right, dsds, dtypes, other=val.left)})"
    elif isinstance(val, spir.UnaryOperator):
        return f"({val.op}{emit_expression(val.value, dsds, dtypes)})"
    elif isinstance(val, spir.TernaryOperator):
        return f"(if ({emit_expression(val.cond, dsds, dtypes)}) {emit_expression(val.if_true, dsds, dtypes)} else {emit_expression(val.if_false, dsds, dtypes)})"
    elif isinstance(val, spir.MultiplyAccumulateOperator):
        return f"({emit_expression(val.a, dsds, dtypes)} + {emit_expression(val.b, dsds, dtypes)} * {emit_expression(val.c, dsds, dtypes)})"
    elif isinstance(val, spir.Identifier):
        return name_to_csl(val)
    elif isinstance(val, spir.ConstantLiteral):
        if val.dtype == spir.ScalarType.UNKNOWN:  # Type is not given, need to perform light type inference
            # TODO(later): Move to separate constant type inference pass
            if other is not None:
                other_dtype = dtypes.get(other.value, None) if isinstance(other.value, spir.Identifier) else None
                if other_dtype is not None:
                    while not isinstance(other_dtype, spir.ScalarType):
                        other_dtype = other_dtype.element_type
                    val = spir.ConstantLiteral(value=val.value, dtype=other_dtype)
        if val.dtype in (spir.ScalarType.f16, spir.ScalarType.f32, spir.ScalarType.f64):
            return str(float(val.value))
        return str(val.value)
    elif isinstance(val, spir.ArraySlice):
        return f"{name_to_csl(val.array)}[{', '.join(map(lambda x: emit_expression(x, dsds, dtypes), val.indices))}]"
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
        indices = [name_to_csl(idx) if isinstance(idx, spir.Identifier) else str(idx) for idx in indices]
    else:
        dst_identifier = statement.destination
        indices = [0]

    # One element assignment
    if isinstance(statement.destination, spir.ArraySlice) or dst_identifier.as_ir() not in dsds:
        if isinstance(dtypes[dst_identifier], spir.ArrayType):
            dst_expr = name_to_csl(dst_identifier) + f'[{", ".join(map(str, indices))}]'
        else:
            dst_expr = name_to_csl(dst_identifier)
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


DISABLE_MAPS = False


def _is_map_compatible(statement: spir.MapStatement, dsds: UniqueDSDDict, dtypes: dict[spir.Identifier,
                                                                                       spir.IRType]) -> bool:
    """
    Checks if a Spatial IR map statement is compatible with CSL ``@map`` semantics.

    :param statement: The Spatial IR map statement to check.
    :param dsds: The unique DSD dictionary.
    :param dtypes: The data types dictionary.
    :return: True if the statement is compatible, False otherwise.
    """
    if DISABLE_MAPS:
        return False

    def collect_dsd_writes_from_body(body: list[spir.Statement], in_loop: bool = False) -> dict[str, list[bool]]:
        """
        Collects all DSD writes from the body and tracks whether they occur in loops.
        Returns a dict mapping DSD names to lists of booleans indicating if writes are in loops.
        """
        dsd_writes = {}

        for stmt in body:
            if isinstance(stmt, spir.AssignmentStatement):
                # Check if the destination is a DSD
                dest_identifier = None
                if isinstance(stmt.destination, spir.ArraySlice):
                    dest_identifier = stmt.destination.array
                elif isinstance(stmt.destination, spir.Identifier):
                    dest_identifier = stmt.destination

                if dest_identifier and dest_identifier.as_ir() in dsds:
                    if dest_identifier.as_ir() not in dsd_writes:
                        dsd_writes[dest_identifier.as_ir()] = []
                    dsd_writes[dest_identifier.as_ir()].append(in_loop)

            elif isinstance(stmt, spir.ForStatement):
                # Recursively check for loop body with in_loop=True
                loop_writes = collect_dsd_writes_from_body(stmt.body, in_loop=True)
                for dsd_name, writes in loop_writes.items():
                    if dsd_name not in dsd_writes:
                        dsd_writes[dsd_name] = []
                    dsd_writes[dsd_name].extend(writes)

            elif isinstance(stmt, spir.ForeachStatement):
                # Recursively check foreach body with in_loop=True
                loop_writes = collect_dsd_writes_from_body(stmt.body, in_loop=True)
                for dsd_name, writes in loop_writes.items():
                    if dsd_name not in dsd_writes:
                        dsd_writes[dsd_name] = []
                    dsd_writes[dsd_name].extend(writes)

            elif isinstance(stmt, spir.MapStatement):
                # Recursively check nested map body
                nested_writes = collect_dsd_writes_from_body(stmt.body, in_loop)
                for dsd_name, writes in nested_writes.items():
                    if dsd_name not in dsd_writes:
                        dsd_writes[dsd_name] = []
                    dsd_writes[dsd_name].extend(writes)

        return dsd_writes

    # Collect all DSD variables referenced in the map statement variables and body
    referenced_dsds = set()

    # Check variables (these could be inputs)
    for var in statement.variables:
        if var.identifier.as_ir() in dsds:
            referenced_dsds.add(var.identifier.as_ir())

    # Collect writes from body to find outputs
    dsd_writes = collect_dsd_writes_from_body(statement.body)

    # Add written DSDs to referenced DSDs
    referenced_dsds.update(dsd_writes.keys())

    # At least one DSD must be referenced (input or output)
    if not referenced_dsds:
        return False

    # Check output constraints: at most one DSD can be written to, and it must be written exactly once outside loops
    output_dsds = []
    for dsd_name, writes in dsd_writes.items():
        # Count non-loop writes
        non_loop_writes = [w for w in writes if not w]
        loop_writes = [w for w in writes if w]

        # For CSL @map compatibility:
        # - There should be exactly one write outside of loops
        # - Loop writes are allowed for accumulation, but there must be one final write outside loops
        if len(non_loop_writes) == 1:
            output_dsds.append(dsd_name)
        elif len(non_loop_writes) > 1:
            # Multiple writes outside loops - not compatible
            return False
        elif len(non_loop_writes) == 0 and len(loop_writes) > 0:
            # Only loop writes, no final write outside - not compatible for @map
            return False

    # At most one output DSD is allowed
    if len(output_dsds) > 1:
        return False

    # Check that index variables are not used within body (except for DSD array slices)
    class _FindNonArrayIndexVars(spir.NodeVisitor):

        def __init__(self):
            super().__init__()
            self.non_array_index_vars = set()

        def visit_ArraySlice(self, node: spir.ArraySlice):
            # Do not track array slices
            return

        def visit_Identifier(self, node: spir.Identifier):
            self.non_array_index_vars.add(node)
            return self.generic_visit(node)

    finder = _FindNonArrayIndexVars()
    for substmt in statement.body:
        finder.visit(substmt)
    if finder.non_array_index_vars & set(v.identifier for v in statement.variables):
        return False

    return True


class MapArgumentCollector(spir.NodeVisitor):

    def __init__(self, index_identifiers: set[spir.Identifier], dsds: UniqueDSDDict):
        super().__init__()
        self.used_identifiers: list[spir.Identifier] = []
        self.output_variables: list[spir.Identifier] = []
        self.index_identifiers = index_identifiers
        self.dsds = dsds

    def visit_Identifier(self, node: spir.Identifier):
        if node not in self.used_identifiers and node not in self.index_identifiers:
            self.used_identifiers.append(node)
        return self.generic_visit(node)

    def visit_AssignmentStatement(self, node: spir.AssignmentStatement):
        self.generic_visit(node.source)

        # Do not visit lhs recursively
        if isinstance(node.destination, spir.ArraySlice):
            dest_identifier = node.destination.array
        elif isinstance(node.destination, spir.Identifier):
            dest_identifier = node.destination
        else:
            dest_identifier = None
        if dest_identifier and dest_identifier.as_ir() in self.dsds:
            self.output_variables.append(dest_identifier)

        return node


def emit_map(statement: spir.MapStatement, dsds: UniqueDSDDict, dtypes: dict[spir.Identifier, spir.IRType],
             header_code: StringIO) -> str:
    """
    Generates a CSL map statement from a Spatial IR map statement.

    :param statement: The Spatial IR map statement to convert.
    :return: The generated CSL map statement.
    """

    # The semantics of a CSL @map statement are as follows:
    # 1. A callback function is called with one or more DSD arguments. Other arguments may be variables.
    # 2. The last argument to the @map statement is the output DSD (if any)
    # 3. The @map statement may only be used with DSDs and variables of compatible types.
    # 4. The callback function does not have a notion of an index
    # Therefore, the following restrictions on a Spatial IR map statement apply:
    # 1. At least one DSD must be referenced (input or output)
    # 2. At most one DSD can be written to, and it must be written exactly once outside loops
    # 3. Index variables must not be used within the body (except for DSD array slices)
    # In other cases, the Spatial IR map becomes a CSL for loop

    global_code = ""  # Code that will appear outside the existing task (for the map callback)
    local_code = ""

    header_code.write(f"\n// Map from {statement.lineinfo}\n")

    # Generate a unique callback function name
    callback_name = f"map_callback_{id(statement)}"

    # Check if compatible with the ``@map`` semantics (see above). If not, emit for loop instead
    if not _is_map_compatible(statement, dsds, dtypes):
        return emit_for(statement, dsds, dtypes, header_code)

    # Collect all variables from expressions recursively
    index_identifiers = set(v.identifier for v in statement.variables)
    mac = MapArgumentCollector(index_identifiers, dsds)
    for substmt in statement.body:
        mac.visit(substmt)

    used_identifiers: list[spir.Identifier] = mac.used_identifiers
    output_variables: list[spir.Identifier] = mac.output_variables

    assert len(output_variables) <= 1, "At most one output DSD is allowed in a CSL @map statement."

    # Create a mapping from input variables to parameter names for substitution
    var_to_param = {}
    input_params = []
    param_counter = 0

    # Add parameters for input variables (excluding loop variables, they come from @map iteration)
    for input_var in used_identifiers:
        # if input_var in output_variables:
        #     continue
        # Get the type from the variable
        var_dtype = dtypes[input_var]
        if isinstance(var_dtype, spir.ArrayType):
            element_type = var_dtype.element_type
        else:
            element_type = var_dtype
        if isinstance(element_type, spir.StreamType):
            element_type = element_type.element_type

        param_type = dtype_as_csl(element_type)
        param_name = f"arg{param_counter}"
        input_params.append(f"{param_name}: {param_type}")
        var_to_param[input_var] = param_name
        param_counter += 1

    # Check if there is a return value (output)
    has_output = len(output_variables) > 0
    return_type = "void"
    if has_output:
        # Use the first output variable's element type as return type
        output_var = output_variables[0]
        output_dtype = dtypes[output_var]
        if isinstance(output_dtype, spir.ArrayType):
            element_type = output_dtype.element_type
        else:
            element_type = output_dtype
        if isinstance(element_type, spir.StreamType):
            element_type = element_type.element_type
        return_type = dtype_as_csl(element_type)

    # Transform the map body: convert assignment to return statement, array slices to identifiers
    class _ParameterSubstitutionTransformer(spir.NodeTransformer):

        def visit_Identifier(self, node: spir.Identifier):
            if node in var_to_param:
                # Create a new identifier with the parameter name
                param_name = var_to_param[node]
                new_node = spir.Identifier(name=param_name, version=0)
                new_node.lineinfo = node.lineinfo
                return new_node
            return self.generic_visit(node)

        def visit_ArraySlice(self, node: spir.ArraySlice):
            if node.array in var_to_param:
                # Create a new array slice with the parameter name
                param_name = var_to_param[node.array]
                new_node = spir.Identifier(name=param_name, version=0)
                new_node.lineinfo = node.lineinfo
                return new_node
            return self.generic_visit(node)

        def visit_AssignmentStatement(self, node: spir.AssignmentStatement):
            if not output_variables:
                return self.generic_visit(node)
            if not isinstance(node.destination, spir.ArraySlice) or node.destination.array != output_variables[0]:
                return self.generic_visit(node)

            # This is an assignment to output - convert to return statement
            # Transform the source expression with parameter substitution
            transformed_source = self.visit(node.source)
            return f"return {emit_expression(transformed_source, dsds, dtypes)};"

    transformer = _ParameterSubstitutionTransformer()
    transformed_body = []

    for stmt in statement.body:
        transformed_stmt = transformer.visit(stmt)
        transformed_body.append(transformed_stmt)

    # Generate the callback function signature
    global_code = f"fn {callback_name}({', '.join(input_params)}) {return_type} {{\n"

    for stmt in transformed_body:
        if isinstance(stmt, str):
            # Direct CSL code (e.g., return statement)
            global_code += f"    {stmt}\n"
        else:
            # Spatial IR statement - convert to CSL
            sub_op = generate_csl_statement(stmt, dsds, dtypes, None, header_code)
            sub_lines = sub_op.splitlines()
            for line in sub_lines:
                global_code += f"    {line}\n"

    global_code += "}\n"

    # Generate the @map call arguments
    map_args = [callback_name]

    # Add input arguments (DSDs for arrays, variables for scalars)
    for input_var in used_identifiers:
        # if input_var in output_variables:
        #     continue
        var_key = input_var.as_ir()
        if var_key in dsds:
            # Use DSD for arrays
            dsd_list = dsds[var_key]
            if dsd_list:
                map_args.append(dsd_list[0][0])  # Use the first DSD name
        else:
            # Use variable name for scalars
            map_args.append(name_to_csl(input_var))

    # Add output DSD if present
    if has_output:
        output_var = output_variables[0]
        var_key = output_var.as_ir()
        if var_key in dsds:
            dsd_list = dsds[var_key]
            if dsd_list:
                map_args.append(dsd_list[0][0])  # Use the first DSD name
        else:
            map_args.append(name_to_csl(output_var))

    # Write the callback function to the header
    header_code.write(global_code)

    # Generate the @map call
    local_code = f"@map({', '.join(map_args)});"

    return local_code


def name_to_csl(name: spir.Identifier) -> str:
    """
    Returns a CSL syntactic equivalent to a Spatial IR identifier.

    :param name: Spatial IR identifier.
    :return: Compilable CSL string representing the identifier.
    """
    if isinstance(name, spir.TypedIdentifier):
        return f'var {name_to_csl(name.identifier)}: {dtype_as_csl(name.dtype)}'
    if isinstance(name, spir.ConstantLiteral):
        return str(name.value)
    if name.version == 0:
        return name.name
    else:
        return f'{name.name}__{name.version}'


def expr_to_csl(expr: spir.Expression) -> str:
    """
    Returns a CSL syntactic equivalent to a Spatial IR expression.

    :param expr: Spatial IR expression.
    :return: Compilable CSL string representing the expression.
    """
    expr = expr.value
    if isinstance(expr, spir.Identifier):
        return name_to_csl(expr)
    elif isinstance(expr, spir.ConstantLiteral):
        return str(expr.value)
    elif isinstance(expr, spir.BinaryOperator):
        left = expr_to_csl(expr.left)
        right = expr_to_csl(expr.right)
        return f"({left} {expr.op} {right})"
    elif isinstance(expr, spir.UnaryOperator):
        operand = expr_to_csl(expr.value)
        return f"({expr.op}{operand})"
    elif isinstance(expr, spir.TernaryOperator):
        condition = expr_to_csl(expr.condition)
        true_case = expr_to_csl(expr.true_case)
        false_case = expr_to_csl(expr.false_case)
        return f"(if ({condition}) {true_case} else {false_case})"
    elif isinstance(expr, spir.MultiplyAccumulateOperator):
        return f"({expr_to_csl(expr.a)} + {expr_to_csl(expr.b)} * {expr_to_csl(expr.c)})"
    else:
        raise NotImplementedError(f"Unsupported expression type: {type(expr)}")


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
