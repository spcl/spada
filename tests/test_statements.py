import pytest
from typing import List
from spatialstencil.syntax.csl.statements import generate_csl_statement, emit_copy, emit_assignment
from spatialstencil.syntax.csl.structures import DataStructureDescriptor, MemoryDSD, FabricDSD, DSDType
from spatialstencil.syntax.spatial_ir import irnodes as spir
from spatialstencil.syntax.common.types import ScalarType


def create_test_identifier(name: str, version: int = 0) -> spir.Identifier:
    """Helper function to create test identifiers."""
    return spir.Identifier(name=name, version=version)


def create_test_stream_type(element_type: ScalarType = ScalarType.f32) -> spir.StreamType:
    """Helper function to create test stream types."""
    return spir.StreamType(element_type=element_type)


def create_test_array_type(base_type: ScalarType = ScalarType.f32, shape: list = None) -> spir.ArrayType:
    """Helper function to create test array types."""
    if shape is None:
        shape = [32]
    return spir.ArrayType(base_type=base_type, shape=shape)


def create_test_expression(value) -> spir.Expression:
    """Helper function to create test expressions."""
    return spir.Expression(value=value)


def create_test_literal(value: float, dtype: ScalarType = ScalarType.f32) -> spir.ConstantLiteral:
    """Helper function to create test literals."""
    return spir.ConstantLiteral(value=value, dtype=dtype)


def create_test_range(start: int, stop: int, step: int = None) -> spir.RangeExpression:
    """Helper function to create test ranges."""
    start_expr = create_test_expression(create_test_literal(start, ScalarType.i16))
    stop_expr = create_test_expression(create_test_literal(stop, ScalarType.i16))
    step_expr = create_test_expression(create_test_literal(step, ScalarType.i16)) if step else None
    return spir.RangeExpression(start=start_expr, stop=stop_expr, step=step_expr)


def create_test_typed_identifier(name: str, dtype) -> spir.TypedIdentifier:
    """Helper function to create typed identifiers."""
    return spir.TypedIdentifier(dtype=dtype, identifier=create_test_identifier(name))


class TestReceiveStatement:
    """Test cases for ReceiveStatement."""

    def test_receive_statement_scalar_without_dsd(self):
        """Test receive statement with scalar types (no DSD)."""
        # Setup
        local_var = create_test_identifier("local_var")
        stream_var = create_test_identifier("stream_var")

        statement = spir.ReceiveStatement(local_array=local_var, stream_name=stream_var)

        dsds = {}  # Empty DSDs to trigger scalar copy
        dtypes = {local_var: ScalarType.f32, stream_var: create_test_stream_type(ScalarType.f32)}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result == "local_var = stream_var;"

    def test_receive_statement_array_without_dsd(self):
        """Test receive statement with array types (no DSD)."""
        # Setup
        local_var = create_test_identifier("local_array")
        stream_var = create_test_identifier("stream_in")

        statement = spir.ReceiveStatement(local_array=local_var, stream_name=stream_var)

        dsds = {}  # Empty DSDs to trigger scalar copy
        dtypes = {
            local_var: create_test_array_type(ScalarType.f32, [1]),
            stream_var: create_test_stream_type(ScalarType.f32)
        }

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result == "local_array[0] = stream_in;"

    def test_receive_statement_with_dsd(self):
        """Test receive statement with DSDs."""
        # Setup
        local_var = create_test_identifier("local_array")
        stream_var = create_test_identifier("stream_in")

        statement = spir.ReceiveStatement(local_array=local_var, stream_name=stream_var)

        local_dsd = MemoryDSD(
            dsd_type=DSDType.mem1d, array="local_array", extent=["32"], idxvars=["i"], expression=["i"])
        stream_dsd = FabricDSD(dsd_type=DSDType.fabin, color="color1", extent=32)
        dsds = {local_var: local_dsd, stream_var: stream_dsd}
        dtypes = {local_var: ScalarType.f32, stream_var: ScalarType.f32}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result == "@fmovs(" + local_dsd.as_csl() + ", " + stream_dsd.as_csl() + ");"


class TestSendStatement:
    """Test cases for SendStatement."""

    def test_send_statement_scalar_without_dsd(self):
        """Test send statement with scalar types (no DSD)."""
        # Setup
        local_var = create_test_identifier("local_val")
        stream_var = create_test_identifier("output_stream")

        statement = spir.SendStatement(local_array=local_var, stream_name=stream_var)

        dsds = {}  # Empty DSDs to trigger scalar copy
        dtypes = {local_var: ScalarType.f32, stream_var: create_test_stream_type(ScalarType.f32)}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result == "output_stream = local_val;"

    def test_send_statement_with_dsd_i16(self):
        """Test send statement with i16 DSDs."""
        # Setup
        local_var = create_test_identifier("local_array")
        stream_var = create_test_identifier("output_stream")

        statement = spir.SendStatement(local_array=local_var, stream_name=stream_var)

        local_dsd = MemoryDSD(
            dsd_type=DSDType.mem1d, array="local_array", extent=["16"], idxvars=["i"], expression=["i"])
        stream_dsd = FabricDSD(dsd_type=DSDType.fabout, color="color2", extent=16)
        dsds = {local_var: local_dsd, stream_var: stream_dsd}
        dtypes = {local_var: ScalarType.i16, stream_var: ScalarType.i16}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result == "@mov16(" + stream_dsd.as_csl() + ", " + local_dsd.as_csl() + ");"


class TestForeachStatement:
    """Test cases for ForeachStatement (two variants)."""

    def test_foreach_statement_with_parameter_range(self):
        """Test foreach statement with parameter range (variant 1)."""
        # Setup
        i_var = create_test_typed_identifier("i", ScalarType.u16)
        stream_var = create_test_typed_identifier("data", ScalarType.f32)

        range_expr = create_test_range(0, 32)
        receive_gen = spir.ReceiveGenerator(stream_name=create_test_identifier("input_stream"))

        body_stmt = spir.AssignmentStatement(
            destination=create_test_identifier("result"), source=create_test_expression(create_test_literal(1.0)))

        statement = spir.ForeachStatement(
            variables=[i_var],
            parameter_range=[range_expr],
            stream_variable=stream_var,
            receive_stream=receive_gen,
            body=[body_stmt])

        dsds = {}
        dtypes = {}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result.startswith("// TODO: Convert")  # Since emit_foreach is not implemented

    def test_foreach_statement_without_parameter_range(self):
        """Test foreach statement without parameter range (variant 2)."""
        # Setup
        stream_var = create_test_typed_identifier("data", ScalarType.f32)

        receive_gen = spir.ReceiveGenerator(stream_name=create_test_identifier("input_stream"))

        body_stmt = spir.AssignmentStatement(
            destination=create_test_identifier("result"), source=create_test_expression(create_test_literal(2.0)))

        statement = spir.ForeachStatement(
            variables=[], parameter_range=[], stream_variable=stream_var, receive_stream=receive_gen, body=[body_stmt])

        dsds = {}
        dtypes = {}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result.startswith("// TODO: Convert")  # Since emit_foreach is not implemented


class TestMapStatement:
    """Test cases for MapStatement."""

    def test_map_statement(self):
        """Test basic map statement."""
        # Setup
        i_var = create_test_typed_identifier("i", ScalarType.u16)
        j_var = create_test_typed_identifier("j", ScalarType.u16)

        range_i = create_test_range(0, 16)
        range_j = create_test_range(0, 16)

        body_stmt = spir.AssignmentStatement(
            destination=create_test_identifier("matrix"), source=create_test_expression(create_test_literal(0.0)))

        statement = spir.MapStatement(variables=[i_var, j_var], range_expression=[range_i, range_j], body=[body_stmt])

        dsds = {}
        dtypes = {}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result.startswith("// TODO: Convert")  # Since emit_map is not implemented


class TestForStatement:
    """Test cases for ForStatement."""

    def test_for_statement(self):
        """Test basic for statement."""
        # Setup
        i_var = create_test_typed_identifier("i", ScalarType.u16)
        range_expr = create_test_range(0, 10)

        body_stmt = spir.AssignmentStatement(
            destination=create_test_identifier("counter"), source=create_test_expression(create_test_literal(1.0)))

        statement = spir.ForStatement(variables=[i_var], range_expression=[range_expr], body=[body_stmt])

        dsds = {}
        dtypes = {}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result.startswith("// TODO: Convert")  # Since emit_for is not implemented


class TestAsyncBlock:
    """Test cases for AsyncBlock."""

    def test_async_block_structure(self):
        """Test async block for proper CSL task structure."""
        # Setup
        completion = spir.Completion(name=create_test_identifier("task_completion"))

        body_stmt = spir.AssignmentStatement(
            destination=create_test_identifier("async_result"),
            source=create_test_expression(create_test_literal(42.0)))

        statement = spir.AsyncBlock(completion_name=completion, body=[body_stmt])

        dsds = {}
        dtypes = {}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify - should contain task structure when implemented
        assert result.startswith("// TODO: Convert")  # Currently returns TODO
        # When implemented, verify it contains proper async task activation/completion


class TestAssignmentStatement:
    """Test cases for AssignmentStatement with various expression types."""

    def test_assignment_scalar_without_dsd(self):
        """Test scalar assignment without DSD."""
        # Setup
        dest = create_test_identifier("result")
        source_literal = create_test_literal(5.0)
        source = create_test_expression(source_literal)

        statement = spir.AssignmentStatement(destination=dest, source=source)

        dsds = {}  # No DSDs
        dtypes = {dest: ScalarType.f32}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result == "result = 5.0;"

    def test_assignment_array_without_dsd(self):
        """Test array assignment without DSD (controls DSD vs non-DSD by array size > 1)."""
        # Setup
        dest = create_test_identifier("result_array")
        source_literal = create_test_literal(3.14)
        source = create_test_expression(source_literal)

        statement = spir.AssignmentStatement(destination=dest, source=source)

        dsds = {}  # No DSDs
        dtypes = {dest: create_test_array_type(ScalarType.f32, [64])}  # Array size > 1

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result == "result_array[0] = 3.14;"

    def test_assignment_binary_expression(self):
        """Test assignment with binary expression (nested expressions)."""
        # Setup
        dest = create_test_identifier("sum")

        # Create a + b expression
        a_literal = create_test_literal(10.0)
        b_literal = create_test_literal(20.0)
        a_expr = create_test_expression(a_literal)
        b_expr = create_test_expression(b_literal)

        binary_op = spir.BinaryOperator(left=a_expr, op='+', right=b_expr)
        source = create_test_expression(binary_op)

        statement = spir.AssignmentStatement(destination=dest, source=source)

        dsds = {}
        dtypes = {dest: ScalarType.f32}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result == "sum = (10.0 + 20.0);"

    def test_assignment_ternary_expression(self):
        """Test assignment with ternary operator."""
        # Setup
        dest = create_test_identifier("conditional_result")

        # Create condition ? true_val : false_val
        condition = create_test_expression(create_test_literal(1))  # True condition
        true_val = create_test_expression(create_test_literal(100.0))
        false_val = create_test_expression(create_test_literal(200.0))

        ternary_op = spir.TernaryOperator(cond=condition, if_true=true_val, if_false=false_val)
        source = create_test_expression(ternary_op)

        statement = spir.AssignmentStatement(destination=dest, source=source)

        dsds = {}
        dtypes = {dest: ScalarType.f32}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result == "conditional_result = (100.0 if 1 else 200.0);"

    def test_assignment_fused_multiply_accumulate(self):
        """Test assignment with fused multiply-accumulate expression."""
        # Setup
        dest = create_test_identifier("fma_result")

        # Create fmac(a, b, c) = a + b * c
        a_expr = create_test_expression(create_test_literal(1.0))
        b_expr = create_test_expression(create_test_literal(2.0))
        c_expr = create_test_expression(create_test_literal(3.0))

        fma_op = spir.MultiplyAccumulateOperator(a=a_expr, b=b_expr, c=c_expr)
        source = create_test_expression(fma_op)

        statement = spir.AssignmentStatement(destination=dest, source=source)

        dsds = {}
        dtypes = {dest: ScalarType.f32}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result == "fma_result = fmac(1.0, 2.0, 3.0);"

    def test_assignment_nested_complex_expression(self):
        """Test assignment with multiple nested expressions."""
        # Setup
        dest = create_test_identifier("complex_result")

        # Create ((a + b) * c) where a, b, c are literals
        a_literal = create_test_literal(5.0)
        b_literal = create_test_literal(7.0)
        c_literal = create_test_literal(2.0)

        a_expr = create_test_expression(a_literal)
        b_expr = create_test_expression(b_literal)
        c_expr = create_test_expression(c_literal)

        # First: a + b
        add_op = spir.BinaryOperator(left=a_expr, op='+', right=b_expr)
        add_expr = create_test_expression(add_op)

        # Then: (a + b) * c
        mul_op = spir.BinaryOperator(left=add_expr, op='*', right=c_expr)
        source = create_test_expression(mul_op)

        statement = spir.AssignmentStatement(destination=dest, source=source)

        dsds = {}
        dtypes = {dest: ScalarType.f32}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result == "complex_result = ((5.0 + 7.0) * 2.0);"

    def test_assignment_with_dsd_binary_operation(self):
        """Test assignment with DSD and binary operation - falls back to scalar."""
        # Setup
        dest = create_test_identifier("dsd_result")

        # Create a + b expression for DSD operation
        a_literal = create_test_literal(1.0)
        b_literal = create_test_literal(2.0)
        a_expr = create_test_expression(a_literal)
        b_expr = create_test_expression(b_literal)

        binary_op = spir.BinaryOperator(left=a_expr, op='+', right=b_expr)
        source = create_test_expression(binary_op)

        statement = spir.AssignmentStatement(destination=dest, source=source)

        # Setup DSDs
        dest_dsd = MemoryDSD(dsd_type=DSDType.mem1d, array="dsd_result", extent=["32"], idxvars=["i"], expression=["i"])
        dsds = {dest: dest_dsd}
        dtypes = {dest: ScalarType.f32}

        # Execute - DSD assignment for complex expressions falls back to scalar assignment
        result = generate_csl_statement(statement, dsds, dtypes)
        # Since get_dsd_op returns None, it falls back to scalar assignment
        assert result == "dsd_result = (1.0 + 2.0);"

    def test_assignment_with_dsd_fma_operation(self):
        """Test assignment with DSD and fused multiply-accumulate - falls back to scalar."""
        # Setup
        dest = create_test_identifier("dsd_fma")

        # Create fmac operation
        a_expr = create_test_expression(create_test_literal(1.0))
        b_expr = create_test_expression(create_test_literal(2.0))
        c_expr = create_test_expression(create_test_literal(3.0))

        fma_op = spir.MultiplyAccumulateOperator(a=a_expr, b=b_expr, c=c_expr)
        source = create_test_expression(fma_op)

        statement = spir.AssignmentStatement(destination=dest, source=source)

        # Setup DSDs
        dest_dsd = MemoryDSD(dsd_type=DSDType.mem1d, array="dsd_fma", extent=["32"], idxvars=["i"], expression=["i"])
        dsds = {dest: dest_dsd}
        dtypes = {dest: ScalarType.f32}

        # Execute - DSD assignment for complex expressions falls back to scalar assignment
        result = generate_csl_statement(statement, dsds, dtypes)
        # Since get_dsd_op returns None, it falls back to scalar assignment
        assert result == "dsd_fma = fmac(1.0, 2.0, 3.0);"


class TestErrorHandling:
    """Test cases for error conditions."""

    def test_unsupported_statement_type(self):
        """Test handling of unsupported statement types."""

        # Create a mock statement that doesn't match any known types
        class MockStatement(spir.Statement):

            def as_ir(self, indent=0):
                return "mock_statement"

        statement = MockStatement()
        dsds = {}
        dtypes = {}

        # Execute
        result = generate_csl_statement(statement, dsds, dtypes)

        # Verify
        assert result.startswith("// TODO: Convert")

    def test_type_mismatch_in_copy(self):
        """Test type mismatch error in copy operations."""
        # Setup
        src = create_test_identifier("source")
        dst = create_test_identifier("destination")

        # Create DSDs using string keys (as the implementation expects)
        src_dsd = MemoryDSD(dsd_type=DSDType.mem1d, array="source", extent=["32"], idxvars=["i"], expression=["i"])
        dst_dsd = MemoryDSD(dsd_type=DSDType.mem1d, array="destination", extent=["32"], idxvars=["i"], expression=["i"])
        dsds = {"source": src_dsd, "destination": dst_dsd}  # Using string keys

        # Different types should cause error
        dtypes = {src: ScalarType.f32, dst: ScalarType.i16}

        # Execute and verify error
        with pytest.raises(ValueError, match="Source and destination types do not match"):
            emit_copy(src, dst, dsds, dtypes)


if __name__ == '__main__':
    pytest.main([__file__])
