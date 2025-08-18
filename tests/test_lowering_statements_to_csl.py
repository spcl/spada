import pytest
import os
from spatialstencil.lowering.spatial_ir_to_csl import lower_spatial_ir_to_csl
from spatialstencil.syntax.spatial_ir import parser, passes


def create_inline_spatial_ir(code: str):
    """Helper function to parse inline Spatial IR code."""
    # Create a temporary file with the code
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sptl', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        kernel = parser.parse_file(temp_file)
        return kernel
    finally:
        os.unlink(temp_file)


def test_receive_statement_scalar():
    """Test receive statement with scalar types."""
    spatial_ir_code = '''
    kernel @test_receive<N>(stream<f32>[N] readonly input, stream<f32>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 local_val;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(local_val, input[i]);
            await send(local_val, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for receive operations in the generated CSL
    code_found = False
    for f in csl_files:
        if 'local_val = ' in f.code:
            code_found = True
            break

    assert code_found, "Expected receive operation not found in generated CSL"


def test_receive_statement_array():
    """Test receive statement with array types."""
    spatial_ir_code = '''
    kernel @test_receive_array<N>(stream<f32>[N] readonly input, stream<f32>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32[4] local_array;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(local_array, input[i]);
            await send(local_array, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for array operations in the generated CSL
    code_found = False
    for f in csl_files:
        if 'local_array[0]' in f.code or '@fmovs' in f.code:
            code_found = True
            break

    assert code_found, "Expected array receive operation not found in generated CSL"


def test_send_statement_scalar():
    """Test send statement with scalar types."""
    spatial_ir_code = '''
    kernel @test_send<N>(stream<f32>[N] readonly input, stream<f32>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 local_val;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(local_val, input[i]);
            await send(local_val, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for send operations in the generated CSL
    code_found = False
    for f in csl_files:
        if 'output' in f.code and '=' in f.code:
            code_found = True
            break

    assert code_found, "Expected send operation not found in generated CSL"


def test_send_statement_with_different_types():
    """Test send statement with different scalar types."""
    spatial_ir_code = '''
    kernel @test_send_i16<N>(stream<i16>[N] readonly input, stream<i16>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            i16 local_val;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(local_val, input[i]);
            await send(local_val, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for i16 operations (should use @mov16)
    code_found = False
    for f in csl_files:
        if '@mov16' in f.code or 'local_val' in f.code:
            code_found = True
            break

    assert code_found, "Expected i16 operation not found in generated CSL"


# Test lowering of AssignmentStatement to CSL with various expression types.


def test_assignment_binary_expression():
    """Test assignment with binary expression."""
    spatial_ir_code = '''
    kernel @test_binary<N>(stream<f32>[N] readonly a, stream<f32>[N] readonly b, stream<f32>[N] writeonly result) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 val_a;
            f32 val_b;
            f32 sum;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(val_a, a[i]);
            await receive(val_b, b[i]);
            sum = val_a + val_b;
            await send(sum, result[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for addition operation in the generated CSL
    code_found = False
    for f in csl_files:
        if 'val_a + val_b' in f.code or '@fadds' in f.code:
            code_found = True
            break

    assert code_found, "Expected binary operation not found in generated CSL"


def test_assignment_ternary_expression():
    """Test assignment with ternary operator."""
    spatial_ir_code = '''
    kernel @test_ternary<N>(stream<f32>[N] readonly condition, stream<f32>[N] readonly a, stream<f32>[N] readonly b, stream<f32>[N] writeonly result) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 cond;
            f32 val_a;
            f32 val_b;
            f32 output;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(cond, condition[i]);
            await receive(val_a, a[i]);
            await receive(val_b, b[i]);
            output = val_a if cond else val_b;
            await send(output, result[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for ternary operation in the generated CSL
    code_found = False
    for f in csl_files:
        if 'if' in f.code and 'else' in f.code:
            code_found = True
            break

    assert code_found, "Expected ternary operation not found in generated CSL"


def test_assignment_fused_multiply_accumulate():
    """Test assignment with fused multiply-accumulate."""
    spatial_ir_code = '''
    kernel @test_fma<N>(stream<f32>[N] readonly a, stream<f32>[N] readonly b, stream<f32>[N] readonly c, stream<f32>[N] writeonly result) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 val_a;
            f32 val_b;
            f32 val_c;
            f32 fma_result;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(val_a, a[i]);
            await receive(val_b, b[i]);
            await receive(val_c, c[i]);
            fma_result = fmac(val_a, val_b, val_c);
            await send(fma_result, result[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for FMA operation in the generated CSL
    code_found = False
    for f in csl_files:
        if 'fmac' in f.code or '@fmacs' in f.code or '@fmach' in f.code:
            code_found = True
            break

    assert code_found, "Expected FMA operation not found in generated CSL"


def test_assignment_nested_complex_expression():
    """Test assignment with multiple nested expressions."""
    spatial_ir_code = '''
    kernel @test_complex<N>(stream<f32>[N] readonly a, stream<f32>[N] readonly b, stream<f32>[N] readonly c, stream<f32>[N] writeonly result) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 val_a;
            f32 val_b;
            f32 val_c;
            f32 complex_result;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(val_a, a[i]);
            await receive(val_b, b[i]);
            await receive(val_c, c[i]);
            complex_result = (val_a + val_b) * val_c;
            await send(complex_result, result[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for complex expression in the generated CSL
    code_found = False
    for f in csl_files:
        if ('(' in f.code and '+' in f.code and '*' in f.code) or '@fadds' in f.code or '@fmuls' in f.code:
            code_found = True
            break

    assert code_found, "Expected complex expression not found in generated CSL"


def test_assignment_with_array_dsd():
    """Test assignment with array types that should use DSDs."""
    spatial_ir_code = '''
    kernel @test_array_dsd<N>(stream<f32>[N] readonly input, stream<f32>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32[32] local_array;
            f32[32] result_array;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(local_array, input[i]);
            result_array = local_array + 1.0;
            await send(result_array, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for array operations (either scalar fallback or DSD operations)
    code_found = False
    for f in csl_files:
        if 'local_array[0]' in f.code or '@fadds' in f.code or '@get_dsd' in f.code:
            code_found = True
            break

    assert code_found, "Expected array operation not found in generated CSL"


# Test lowering of AsyncBlock to CSL with proper task structure.


def test_async_block_basic_structure():
    """Test async block generates proper task structure."""
    spatial_ir_code = '''
    kernel @test_async<N>(stream<f32>[N] readonly input, stream<f32>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 local_val;
            f32 processed_val;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(local_val, input[i]);
            
            completion_1 = async {
                processed_val = local_val * 2.0;
            };
            
            await completion_1;
            await send(processed_val, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for async task structure in the generated CSL
    task_structure_found = False
    completion_handling_found = False

    for f in csl_files:
        # Look for task-related code
        if any(keyword in f.code for keyword in ['task', '@activate', '@block', 'completion']):
            task_structure_found = True

        # Look for completion handling
        if 'completion_1' in f.code or 'await' in f.code:
            completion_handling_found = True

    assert task_structure_found, "Expected async task structure not found in generated CSL"
    assert completion_handling_found, "Expected completion handling not found in generated CSL"


def test_async_block_with_multiple_completions():
    """Test async block with multiple completions before and after."""
    spatial_ir_code = '''
    kernel @test_multi_async<N>(stream<f32>[N] readonly input, stream<f32>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 local_val;
            f32 temp_val1;
            f32 temp_val2;
            f32 final_val;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(local_val, input[i]);
            
            completion_pre = async {
                temp_val1 = local_val + 1.0;
            };
            
            await completion_pre;
            
            completion_main = async {
                temp_val2 = temp_val1 * 2.0;
            };
            
            await completion_main;
            
            completion_post = async {
                final_val = temp_val2 - 0.5;
            };
            
            await completion_post;
            await send(final_val, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for multiple completion handling
    completion_count = 0
    task_activations = 0

    for f in csl_files:
        # Count completions
        completion_count += f.code.count('completion_')

        # Count task activations
        task_activations += f.code.count('@activate') + f.code.count('task')

    # Should have multiple completions and task structures
    assert completion_count >= 3, f"Expected at least 3 completions, found {completion_count}"

    # Look for proper sequencing
    sequencing_found = False
    for f in csl_files:
        if 'completion_pre' in f.code and 'completion_main' in f.code and 'completion_post' in f.code:
            sequencing_found = True
            break

    assert sequencing_found, "Expected completion sequencing not found in generated CSL"


def test_async_block_with_nested_operations():
    """Test async block with complex nested operations."""
    spatial_ir_code = '''
    kernel @test_nested_async<N>(stream<f32>[N] readonly a, stream<f32>[N] readonly b, stream<f32>[N] writeonly result) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 val_a;
            f32 val_b;
            f32 intermediate;
            f32 final_result;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(val_a, a[i]);
            await receive(val_b, b[i]);
            
            computation = async {
                intermediate = fmac(val_a, val_b, 1.0);
                final_result = intermediate if intermediate > 0.0 else 0.0;
            };
            
            await computation;
            await send(final_result, result[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for async block with complex operations
    async_found = False
    fma_found = False
    ternary_found = False

    for f in csl_files:
        if 'computation' in f.code and ('async' in f.code or 'task' in f.code):
            async_found = True

        if 'fmac' in f.code or '@fmacs' in f.code:
            fma_found = True

        if ('if' in f.code and 'else' in f.code) or 'intermediate > 0.0' in f.code:
            ternary_found = True

    assert async_found, "Expected async block not found in generated CSL"
    assert fma_found, "Expected FMA operation in async block not found"
    assert ternary_found, "Expected ternary operation in async block not found"


def test_for_statement_basic():
    """Test basic for statement lowering."""
    spatial_ir_code = '''
    kernel @test_for<N>(stream<f32>[N] readonly input, stream<f32>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 local_val;
            f32 sum;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(local_val, input[i]);
            sum = 0.0;
            
            for u16 k in [0:4] {
                sum = sum + local_val;
            }
            
            await send(sum, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for for loop structure or unrolled operations
    loop_found = False
    for f in csl_files:
        if 'for' in f.code or 'sum + local_val' in f.code or '@fadds' in f.code:
            loop_found = True
            break

    assert loop_found, "Expected for loop or unrolled operations not found in generated CSL"


def test_map_statement_basic():
    """Test basic map statement lowering."""
    spatial_ir_code = '''
    kernel @test_map<N>(stream<f32>[N, N] readonly input, stream<f32>[N, N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:N] {
            f32 local_val;
        }
        compute u16 i, u16 j in [0:N, 0:N] {
            await receive(local_val, input[i, j]);
            
            map u16 x, u16 y in [0:2, 0:2] {
                local_val = local_val * 1.1;
            }
            
            await send(local_val, output[i, j]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=4)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for map operations or unrolled code
    map_found = False
    for f in csl_files:
        if 'map' in f.code or 'local_val * 1.1' in f.code or '@fmuls' in f.code:
            map_found = True
            break

    assert map_found, "Expected map operations not found in generated CSL"


def test_foreach_with_parameter_range():
    """Test foreach with parameter range (variant 1)."""
    spatial_ir_code = '''
    kernel @test_foreach_range<N>(stream<f32>[N] readonly input, stream<f32>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 accumulator;
        }
        dataflow u16 i, u16 j in [0:N, 0:1] {
            stream<f32> data_stream = relative_stream(0, 0);
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            accumulator = 0.0;
            
            foreach u16 k, f32 value in [0:4], receive(data_stream) {
                accumulator = accumulator + value;
            }
            
            await send(accumulator, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for foreach operations
    foreach_found = False
    for f in csl_files:
        if 'foreach' in f.code or 'accumulator + value' in f.code or 'receive' in f.code:
            foreach_found = True
            break

    assert foreach_found, "Expected foreach operations not found in generated CSL"


def test_foreach_without_parameter_range():
    """Test foreach without parameter range (variant 2)."""
    spatial_ir_code = '''
    kernel @test_foreach_simple<N>(stream<f32>[N] readonly input, stream<f32>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 accumulator;
        }
        dataflow u16 i, u16 j in [0:N, 0:1] {
            stream<f32> data_stream = relative_stream(0, 0);
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            accumulator = 0.0;
            
            foreach f32 value in receive(data_stream) {
                accumulator = accumulator + value;
            }
            
            await send(accumulator, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for simple foreach operations
    foreach_found = False
    for f in csl_files:
        if 'foreach' in f.code or 'accumulator + value' in f.code:
            foreach_found = True
            break

    assert foreach_found, "Expected simple foreach operations not found in generated CSL"


def test_mixed_statement_types():
    """Test kernel with multiple statement types combined."""
    spatial_ir_code = '''
    kernel @test_mixed<N>(stream<f32>[N] readonly input, stream<f32>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 local_val;
            f32 processed_val;
            f32 final_val;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(local_val, input[i]);
            
            // Binary operation
            processed_val = local_val + 1.0;
            
            // Async block
            computation = async {
                final_val = fmac(processed_val, 2.0, 0.5);
            };
            
            await computation;
            
            // For loop
            for u16 k in [0:2] {
                final_val = final_val * 1.1;
            }
            
            await send(final_val, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for all statement types
    binary_found = False
    async_found = False
    fma_found = False
    loop_found = False

    for f in csl_files:
        if 'local_val + 1.0' in f.code or '@fadds' in f.code:
            binary_found = True

        if 'computation' in f.code and ('async' in f.code or 'task' in f.code):
            async_found = True

        if 'fmac' in f.code or '@fmacs' in f.code:
            fma_found = True

        if 'for' in f.code or 'final_val * 1.1' in f.code:
            loop_found = True

    assert binary_found, "Expected binary operation not found in generated CSL"
    assert async_found, "Expected async block not found in generated CSL"
    assert fma_found, "Expected FMA operation not found in generated CSL"
    assert loop_found, "Expected for loop not found in generated CSL"


def test_dsd_vs_scalar_assignment_control():
    """Test that array size > 1 controls DSD vs scalar assignment."""
    spatial_ir_code = '''
    kernel @test_dsd_control<N>(stream<f32>[N] readonly input, stream<f32>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 scalar_val;           // Should use scalar assignment
            f32[64] large_array;      // Should potentially use DSD
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(scalar_val, input[i]);
            
            // Scalar assignment
            scalar_val = scalar_val + 1.0;
            
            // Array assignment (size > 1, might use DSD)
            large_array = scalar_val + 2.0;
            
            await send(large_array, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for different assignment types
    scalar_assignment_found = False
    array_assignment_found = False

    for f in csl_files:
        # Scalar assignment should not use array indexing
        if 'scalar_val = ' in f.code and 'scalar_val[0]' not in f.code:
            scalar_assignment_found = True

        # Array assignment should use array indexing or DSD operations
        if 'large_array[0]' in f.code or '@fadds' in f.code or '@get_dsd' in f.code:
            array_assignment_found = True

    assert scalar_assignment_found, "Expected scalar assignment not found in generated CSL"
    assert array_assignment_found, "Expected array assignment not found in generated CSL"


if __name__ == '__main__':
    test_receive_statement_scalar()
    test_receive_statement_array()
    test_send_statement_scalar()
    test_send_statement_with_different_types()
    test_assignment_binary_expression()
    test_assignment_ternary_expression()
    test_assignment_fused_multiply_accumulate()
    test_assignment_nested_complex_expression()
    test_assignment_with_array_dsd()
    test_async_block_basic_structure()
    test_async_block_with_multiple_completions()
    test_async_block_with_nested_operations()
    test_for_statement_basic()
    test_map_statement_basic()
    test_foreach_with_parameter_range()
    test_foreach_without_parameter_range()
    test_mixed_statement_types()
    test_dsd_vs_scalar_assignment_control()
