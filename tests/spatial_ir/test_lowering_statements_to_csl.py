import pytest
from spatialstencil.lowering.spatial_ir_to_csl import lower_spatial_ir_to_csl
from spatialstencil.syntax.spatial_ir import parser, passes


def create_inline_spatial_ir(code: str):
    """Helper function to parse inline Spatial IR code."""
    return parser.parse_string(code, 'test.sptl')


def test_receive_statement_scalar():
    """Test receive statement with scalar types."""
    spatial_ir_code = '''
    kernel @test_receive<N>(stream<f32, 1>[N] readonly input, stream<f32, 1>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 local_val;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(local_val, input[i]);
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
    kernel @test_receive_array<N>(stream<f32, 4>[N] readonly input, stream<f32, 4>[N] writeonly output) {
        place u16 i, u16 j in [0:1, 0:N] {
            f32[4] local_array;
        }
        compute u16 i, u16 j in [0:1, 0:N] {
            await receive(local_array, input[j]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel, copy_elision=False)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for array operations in the generated CSL
    code_found = False
    for f in csl_files:
        if '@fmovs' in f.code:
            code_found = True
            break

    assert code_found, "Expected array receive operation not found in generated CSL"


def test_receive_statement_array_strided():
    """Test receive statement with array types."""
    spatial_ir_code = '''
    kernel @test_receive_array_strided<N>(stream<f32, 4>[N] readonly input, stream<f32, 4>[N] writeonly output) {
        place u16 i, u16 j in [0:1, 0:N:3] {
            f32[4] local_array;
        }
        compute u16 i, u16 j in [0:1, 0:N:3] {
            await receive(local_array, input[j]);
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
    for f in csl_files:
        if 'layout' in f.code:
            break
    else:
        assert False, "Expected layout definition not found in generated CSL"

    assert '@range(i16, 0, 9, 3)' in f.code, "Expected strided range not found in generated CSL"


def test_send_statement_scalar():
    """Test send statement with scalar types."""
    spatial_ir_code = '''
    kernel @test_send<N>(stream<f32, 1>[N] readonly input, stream<f32, 1>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 local_val;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            local_val = 1.0;
            await send(local_val, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel, copy_elision=False)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for send operations in the generated CSL
    code_found = False
    for f in csl_files:
        if 'output' in f.code and '=' in f.code:
            code_found = True
            break

    assert code_found, "Expected send operation not found in generated CSL"


def test_send_statement_array():
    """Test send statement with array types."""
    spatial_ir_code = '''
    kernel @test_send_array<N>(stream<f32, 4>[N] readonly input, stream<f32, 4>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32[4] local_array;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            local_array[0] = 1.0;
            local_array[1] = 2.0;
            local_array[2] = 3.0;
            local_array[3] = 4.0;
            await send(local_array, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel, copy_elision=False)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for send operations in the generated CSL
    code_found = False
    for f in csl_files:
        if '@fmovs' in f.code:
            code_found = True
            break

    assert code_found, "Expected send operation not found in generated CSL"


def test_send_statement_with_different_types():
    """Test send statement with different scalar types."""
    spatial_ir_code = '''
    kernel @test_send_i16<N>(stream<i16, 4>[N] readonly input, stream<i16, 4>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            i16[4] local_val;
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


def test_send_statement_with_constant():
    """Test send statement with constant values."""
    spatial_ir_code = '''
    kernel @test_send_constant<N>(stream<f32, 4>[N] readonly input, stream<f32, 4>[N] writeonly output) {
        compute u16 i, u16 j in [0:N, 0:1] {
            await send(1.0, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for DSD operations
    code_found = False
    for f in csl_files:
        if '@fmovs' in f.code:
            code_found = True
            break

    assert code_found, "Expected @fmovs operation not found in generated CSL"


# Test lowering of AssignmentStatement to CSL with various expression types.


@pytest.mark.parametrize("dsd", (False, True))
@pytest.mark.parametrize("op", ('+', '-', '*'))
def test_assignment_binary_expression(dsd, op):
    """Test assignment with binary expression."""
    arrexp = "[8]" if dsd else ""
    spatial_ir_code = f'''
    kernel @test_binary<N>() {{
        place u16 i, u16 j in [0:N, 0:1] {{
            f32{arrexp} val_a;
            f32{arrexp} val_b;
            f32{arrexp} sum;
        }}
        compute u16 i, u16 j in [0:N, 0:1] {{
            sum = val_a {op} val_b;
        }}
    }}
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    op_to_dsd_op = {
        '+': '@fadds',
        '-': '@fsubs',
        '*': '@fmuls',
    }

    # Look for operation in the generated CSL
    code_found = False
    for f in csl_files:
        if dsd and op_to_dsd_op[op] in f.code:
            code_found = True
            break
        elif not dsd and f'val_a {op} val_b' in f.code:
            code_found = True
            break

    assert code_found, "Expected binary operation not found in generated CSL"


@pytest.mark.parametrize('dsd', (False, True))
@pytest.mark.parametrize('op', ('%', '==', '>='))
def test_assignment_binary_expression_dsd_fallback(dsd, op):
    """
    Test assignment with binary expression (DSD op fallback to map).
    """
    arrexp = "[8]" if dsd else ""
    spatial_ir_code = f'''
    kernel @test_binary<N>() {{
        place u16 i, u16 j in [0:N, 0:1] {{
            f32{arrexp} val_a;
            f32{arrexp} val_b;
            f32{arrexp} out;
        }}
        compute u16 i, u16 j in [0:N, 0:1] {{
            out = val_a {op} val_b;
        }}
    }}
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for operation in the generated CSL
    code_found = False
    for f in csl_files:
        if (not dsd and f'val_a {op} val_b' in f.code) or (dsd and '@map' in f.code and
                                                           f'return (arg0 {op} arg1)' in f.code):
            code_found = True
            break

    assert code_found, "Expected binary operation not found in generated CSL"


@pytest.mark.parametrize('dsd', (False, True))
def test_assignment_ternary_expression(dsd):
    """Test assignment with ternary operator."""
    arrexp = "[8]" if dsd else ""
    spatial_ir_code = f'''
    kernel @test_ternary<N>() {{
        place u16 i, u16 j in [0:N, 0:1] {{
            bool{arrexp} cond;
            f32{arrexp} val_a;
            f32{arrexp} val_b;
            f32{arrexp} output;
        }}
        compute u16 i, u16 j in [0:N, 0:1] {{
            output = val_a if cond else val_b;
        }}
    }}
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
        if (not dsd and 'if (cond) val_a else val_b' in f.code) or (dsd and '@map' in f.code and
                                                                    f'return (if (arg1) arg0 else arg2)' in f.code):
            code_found = True
            break

    assert code_found, "Expected ternary operation not found in generated CSL"


@pytest.mark.parametrize('dsd', (False, True))
def test_assignment_fused_multiply_accumulate(dsd):
    """Test assignment with fused multiply-accumulate."""
    arrexp = "[8]" if dsd else ""
    spatial_ir_code = f'''
    kernel @test_fma<N>() {{
        place u16 i, u16 j in [0:N, 0:1] {{
            f32{arrexp} val_a;
            f32{arrexp} val_b;
            f32 val_c;
            f32{arrexp} fma_result;
        }}
        compute u16 i, u16 j in [0:N, 0:1] {{
            fma_result = fmac(val_a, val_b, val_c);
        }}
    }}
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
        if (not dsd and 'val_a + val_b * val_c' in f.code) or (dsd and '@fmacs' in f.code):
            code_found = True
            break

    assert code_found, "Expected FMA operation not found in generated CSL"


def test_assignment_nested_complex_expression():
    """Test assignment with multiple nested expressions."""
    spatial_ir_code = '''
    kernel @test_complex<N>() {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 val_a;
            f32 val_b;
            f32 val_c;
            f32 complex_result;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            complex_result = (val_a + val_b) * val_c;
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
        if ('(' in f.code and '+' in f.code and '*' in f.code) or ('@fadds' in f.code and '@fmuls' in f.code):
            code_found = True
            break

    assert code_found, "Expected complex expression not found in generated CSL"


def test_assignment_with_array_dsd():
    """Test assignment with array types that should use DSDs."""
    spatial_ir_code = '''
    kernel @test_array_dsd<N>() {
        place u16 i, u16 j in [0:N, 0:1] {
            f32[32] local_array;
            f32[32] result_array;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            result_array = local_array + 1.0;
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
        if '@fadds' in f.code and '@get_dsd' in f.code:
            code_found = True
            break

    assert code_found, "Expected array operation not found in generated CSL"


# Test lowering of AsyncBlock to CSL with proper task structure.


def test_async_block_basic_structure():
    """Test async block generates proper task structure."""
    spatial_ir_code = '''
    kernel @test_async<N>(stream<f32, 1>[N] readonly input, stream<f32, 1>[N] writeonly output, 
                          stream<f32, 1>[N] writeonly processed) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 local_val;
            f32 processed_val;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(local_val, input[i]);
            
            completion c1 = async {
                processed_val = local_val * 2.0;
            };
            completion c2 = send(local_val, output[i]);

            await c1;
            await c2;
            await send(processed_val, processed[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel, task_fusion=False)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for async block
    csl_code = csl_files[0].code
    csl_lines = [l.strip() for l in csl_code.splitlines()]

    assert sum(l == '{' for l in csl_lines) == 1  # One async block

    # Check that @activate follows the orphan "{" lines
    start_line = None
    end_line = None
    for i, line in enumerate(csl_lines):
        if line == '{' and i + 1 < len(csl_lines):
            start_line = i + 1
        if line == '}' and start_line is not None:
            end_line = i
            break

    assert start_line is not None and end_line is not None
    async_block_code = "\n".join(csl_lines[start_line:end_line])

    # Look for code in async task
    assert 'local_val * 2.0' in async_block_code
    assert '@activate' in csl_lines[start_line]

    # Look for async block CSL task structure
    assert 'task_3' in csl_code
    assert 'task_4' not in csl_code


def test_async_block_with_nested_operations():
    """Test async block with complex nested operations."""
    spatial_ir_code = '''
    kernel @test_nested_async<N>(stream<f32, 1>[N] readonly a, stream<f32, 1>[N] readonly b,
                                 stream<f32, 1>[N] writeonly result) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 val_a;
            f32 val_b;
            f32 intermediate;
            f32 final_result;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(val_a, a[i]);
            await receive(val_b, b[i]);
            
            completion computation = async {
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

    csl_files = lower_spatial_ir_to_csl(kernel, task_fusion=False)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for async block with complex operations
    csl_code = csl_files[0].code
    csl_lines = [l.strip() for l in csl_code.splitlines()]

    assert sum(l == '{' for l in csl_lines) == 1  # One async block

    # Check that @activate follows the orphan "{" lines
    start_line = None
    end_line = None
    for i, line in enumerate(csl_lines):
        if line == '{' and i + 1 < len(csl_lines):
            start_line = i + 1
        if line == '}' and start_line is not None:
            end_line = i
            break

    assert start_line is not None and end_line is not None
    async_block_code = "\n".join(csl_lines[start_line:end_line])

    # Look for async task in code, find name and only search for fma and ternary in its contents
    assert 'val_a + val_b * 1.0' in async_block_code
    assert 'if ((intermediate > 0.0)) intermediate else 0.0' in async_block_code
    assert '@activate' not in async_block_code
    assert '@unblock' not in async_block_code

    # Look for async block CSL task structure
    assert 'task_2' in csl_code
    assert 'task_3' not in csl_code


def test_async_block_chain():
    """Test two chained async blocks."""
    spatial_ir_code = '''
    kernel @test_async_chain<N>(stream<f32, 1>[N] readonly a, stream<f32, 1>[N] readonly b,
                                 stream<f32, 1>[N] writeonly result) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 val_a;
            f32 val_b;
            f32 intermediate;
            f32 final_result1;
            f32 final_result2;
            f32 final_result3;
            f32 final_result4;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(val_a, a[i]);
            await receive(val_b, b[i]);
            
            completion c1 = async {
                intermediate = fmac(val_a, val_b, 1.0);
                final_result1 = intermediate if intermediate > 0.0 else 0.0;
            };
            completion c2 = async {
                final_result2 = val_a + val_b;
            };
            final_result3 = val_a * val_b;
            await c1;
            await c2;
            final_result4 = final_result1 + final_result2 + final_result3;
            await send(final_result4, result[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel, task_fusion=False)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    csl_code = csl_files[0].code
    csl_lines = [l.strip() for l in csl_code.splitlines()]

    assert sum(l == '{' for l in csl_lines) == 2  # Two async blocks

    # Check that @activate follows the orphan "{" lines
    for i, line in enumerate(csl_lines):
        if line == '{' and i + 1 < len(csl_lines):
            assert csl_lines[i + 1].startswith(
                '@activate'), f"Expected @activate after '{{' at line {i}, but got: {csl_lines[i + 1]}"

    assert 'val_a + val_b * 1.0' in csl_code
    assert 'if ((intermediate > 0.0)) intermediate else 0.0' in csl_code

    # Look for async block CSL task structure
    assert 'task_4' in csl_code
    assert 'task_5' not in csl_code


def test_for_statement_basic():
    """Test basic for statement lowering."""
    spatial_ir_code = '''
    kernel @test_for<N>(stream<f32, 4>[N] readonly input, stream<f32, 1>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32[4] local_val;
            f32 sum;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(local_val, input[i]);
            sum = 0.0;
            
            for u16 k in [0:4] {
                sum = sum + local_val[k];
            }
            
            await send(sum, output[i]);
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel, copy_elision=False)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for for loop structure or unrolled operations
    loop_found = False
    for f in csl_files:
        if 'for' in f.code and ('sum + local_val' in f.code or '@fadds' in f.code):
            loop_found = True
            break

    assert loop_found, "Expected for loop or unrolled operations not found in generated CSL"


def test_map_statement_basic():
    """Test basic map statement lowering."""
    spatial_ir_code = '''
    kernel @test_map<N>() {
        place u16 i, u16 j in [0:N, 0:N] {
            f32 local_val;
            f32[2, 2] output;
        }
        compute u16 i, u16 j in [0:N, 0:N] {
            await map u16 x, u16 y in [0:2, 0:2] {
                output[x, y] = local_val * 1.1 * local_val;
            };
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
        if '@map' in f.code and 'arg0' in f.code and 'arg0 * (1.1 * arg0)' in f.code:
            map_found = True
            break

    assert map_found, "Expected map operations with arg0 parameter substitution not found in generated CSL"


def test_map_statement_with_multiple_inputs():
    """Test map statement with multiple input variables to verify arg0, arg1, etc. usage."""
    spatial_ir_code = '''
    kernel @test_map_multi<N>() {
        place u16 i, u16 j in [0:N, 0:N] {
            f32[2, 2] val_a;
            f32[2, 2] val_b;
            f32[2, 2] output;
        }
        compute u16 i, u16 j in [0:N, 0:N] {
            await map u16 x, u16 y in [0:2, 0:2] {
                output[x, y] = val_a[x, y] * -val_b[x, y] / 10.0;
            };
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=4)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for map operations with multiple arguments
    map_found = False
    for f in csl_files:
        if ('@map' in f.code and 'arg0' in f.code and 'arg1' in f.code and 'arg0 * ((-arg1' in f.code):
            map_found = True
            break

    assert map_found, "Expected map operations with arg0, arg1 parameter substitution not found in generated CSL"


def test_map_statement_with_nonmap_loop():
    """Test basic map statement lowering, where a CSL ``@map`` cannot work."""
    spatial_ir_code = '''
    kernel @test_map<N>() {
        place u16 i, u16 j in [0:N, 0:N] {
            f32[2, 2] local_val;
            f32[2, 2] output;
        }
        compute u16 i, u16 j in [0:N, 0:N] {
            await map u16 x, u16 y in [0:2, 0:2] {
                for u16 k in [0:4] {
                    output[x, y] = local_val * 1.1 + x + y * 10.0;
                }
            };
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
    loop_found = False
    for f in csl_files:
        if f.code.count('for ') == 3 and 'local_val * 1.1' in f.code:
            loop_found = True
            break

    assert loop_found, "Expected loop operations not found in generated CSL"


@pytest.mark.parametrize('multidimensional', [False, True])
def test_map_lifting_to_dsd_op(multidimensional):
    """
    Tests map lifting to DSD operations.
    """
    arrdims = '[2, 2]' if multidimensional else '[4]'
    map_expr = 'map u16 x, u16 y in [0:2, 0:2]' if multidimensional else 'map u16 x in [0:4]'
    outind = 'x, y' if multidimensional else 'x'
    spatial_ir_code = f'''
    kernel @test_map_lifting<N>() {{
        place u16 i, u16 j in [0:N, 0:N] {{
            f32 local_val;
            f32{arrdims} output;
        }}
        compute u16 i, u16 j in [0:N, 0:N] {{
            await {map_expr} {{
                output[{outind}] = local_val * 1.1;
            }};
        }}
    }}
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=4)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for map lifting to DSD operations
    for f in csl_files:
        if '@fmuls' in f.code:
            break
    else:
        raise AssertionError("Expected DSD operations not found in generated CSL")

    # Check for the DSD structure (dimensionality)
    if multidimensional:
        assert 'mem4d' in f.code
    else:
        assert 'mem1d' in f.code


@pytest.mark.parametrize('multidimensional', [False, True])
def test_map_lowering_to_for_loop(multidimensional):
    """
    Tests map lowering to for loop.
    """
    arrdims = '[2, 2]' if multidimensional else '[4]'
    map_expr = 'map u16 x, u16 y in [0:2, 0:2]' if multidimensional else 'map u16 x in [0:4]'
    outind = 'x, y' if multidimensional else 'x'
    spatial_ir_code = f'''
    kernel @test_map_lowering<N>() {{
        place u16 i, u16 j in [0:N, 0:N] {{
            f32 local_val;
            f32{arrdims} output1;
            f32{arrdims} output2;
        }}
        compute u16 i, u16 j in [0:N, 0:N] {{
            await {map_expr} {{
                output1[{outind}] = local_val * 1.1 + x + y * 10.0;
                output2[{outind}] = local_val * 1.2 + x + y * 11.0;
            }};
        }}
    }}
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=4)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0
    code = csl_files[0].code

    # Check for loop structure
    if multidimensional:
        assert code.count('for') == 2
        assert '@range(u16, 0, 2, 1)' in code
        assert 'output1[x, y]' in code
        assert 'output2[x, y]' in code
    else:
        assert code.count('for') == 1
        assert '@range(u16, 0, 4, 1)' in code
        assert 'output1[x]' in code
        assert 'output2[x]' in code


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_foreach_with_parameter_range(streaming, asynchronous):
    """Test foreach with parameter range (variant 1)."""
    suffix = ', 4' if not streaming else ''
    spatial_ir_code = f'''
    kernel @test_foreach_range<N>(stream<f32{suffix}>[N] readonly input) {{
        place u16 i, u16 j in [0:N, 0:1] {{
            f32 accumulator;
        }}
        compute u16 i, u16 j in [0:N, 0:1] {{
            accumulator = 0.0;

            {'completion c = ' if asynchronous else 'await'} foreach u16 k, f32 value in [0:4], receive(input[i]) {{
                accumulator = accumulator + value;
            }};
            {'await c;' if asynchronous else ''}
        }}
    }}
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for data task operations
    if streaming:
        data_task_found = False
        for f in csl_files:
            if 'data_task' in f.code and 'accumulator + value' in f.code and '4' in f.code:
                data_task_found = True
                break

        assert data_task_found, "Expected data task operations not found in generated CSL"
    else:
        loop_found = False
        for f in csl_files:
            if 'for' in f.code and 'accumulator + value' in f.code and 'input[k]' in f.code:
                loop_found = True
                break

        assert loop_found, "Expected for loop operations not found in generated CSL"


def test_foreach_without_parameter_range():
    """Test foreach without parameter range (variant 2)."""
    spatial_ir_code = '''
    kernel @test_foreach_simple<N>(stream<f32>[N] readonly input) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 accumulator;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            accumulator = 0.0;
            
            await foreach f32 value in receive(input[i]) {
                accumulator = accumulator + value;
            }
        }
    }
    '''
    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for data task operations
    data_task_found = False
    for f in csl_files:
        if 'data_task' in f.code and 'accumulator + value' in f.code:
            data_task_found = True
            break

    assert data_task_found, "Expected data task operations not found in generated CSL"


@pytest.mark.parametrize('with_binop', [False, True])
def test_foreach_lifting_to_dsd_op(with_binop):
    """
    Tests foreach lifting to DSD operations.
    """
    suffix = ' + 1.1' if with_binop else ''
    spatial_ir_code = f'''
    kernel @test_foreach_lifting<N>(stream<f32, 8>[N, N] readonly input) {{
        place u16 i, u16 j in [0:N, 0:N] {{
            f32[8] local_val;
        }}
        compute u16 i, u16 j in [0:N, 0:N] {{
            await foreach u16 k, f32 inp in [0:8], receive(input[i, j]) {{
                local_val[k] = inp{suffix};
            }};
        }}
    }}
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.concretize_parameters(kernel, N=4)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel, copy_elision=False)

    # Check that CSL files were generated
    assert len(csl_files) > 0

    # Look for map lifting to DSD operations
    dsd_found = False
    for f in csl_files:
        if with_binop and '@fadds' in f.code:
            dsd_found = True
            break
        elif not with_binop and '@fmovs' in f.code:
            dsd_found = True
            break

    assert dsd_found, "Expected DSD operations not found in generated CSL"


def test_foreach_receive_op_send_lifting_to_dsd_op():
    spatial_ir_code = '''
    kernel @test_foreach_relay<>() {
        place u16 i, u16 j in [0:1, 0:1] {
            f32[8] local_val;
        }
        dataflow u16 i, u16 j in [0:1, 0:1] {
            stream<f32> input = relative_stream(-1, 0) {
                hops = [(-1, 0)],
                channel = 0
            }
            stream<f32> output = relative_stream(1, 0) {
                hops = [(1, 0)],
                channel = 1
            }
        }
        compute u16 i, u16 j in [0:1, 0:1] {
            await foreach u16 k, f32 inp in [0:8], receive(input) {
                local_val[k] = local_val[k] + inp;
                await send(local_val[k], output);
            };
        }
    }
    '''

    kernel = create_inline_spatial_ir(spatial_ir_code)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel, copy_elision=False)

    assert len(csl_files) > 0

    dsd_found = False
    data_task_found = False
    for f in csl_files:
        if '@fadds' in f.code:
            dsd_found = True
        if 'data_task' in f.code and 'local_val[k] = local_val[k] + inp' in f.code:
            data_task_found = True

    assert dsd_found, "Expected relay DSD operation not found in generated CSL"
    assert not data_task_found, "Relay foreach should not be lowered as a data task"


if __name__ == '__main__':
    test_receive_statement_scalar()
    test_receive_statement_array()
    test_receive_statement_array_strided()
    test_send_statement_scalar()
    test_send_statement_array()
    test_send_statement_with_different_types()
    test_send_statement_with_constant()
    test_assignment_binary_expression(dsd=False, op='+')
    test_assignment_binary_expression(dsd=True, op='+')
    test_assignment_binary_expression_dsd_fallback(dsd=False, op='%')
    test_assignment_binary_expression_dsd_fallback(dsd=True, op='%')
    test_assignment_ternary_expression(False)
    test_assignment_ternary_expression(True)
    test_assignment_fused_multiply_accumulate(False)
    test_assignment_fused_multiply_accumulate(True)
    test_assignment_nested_complex_expression()
    test_assignment_with_array_dsd()
    test_async_block_basic_structure()
    test_async_block_with_nested_operations()
    test_async_block_chain()
    test_for_statement_basic()
    test_map_statement_basic()
    test_map_statement_with_multiple_inputs()
    test_map_statement_with_nonmap_loop()
    test_map_lifting_to_dsd_op(False)
    test_map_lifting_to_dsd_op(True)
    test_map_lowering_to_for_loop(False)
    test_map_lowering_to_for_loop(True)
    test_foreach_with_parameter_range(False, False)
    test_foreach_with_parameter_range(False, True)
    test_foreach_with_parameter_range(True, False)
    test_foreach_with_parameter_range(True, True)
    test_foreach_without_parameter_range()
    test_foreach_lifting_to_dsd_op(False)
    test_foreach_lifting_to_dsd_op(True)
