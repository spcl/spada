"""Tests for lowering affine loop nests to DSD operations.

A sequential nest such as

    for k, l in [0:K, 0:K] { z[k] = z[k] + A[k*K + l] * x[l] }

is rewritten (``preprocessing.vectorize_affine_loops``) into a sequential loop around a map over the
vector axis, which the existing DSD detection (``dsd_ops.get_dsd_op``) lowers to one operation. The
op kind, dtype dispatch and DSD names therefore come from the existing code; these tests check the
nests that must and must not take that path, and that the lowered code computes what the scalar
loops do.

Numerics are checked with a small interpreter for the generated CSL (memory DSDs, the DSD builtins
used here, ``@increment_dsd_offset``, ``for``/``@range`` loops and scalar statements), run on
identical random inputs against the same program lowered with ``--disable-dsd``.
"""
import contextlib
import io
import re
from pathlib import Path
import numpy as np
import pytest
from spada.lowering.spatial_ir_to_csl import lower_spatial_ir_to_csl
from spada.syntax.csl import dsd_ops, preprocessing
from spada.syntax.spatial_ir import parser, passes


@pytest.fixture(autouse=True)
def _restore_dsd_flag():
    # ``disable_dsd=True`` sets a module-level flag that lowering never clears
    saved = dsd_ops.DISABLE_DSD
    dsd_ops.DISABLE_DSD = False
    yield
    dsd_ops.DISABLE_DSD = saved


def _kernel(body: str, ta: str = 'f32', tx: str = None, tz: str = None, extra: str = '', extra_arg: str = '') -> str:
    tx = tx or ta
    tz = tz or ta
    return f'''
    kernel @t<K>(
        stream<{ta}, K*K>[2, 2] readonly  inp_A,
        stream<{tx}, K*K>[2, 2] readonly  inp_x,
        stream<{tz}, K*K>[2, 2] writeonly out{extra_arg}
    ) {{
        place i16 i, i16 j in [0:2, 0:2] {{
            {ta}[K*K] A
            {tx}[K*K] x
            {tz}[K*K] z
        }}
        phase {{
            compute i16 i, i16 j in [0:2, 0:2] {{
                await receive(A, inp_A[i, j])
                await receive(x, inp_x[i, j])
                {body}
                await send(z, out[i, j])
                {extra}
            }}
        }}
    }}
    '''


def _lower(code: str, **kwargs) -> str:
    kernel = parser.parse_string(code, 'test.sptl')
    kernel = passes.concretize_parameters(kernel, K=4)
    kernel = passes.constexpr_propagation(kernel)
    files = lower_spatial_ir_to_csl(kernel, **kwargs)
    return '\n'.join(f.code for f in files if f.filename.startswith('code_0_0'))


def _block(body: str, ta: str = 'f32', tx: str = None, tz: str = None, extra_place: str = ''):
    """ A PE block (place/dataflow/compute) around ``body``, for tests of the pass itself. """
    from spada.lowering import spatial_ir_to_csl as s2c
    from spada.syntax.spatial_ir.canonicalization import PEBlock
    tx, tz = tx or ta, tz or ta
    kernel = parser.parse_string(f'''
    kernel @t<K>(stream<f32>[4] readonly inp, stream<f32> readonly outp) {{
        place i16 i, i16 j in [0, 0] {{
            {ta}[K*K] A
            {tx}[K*K] x
            {tz}[K*K] z
            {extra_place}
        }}
        dataflow i32 i, i32 j in [0, 0] {{
            stream<f32> s = relative_stream(-1, 0) {{ hops = [(-1, 0)], channel = 0 }}
        }}
        compute i32 i, i32 j in [0, 0] {{
            {body}
        }}
    }}''', 't.sptl')
    kernel = passes.constexpr_propagation(passes.concretize_parameters(kernel, K=4))
    place, dataflow, compute = kernel.body
    block = PEBlock(place, dataflow, compute)
    return block, s2c._collect_identifier_types(block, kernel.arguments)


def _rewrites(body: str, **kw) -> int:
    """ How many loop nests the pass rewrites in ``body``. """
    block, dtypes = _block(body, **kw)
    return preprocessing.vectorize_affine_loops(block, dtypes)


# ---------------------------------------------------------------------------------------------
# A minimal interpreter for the generated CSL


def _split_args(text: str) -> list[str]:
    args, depth, cur = [], 0, ''
    for ch in text:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            args.append(cur.strip())
            cur = ''
        else:
            cur += ch
    args.append(cur.strip())
    return args


class _Dsd:

    def __init__(self, array: str, extent: int, expr: str, offset: int = 0):
        self.array, self.extent, self.expr, self.offset = array, extent, expr, offset

    def indices(self, env) -> list[int]:
        return [int(eval(self.expr, {}, {**env, '__index': i})) + self.offset for i in range(self.extent)]


_UNARY = {'@fnegs': lambda a: -a, '@fnegh': lambda a: -a}
_BINARY = {
    '@fadds': np.add,
    '@faddh': np.add,
    '@faddhs': np.add,
    '@add16': np.add,
    '@fsubs': np.subtract,
    '@fsubh': np.subtract,
    '@sub16': np.subtract,
    '@fmuls': np.multiply,
    '@fmulh': np.multiply,
}
_MOVES = ('@fmovs', '@fmovh', '@mov16', '@mov32')
_MACS = ('@fmacs', '@fmach', '@fmachs')
_KNOWN_OPS = list(_UNARY) + list(_BINARY) + list(_MOVES) + list(_MACS)


def _interpret(code: str, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """ Runs the straight-line/looping compute statements of one generated code file. """
    arrays = {k: v.copy() for k, v in arrays.items()}
    dsds = {
        m.group(1): _Dsd(m.group(3), int(m.group(2)), m.group(4))
        for m in re.finditer(
            r'const (\w+) = @get_dsd\(mem1d_dsd, \.\{ \.tensor_access = \|__index\|\{(\d+)\} -> (\w+)\[(.*?)\] \}\);',
            code)
    }
    root: list = []
    stack: list = [(None, root)]
    for raw in code.splitlines():
        line = raw.strip()
        m = re.match(r'for \(@range\(\w+, (-?\d+), (-?\d+), (-?\d+)\)\) \|(\w+)\| \{$', line)
        if m:
            node = ('for', m.group(4), range(int(m.group(1)), int(m.group(2)), int(m.group(3))), [])
            stack[-1][1].append(node)
            stack.append((node, node[3]))
        elif line.endswith('{') and not line.startswith('const') and not line.startswith('.'):
            stack.append((None, stack[-1][1]))  # fn/task/comptime block: transparent
        elif line in ('}', '};'):
            if len(stack) > 1:
                stack.pop()
        elif re.match(r'(%s)\(.*\);$' % '|'.join(_KNOWN_OPS), line):
            stack[-1][1].append(('op', line))
        elif re.match(r'\w+\[.*\] = .*;$', line):
            stack[-1][1].append(('assign', line))

    def operand(token: str, env):
        if token in dsds:
            return dsds[token]
        m = re.match(r'@increment_dsd_offset\((\w+), (.*), \w+\)$', token)
        if m:
            base = dsds[m.group(1)]
            return _Dsd(base.array, base.extent, base.expr, base.offset + int(eval(m.group(2), {}, env)))
        return float(eval(token, {}, {**arrays, **env}))

    def vec(value, env, n):
        if isinstance(value, _Dsd):
            return arrays[value.array][value.indices(env)]
        return np.full(n, value)

    def run(nodes, env):
        for node in nodes:
            if node[0] == 'for':
                for v in node[2]:
                    run(node[3], {**env, node[1]: v})
            elif node[0] == 'assign':
                lhs, rhs = node[1][:-1].split(' = ', 1)
                name, idx = re.match(r'(\w+)\[(.*)\]$', lhs).groups()
                arrays[name][int(eval(idx, {}, {**arrays, **env}))] = eval(rhs, {}, {**arrays, **env})
            else:
                m = re.match(r'(@\w+)\((.*)\);$', node[1])
                op, args = m.group(1), [operand(a, env) for a in _split_args(m.group(2))]
                dst = args[0]
                n = dst.extent
                srcs = [vec(a, env, n) for a in args[1:]]
                if op in _MACS:
                    result = srcs[0] + srcs[1] * srcs[2]
                elif op in _BINARY:
                    result = _BINARY[op](srcs[0], srcs[1])
                elif op in _UNARY:
                    result = _UNARY[op](srcs[0])
                elif op in _MOVES:
                    result = srcs[0]
                else:
                    raise NotImplementedError(op)
                arrays[dst.array][dst.indices(env)] = result

    run(root, {})
    return arrays


def _inputs(ta: str = 'f32', tx: str = None, tz: str = None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    # small integers keep every dtype exact
    return {
        'inp_A': rng.integers(-3, 4, 16).astype(np.float64),
        'inp_x': rng.integers(-3, 4, 16).astype(np.float64),
        'out': rng.integers(-3, 4, 16).astype(np.float64),
    }


def _check(body: str, expect: list[str] = (), forbid: list[str] = (), reference=None, **kw) -> str:
    """
    Lowers ``body`` with DSD detection and checks the generated code, then runs it on random
    inputs and compares with ``reference(arrays)`` or, by default, with the same program lowered
    with ``--disable-dsd`` (the scalar loops of the source).
    """
    code = _kernel(body, **kw)
    dsd_code = _lower(code)
    dsd_ops.DISABLE_DSD = False
    for pattern in expect:
        assert re.search(pattern, dsd_code), f'{pattern!r} not in:\n{dsd_code}'
    for pattern in forbid:
        assert not re.search(pattern, dsd_code), f'{pattern!r} unexpectedly in:\n{dsd_code}'
    inputs = _inputs()
    got = _interpret(dsd_code, inputs)
    if reference is not None:
        want = reference({k: v.copy() for k, v in inputs.items()})
    else:
        want = _interpret(_lower(code, disable_dsd=True), inputs)
        dsd_ops.DISABLE_DSD = False
    for name in inputs:
        np.testing.assert_array_equal(got[name], want[name], err_msg=f'{name} differs\n{dsd_code}')
    return dsd_code


# ---------------------------------------------------------------------------------------------
# Positive cases

_NEST2 = 'for i16 k, i16 l in [0:K, 0:K] {{ {stmt} }}'


@pytest.mark.parametrize('stmt, builtin', [
    ('z[k] = z[k] + A[k*K+l] * x[l]', '@fmacs'),
    ('z[k] = x[l] * A[k*K+l] + z[k]', '@fmacs'),
    ('z[k] = fmac(z[k], A[k*K+l], x[l])', '@fmacs'),
    ('z[k] = A[k*K+l] * x[l]', '@fmuls'),
    ('z[k] = z[k] + A[k*K+l]', '@fadds'),
    ('z[k] = z[k] - A[k*K+l]', '@fsubs'),
    ('z[k] = A[k*K+l] - x[l]', '@fsubs'),
    ('z[k] = -A[k*K+l]', '@fnegs'),
    ('z[k] = A[k*K+l]', '@fmovs'),
],
                         ids=['mac', 'mac_commuted', 'fmac_call', 'mul', 'add', 'sub', 'sub_scalar', 'neg', 'mov'])
def test_op_kinds(stmt, builtin):
    code = _check(_NEST2.format(stmt=stmt),
                  expect=[re.escape(builtin) + r'\('],
                  forbid=[r'for \(@range\(i16, 0, 4, 1\)\) \|k\|'])
    assert 'increment_dsd_offset' in code


@pytest.mark.parametrize('header', ['k, i16 l in [0:K, 0:K]', 'l, i16 k in [0:K, 0:K]'], ids=['kl', 'lk'])
def test_depth2_both_loop_orders(header):
    code = _check(f'for i16 {header} {{ z[k] = z[k] + A[k*K+l] * x[l] }}',
                  expect=[r'@fmacs\(out_dsd, out_dsd, @increment_dsd_offset\(inp_A_dsd, l, f32\), inp_x\[l\]\)'])
    assert code.count('@get_dsd(mem1d_dsd') == 2  # one DSD per array, none per loop


def test_nested_for_statements_equal_one_multivariable_for():
    flat = _lower(_kernel('for i16 k, i16 l in [0:K, 0:K] { z[k] = z[k] + A[k*K+l] * x[l] }'))
    nested = _lower(_kernel('for i16 k in [0:K] { for i16 l in [0:K] { z[k] = z[k] + A[k*K+l] * x[l] } }'))
    pick = lambda c: [l for l in c.splitlines() if '@fmacs' in l or '@get_dsd(mem1d' in l]
    assert pick(flat) == pick(nested) and pick(flat)


@pytest.mark.parametrize('header, stmt, offset', [
    ('k, i16 l, i16 m in [0:K, 0:2, 0:2]', 'z[k] = z[k] + A[k + l*4 + m*8] * x[l*2+m]', r'l \* 4 \+ m \* 8'),
    ('l, i16 k, i16 m in [0:2, 0:K, 0:2]', 'z[k] = z[k] + A[k + l*4 + m*8] * x[l*2+m]', r'l \* 4 \+ m \* 8'),
    ('l, i16 m, i16 k in [0:2, 0:2, 0:K]', 'z[k] = z[k] + A[k + l*4 + m*8] * x[l*2+m]', r'l \* 4 \+ m \* 8'),
],
                         ids=['axis_first', 'axis_middle', 'axis_last'])
def test_depth3_axis_positions(header, stmt, offset):
    code = _check(f'for i16 {header} {{ {stmt} }}',
                  expect=[r'@fmacs\(', rf'@increment_dsd_offset\(inp_A_dsd, {offset}, f32\)'])
    assert len(re.findall(r'for \(@range', code)) == 2  # the two non-axis loops stay sequential


def test_depth3_destination_indexed_by_an_outer_loop_variable():
    def reference(a):
        for m in range(2):
            for l in range(2):
                for k in range(4):
                    a['out'][k + m * 4] += a['inp_A'][k + l * 8 + m * 4] * a['inp_x'][l]
        return a

    code = _check('for i16 m, i16 l, i16 k in [0:2, 0:2, 0:K] { z[k + m*4] = z[k + m*4] + A[k + l*8 + m*4] * x[l] }',
                  expect=[r'@fmacs\(@increment_dsd_offset\(out_dsd, m \* 4, f32\), '
                          r'@increment_dsd_offset\(out_dsd, m \* 4, f32\), ',
                          r'@increment_dsd_offset\(out_dsd, m \* 4, f32\)',
                          r'@increment_dsd_offset\(inp_A_dsd, l \* 8 \+ m \* 4, f32\)'],
                  reference=reference)
    assert len(re.findall(r'for \(@range', code)) == 2


def test_strided_destination_and_source():
    # Z[k*C1] = Z[k*C1] + X[k*C2]. Lowering this as scalar code is not possible (a destination index
    # must be a plain variable or constant there), so the reference is computed directly
    def reference(a):
        for k in range(4):
            a['out'][k * 2] += a['inp_x'][k * 3]
        return a

    code = _check('for i16 k in [0:K] { z[k*2] = z[k*2] + x[k*3] }',
                  expect=[r'@fadds\(out_dsd, out_dsd, inp_x_dsd\)'],
                  reference=reference)
    assert '-> out[__index * 2]' in code and '-> inp_x[__index * 3]' in code
    assert 'for (@range' not in code.split('@fadds')[0].split('task_0')[-1]


@pytest.mark.parametrize('rng, text', [('1:K', '{3}'), ('0:K:2', '{2}'), ('1:K:2', '{2}')],
                         ids=['start', 'step', 'both'])
def test_nonzero_start_and_nonunit_step(rng, text):
    code = _check(f'for i16 l, i16 k in [0:K, {rng}] {{ z[k] = z[k] + A[k*K+l] * x[l] }}', expect=[r'@fmacs\('])
    assert text in code


def test_start_adds_to_base():
    code = _check('for i16 l, i16 k in [0:K, 1:K] { z[k] = z[k] + A[k*K+l] * x[l] }')
    assert '-> inp_A[__index * 4 + 4]' in code and '-> out[__index + 1]' in code


@pytest.mark.parametrize('ta, tx, tz, builtin', [
    ('f16', 'f16', 'f16', '@fmach'),
    ('f32', 'f32', 'f32', '@fmacs'),
],
                         ids=['f16', 'f32'])
def test_mac_dtype_dispatch(ta, tx, tz, builtin):
    # A is the DSD multiplicand, x the scalar; the accumulator shares the DSD dtype
    code = _check('for i16 k, i16 l in [0:K, 0:K] { z[k] = z[k] + A[k*K+l] * x[l] }',
                  ta=ta,
                  tx=tx,
                  tz=tz,
                  expect=[re.escape(builtin) + r'\('])
    assert 'increment_dsd_offset(inp_A_dsd, l, ' + ta + ')' in code


def test_f32_accumulate_with_f16_scalar_is_off_by_default(monkeypatch):
    # get_dsd_op maps (f32 acc, f32 vector, f16 scalar) to @fmachs. That compiles with SDK 2.10 but
    # does not accumulate on the simulator, so by default the nest stays a scalar loop...
    body = 'for i16 k, i16 l in [0:K, 0:K] { z[k] = z[k] + A[k*K+l] * x[l] }'
    code = _check(body, ta='f32', tx='f16', tz='f32', forbid=[r'@fmachs', r'increment_dsd_offset'])
    assert 'for (@range' in code
    # ... and the switch (one place) turns it on
    monkeypatch.setattr(preprocessing, 'VECTORIZE_F32_DSD_FMACHS', True)
    code = _lower(_kernel(body, ta='f32', tx='f16', tz='f32'))
    assert re.search(r'@fmachs\(out_dsd, out_dsd, @increment_dsd_offset\(inp_A_dsd, l, f32\), inp_x\[l\]\)', code)


@pytest.mark.parametrize('ty, builtin', [('i16', '@add16'), ('u16', '@add16')])
def test_integer_add(ty, builtin):
    _check('for i16 k, i16 l in [0:K, 0:K] { z[k] = z[k] + A[k*K+l] }', ta=ty, expect=[re.escape(builtin) + r'\('])


def test_integer_move():
    _check('for i16 k, i16 l in [0:K, 0:K] { z[k] = A[k*K+l] }', ta='i16', expect=[r'@mov16\('])


def test_dsd_names_are_the_existing_ones():
    code = _lower(_kernel('for i16 k, i16 l in [0:K, 0:K] { z[k] = z[k] + A[k*K+l] * x[l] }'))
    assert '__vecmac' not in code
    assert re.findall(r'^const (\w+_dsd) = ', code, re.M) == ['out_dsd', 'inp_A_dsd']


# ---------------------------------------------------------------------------------------------
# Negative cases: the pass leaves these nests alone (the loops stay as written)

_LOOPS = 'for i16 k, i16 l in [0:K, 0:K] {{ {stmt} }}'


@pytest.mark.parametrize('stmt', [
    pytest.param('z[k+1] = z[k] + A[k*K+l]', id='loop_carried'),
    pytest.param('z[k] = z[k] + z[K-1-k] * x[l]', id='dest_read_at_other_index'),
    pytest.param('z[k] = z[k] + A[k*l] * x[l]', id='non_affine'),
    pytest.param('z[3-k] = z[3-k] + A[k*K+l] * x[l]', id='decreasing_dest'),
    pytest.param('z[k+l] = z[k+l] + A[k*K+l]', id='dest_not_injective'),
    pytest.param('z[k] = z[k] + A[k*K+l] * A[l]', id='two_forms_of_one_array'),
    pytest.param('z[k] = fmac(z[k], A[k*K+l], k)', id='loop_variable_as_value'),
    pytest.param('z[k] = z[k] + A[k*K+l] * x[l]\n z[k] = z[k] + x[l]', id='two_statements'),
])
def test_neg_pass_rewrites_nothing(stmt):
    assert _rewrites(_LOOPS.format(stmt=stmt)) == 0


def test_neg_unsupported_dtype_combination():
    # f32 accumulator with an f16 vector operand: the classifier has no operation for it
    assert _rewrites(_LOOPS.format(stmt='z[k] = z[k] + A[k*K+l] * x[l]'), ta='f16', tx='f16', tz='f32') == 0
    # the same nest with matching dtypes is rewritten, so the above is the dtype's doing
    assert _rewrites(_LOOPS.format(stmt='z[k] = z[k] + A[k*K+l] * x[l]'), ta='f16', tx='f16', tz='f16') == 1


def test_neg_rank_two_array():
    assert _rewrites(_LOOPS.format(stmt='z[k] = z[k] + B[k, l] * x[l]'), extra_place='f32[K, K] B') == 0


def test_neg_non_constant_bounds():
    assert _rewrites('for i16 k, i16 l in [0:K, 0:k] { z[k] = z[k] + A[k*K+l] }') == 0


def test_neg_disable_dsd_keeps_scalar_loops():
    code = _lower(_kernel('for i16 k, i16 l in [0:K, 0:K] { z[k] = z[k] + A[k*K+l] * x[l] }'), disable_dsd=True)
    assert 'increment_dsd_offset' not in code and '@fmacs' not in code and 'for (@range' in code


def test_neg_conflicting_memory_dsd_of_same_array_falls_back():
    # Sending A builds the whole-array DSD of A, which cannot coexist with the strided one
    # (before the pass this program compiled; it must keep compiling)
    code = _lower(
        _kernel('for i16 k, i16 l in [0:K, 0:K] { z[k] = z[k] + A[k*K+l] * x[l] }',
                extra='await send(A, out2[i, j])',
                extra_arg=',\n        stream<f32, K*K>[2, 2] writeonly out2'))
    assert 'increment_dsd_offset' not in code and '@fmacs' not in code
    assert len(re.findall(r'for \(@range', code)) >= 2


def test_neg_rank_two_array_is_not_vectorized():
    code = _kernel('for i16 k, i16 l in [0:K, 0:K] { z[k] = z[k] + B[k, l] * x[l] }')
    code = code.replace('f32[K*K] A\n', 'f32[K, K] B\n            f32[K*K] A\n', 1)
    lowered = _lower(code)
    assert 'increment_dsd_offset' not in lowered and '@fmacs' not in lowered


# ---------------------------------------------------------------------------------------------
# Classifier-level


def test_scalar_element_of_a_map_is_the_fmac_scalar():
    block, dtypes = _block('''
        for i16 l in [0:K] {
            await map i16 k in [0:K] { z[k] = fmac(z[k], A[k], x[l]) }
        }
        await map i16 k in [0:K] { z[k] = fmac(z[k], A[k], x[k]) }
    ''')
    inner_map = block.compute.statements[0].body[0]
    vector_map = block.compute.statements[1]
    # only maps made by the vectorizer follow their ranges; written maps keep the original lowering
    assert dsd_ops.get_dsd_op(dtypes, inner_map) is None
    inner_map.follow_ranges = vector_map.follow_ranges = True
    assert dsd_ops.get_dsd_op(dtypes, inner_map) == '@fmacs'
    assert dsd_ops.get_dsd_op(dtypes, vector_map) is None  # c is a vector there: not an fmac


def test_pass_output_is_a_loop_around_a_map():
    block, dtypes = _block(_LOOPS.format(stmt='z[k] = z[k] + A[k*K+l] * x[l]'))
    assert preprocessing.vectorize_affine_loops(block, dtypes) == 1
    ir = block.compute.statements[0].as_ir()
    assert ir.startswith('for') and 'map' in ir and 'fmac(' in ir.replace('(z', '(z')
    dsd_ops.DISABLE_DSD = True
    block, dtypes = _block(_LOOPS.format(stmt='z[k] = z[k] + A[k*K+l] * x[l]'))
    assert preprocessing.vectorize_affine_loops(block, dtypes) == 0


# ---------------------------------------------------------------------------------------------
# Every shipped sample: lowering must not change except where the pass rewrites a loop

_ROOT = Path(__file__).resolve().parents[2]
#: Large instances take minutes to lower and the 16x16 ones repeat the 4x4x4 structure; shearsort_2D takes over a
#: minute and does not lower at small sizes (before or after the pass)
_SKIP = ('746_990', '128_128', '16_16_4', 'shearsort_2D')
_SAMPLES = sorted(
    str(p.relative_to(_ROOT)) for pattern in ('samples/**/*.sptl', 'tests/csl_runtime/samples/**/*.sptl')
    for p in _ROOT.glob(pattern) if not any(s in p.name for s in _SKIP))

#: Samples the pass rewrites (everything else must lower exactly as it does without the pass)
_VECTORIZED = {
    'samples/benchmarks/pure_vertical_4_4_4.sptl',
    'samples/benchmarks/vertical_advection_4_4_4.sptl',
    'samples/spatial/blas/gemv.sptl',
    'samples/spatial/blas/gemv_twophase.sptl',
    'samples/spatial/blas/matvec.sptl',
}


def _lower_sample(path: str, tmp_path: Path, vectorize: bool, monkeypatch) -> dict[str, str] | type:
    """
    Lowered files of a sample, or the type of the last exception if it does not lower for any of a few
    small parameter values (which samples lower depends on the architecture, ``WSE_ARCH``).
    """
    from spada.cli.compiler import generate_program
    monkeypatch.setattr(preprocessing, 'VECTORIZE_AFFINE_LOOPS', vectorize)
    failure = Exception
    for value in (4, 2, 8):
        kernel = parser.parse_file(str(_ROOT / path))
        params = [f'{p.name}={value}' for p in kernel.parameters if p.value is None]
        out = tmp_path / f'{vectorize}_{value}'
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                generate_program(str(_ROOT / path), str(out), params)
        except Exception as e:
            failure = type(e)
            continue
        # a map callback is named after the id of its statement, which differs between runs
        return {f.name: re.sub(r'map_callback_\d+', 'map_callback', f.read_text()) for f in sorted(out.iterdir())}
    return failure


@pytest.mark.parametrize('path', _SAMPLES)
def test_samples_lower_and_only_vectorized_ones_change(path, tmp_path, monkeypatch):
    without = _lower_sample(path, tmp_path, False, monkeypatch)
    with_pass = _lower_sample(path, tmp_path, True, monkeypatch)
    if isinstance(without, type):
        # Not lowerable on this architecture whatever the pass does: it must not change that
        assert with_pass is without, f'{path} lowers with the pass but not without it' if not isinstance(
            with_pass, type) else f'{path} fails differently with the pass'
        pytest.skip(f'{path} does not lower on this architecture')
    assert not isinstance(with_pass, type), f'{path} fails to lower with the pass ({with_pass.__name__})'
    if path in _VECTORIZED:
        assert with_pass != without, f'{path} was expected to be vectorized'
    else:
        assert with_pass == without, f'{path} changed although it is not expected to be vectorized'
