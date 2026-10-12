import copy
from typing import Optional
from spada.syntax.spatial_ir import irnodes as spir, affine
from spada.syntax.spatial_ir.canonicalization import PEBlock


def preprocess_rectangle(rect: PEBlock):
    """
    Performs a set of pre-processing passes on code.
    """
    _fma_fusion(rect)


def _fma_fusion(rect: PEBlock):
    """
    Detects multiply-accumulate chains and converts them to ``FusedMultiplyAccumulate`` IR nodes.
    """


#: Whether ``acc + b * c`` is fused to a multiply-accumulate where the vectorizer needs one. The
#: fusion is applied only to the loops ``vectorize_affine_loops`` rewrites; the general
#: ``_fma_fusion`` above stays a TODO. This is the single switch for that scope.
FUSE_MAC_IN_VECTORIZED_LOOPS = True

#: Master switch of ``vectorize_affine_loops``. Off, lowering is exactly what it was without the pass.
VECTORIZE_AFFINE_LOOPS = True

#: Whether a nest may become ``@fmachs`` with a 32-bit DSD multiplicand (f32 accumulator, f32 vector,
#: f16 scalar), which is what ``get_dsd_op`` returns for that combination. The SDK 2.10 compiler accepts
#: it, but on the simulator it leaves the accumulator unchanged, while the library form of
#: ``@fmachs`` (f16 vector and scalar, f32 accumulator) computes correctly. Until the classifier
#: maps that combination to the library form, these nests stay scalar loops.
VECTORIZE_F32_DSD_FMACHS = False


def vectorize_affine_loops(rect: PEBlock, dtypes: dict[spir.Identifier, spir.IRType]) -> int:
    """
    Rewrites sequential affine loop nests into a sequential outer nest around one map, so the
    existing DSD detection (``dsd_ops.get_dsd_op``) lowers the innermost work to one DSD operation.

    A nest

        for k, l, ... in [...] { dst[f(k, l, ...)] = expr over src[g(k, l, ...)] }

    is recognized when the body is a single assignment, every array access is a one-index,
    integer-affine access to a one-dimensional array, and the loop bounds are compile-time
    constants. One variable becomes the map's (vector) axis, the others stay sequential loops;
    accesses to other variables become scalar operands or per-iteration DSD offsets when the
    operation is generated (see ``dsd_ops._with_map_operand_forms``). The rewrite is kept only
    if ``get_dsd_op`` accepts the resulting map, so op kinds and dtype dispatch are decided in
    one place, and only if reordering the iterations cannot change the result (see
    ``_Vectorizer._try`` for the conditions).

    :param rect: The PE block whose compute block is rewritten in place.
    :param dtypes: Types of the identifiers of the block.
    :return: The number of loop nests rewritten.
    """
    from spada.syntax.csl import dsd_ops
    if dsd_ops.DISABLE_DSD or not VECTORIZE_AFFINE_LOOPS:
        return 0
    vectorizer = _Vectorizer(rect.compute, dtypes)
    rect.compute = vectorizer.visit(rect.compute)
    return vectorizer.rewritten


class _Vectorizer(spir.NodeTransformer):

    def __init__(self, compute: spir.ComputeBlock, dtypes: dict[spir.Identifier, spir.IRType]):
        super().__init__()
        self.compute = compute
        self.dtypes = dtypes
        self.rewritten = 0
        #: DSD description (extent, expression) already chosen for an array in this block
        self.forms: dict[spir.Identifier, tuple[str, str]] = {}

    def visit_ForStatement(self, node: spir.ForStatement):
        result = self._try(node)
        if result is not None:
            self.rewritten += 1
            return result
        return self.generic_visit(node)

    # ------------------------------------------------------------------------------------------

    def _vector_array(self, name: spir.Identifier) -> Optional[spir.ArrayType]:
        dtype = self.dtypes.get(name)
        if (isinstance(dtype, spir.ArrayType) and len(dtype.shape) == 1 and self._extent(dtype) is not None and
                isinstance(dtype.base_type, spir.ScalarType)):
            return dtype
        return None

    @staticmethod
    def _extent(dtype: spir.ArrayType) -> Optional[int]:
        dim = dtype.shape[0]
        try:
            dim = dim if isinstance(dim, int) else dim.eval()
        except (ValueError, TypeError, AttributeError):
            return None
        return dim if isinstance(dim, int) else None

    def _try(self, node: spir.ForStatement) -> Optional[spir.Statement]:
        # Collect the (possibly nested) loop variables around a single assignment
        loop_vars: list[spir.TypedIdentifier] = []
        loop_ranges: list[spir.RangeExpression] = []
        inner: spir.Statement = node
        while isinstance(inner, spir.ForStatement):
            loop_vars += inner.variables
            loop_ranges += inner.range_expression
            if len(inner.body) != 1:
                return None
            inner = inner.body[0]
        if not isinstance(inner, spir.AssignmentStatement):
            return None
        assign = inner
        names = [v.identifier for v in loop_vars]
        if len(set(names)) != len(names):
            return None
        bounds = [affine.const_range(r) for r in loop_ranges]
        if any(b is None for b in bounds):
            return None
        trips = [len(range(*b)) for b in bounds]
        if any(t < 1 for t in trips):
            return None
        steps = [b[2] for b in bounds]
        starts = [b[0] for b in bounds]
        loop_set = set(names)

        # Every access: one affine index over loop variables, into a 1-D array
        dst = assign.destination
        slices = [dst] + [n for n in assign.source.walk() if isinstance(n, spir.ArraySlice)]
        if not isinstance(dst, spir.ArraySlice):
            return None
        forms: dict[spir.ArraySlice, affine.AffineForm] = {}
        for sl in slices:
            if len(sl.indices) != 1 or isinstance(sl.indices[0], spir.RangeExpression):
                return None
            if self._vector_array(sl.array) is None:
                return None
            form = affine.affine_form(sl.indices[0])
            if form is None or not set(form[0]) <= loop_set:
                return None
            forms[id(sl)] = form
        if _loop_var_outside_slice(assign.source, loop_set):
            return None

        # Axis: a variable with positive step that strides the destination
        dst_coeffs = forms[id(dst)][0]
        candidates = [i for i, v in enumerate(names) if dst_coeffs.get(v, 0) > 0 and steps[i] > 0]
        if not candidates:
            return None
        axis_pos = max(candidates, key=lambda i: (trips[i], i))
        axis = names[axis_pos]
        if any(forms[id(sl)][0].get(axis, 0) < 0 for sl in slices):
            return None

        # One access form per array, so one DSD describes it. This also means that the destination
        # array is read only at the element it writes.
        per_array: dict[spir.Identifier, tuple] = {}
        for sl in slices:
            coeffs, const = forms[id(sl)]
            key = (const, tuple(sorted((v.as_ir(), c) for v, c in coeffs.items())))
            if per_array.setdefault(sl.array, key) != key:
                return None

        # Distinct iterations of the variables that index the destination write distinct
        # elements, so moving the axis loop innermost keeps the update order of every element
        strides = sorted(((abs(dst_coeffs[v] * steps[i]), trips[i]) for i, v in enumerate(names) if dst_coeffs.get(v)),
                         key=lambda t: t[0])
        span = 0
        for stride, trip in strides:
            if stride <= span:
                return None
            span += stride * (trip - 1)

        # Each array's DSD: no conflict with other memory DSDs of the array in this block
        new_forms: dict[spir.Identifier, tuple[str, str]] = {}
        for sl in slices:
            if sl.array in new_forms:
                continue
            coeffs, const = forms[id(sl)]
            if coeffs.get(axis, 0) == 0:
                continue  # scalar element, no DSD
            c = coeffs[axis]
            base = const + c * starts[axis_pos]
            stride = c * steps[axis_pos]
            expr = '__index' if stride == 1 else f'__index * {stride}'
            if base:
                expr += f' + {base}' if base > 0 else f' - {-base}'
            form = (str(trips[axis_pos]), expr)
            if self.forms.get(sl.array, form) != form:
                return None
            if not self._compatible_elsewhere(sl.array, node, form):
                return None
            new_forms[sl.array] = form
        # The map must read at least one vector (or be a fill with a constant)
        reads_vector = any(isinstance(n, spir.Identifier) and n == axis for sl in slices[1:] for n in sl.walk())
        if not reads_vector and not isinstance(assign.source.value, spir.ConstantLiteral):
            return None

        # Build the map
        new_assign = copy.deepcopy(assign)
        if FUSE_MAC_IN_VECTORIZED_LOOPS:
            new_assign = _fuse_multiply_accumulate(new_assign, axis)
        axis_var = copy.deepcopy(loop_vars[axis_pos])
        axis_range = copy.deepcopy(loop_ranges[axis_pos])
        vector_map = spir.MapStatement([axis_var], [axis_range], [new_assign], follow_ranges=True)
        from spada.syntax.csl import dsd_ops
        op = dsd_ops.get_dsd_op(self.dtypes, vector_map)
        if op in (None, ''):
            return None
        if op == '@fmachs' and not VECTORIZE_F32_DSD_FMACHS and self._multiplicand_is_f32(new_assign):
            return None
        if _unary_of_scalar(new_assign, axis):
            return None

        self.forms.update(new_forms)
        rest = [i for i in range(len(names)) if i != axis_pos]
        result: spir.Statement = vector_map
        if rest:
            result = spir.ForStatement([copy.deepcopy(loop_vars[i]) for i in rest],
                                       [copy.deepcopy(loop_ranges[i]) for i in rest], [vector_map])
        for new in (vector_map, result):
            if hasattr(node, 'lineinfo'):
                new.lineinfo = node.lineinfo
        return result

    def _multiplicand_is_f32(self, assign: spir.AssignmentStatement) -> bool:
        mac = assign.source.value
        if not isinstance(mac, spir.MultiplyAccumulateOperator) or not isinstance(mac.b.value, spir.ArraySlice):
            return False
        dtype = self.dtypes.get(mac.b.value.array)
        return isinstance(dtype, spir.ArrayType) and dtype.base_type == spir.ScalarType.f32

    def _compatible_elsewhere(self, array: spir.Identifier, nest: spir.ForStatement, form: tuple[str, str]) -> bool:
        """
        Whether the memory DSD this loop needs for ``array`` can coexist with every other use of
        the array in the block. The lowering keeps one memory DSD per array, so another use that
        builds a DSD of a different shape would be an error where the loop was valid before.
        """
        extent = self._extent(self.dtypes[array])
        return _other_uses_compatible(self.compute, nest, array, extent, form == (str(extent), '__index'))


def _loop_var_outside_slice(node: spir.SpatialNode, loop_vars: set[spir.Identifier]) -> bool:
    """ Whether a loop variable is used other than inside an array index. """
    if isinstance(node, spir.ArraySlice):
        return False
    if isinstance(node, spir.Identifier):
        return node in loop_vars
    return any(_loop_var_outside_slice(child, loop_vars) for child in node.iter_child_nodes())


def _unary_of_scalar(assign: spir.AssignmentStatement, axis: spir.Identifier) -> bool:
    src = assign.source.value
    return (isinstance(src, spir.UnaryOperator) and
            not any(isinstance(n, spir.Identifier) and n == axis for n in src.walk()))


def _fuse_multiply_accumulate(new_assign: spir.AssignmentStatement,
                              axis: spir.Identifier) -> spir.AssignmentStatement:
    """
    Rewrites ``acc + b * c`` (either operand order) with a vector ``acc`` and ``b`` and a scalar
    element ``c`` into a multiply-accumulate. Anything else is returned unchanged.
    """
    src = new_assign.source.value
    if not (isinstance(src, spir.BinaryOperator) and src.op == '+'):
        return new_assign

    def is_vector(expr: spir.Expression) -> bool:
        v = expr.value
        return isinstance(v, spir.ArraySlice) and any(
            isinstance(n, spir.Identifier) and n == axis for idx in v.indices for n in idx.walk())

    def is_scalar_element(expr: spir.Expression) -> bool:
        v = expr.value
        return isinstance(v, spir.ArraySlice) and not is_vector(expr)

    for add, mul in ((src.left, src.right), (src.right, src.left)):
        m = mul.value
        if not (isinstance(m, spir.BinaryOperator) and m.op == '*') or not is_vector(add):
            continue
        for vec, scal in ((m.left, m.right), (m.right, m.left)):
            if is_vector(vec) and is_scalar_element(scal):
                fused = spir.MultiplyAccumulateOperator(add, vec, scal)
                new_assign.source = spir.Expression(fused)
                return new_assign
    return new_assign


def _other_uses_compatible(compute: spir.ComputeBlock, nest: spir.ForStatement, array: spir.Identifier, extent: int,
                           identity_form: bool) -> bool:
    """
    Whether all uses of ``array`` outside ``nest`` are ones that never build a memory DSD, or build
    the whole-array one (``array[__index]`` over ``extent`` elements) while the loop's own DSD is
    that same one (``identity_form``).
    """

    def identity_vars(node: spir.SpatialNode, known: set) -> set:
        # Variables whose plain use as ``array[v]`` yields the whole-array DSD
        if isinstance(node, spir.ForeachStatement):
            return known | {v.identifier for v in node.variables}
        if isinstance(node, spir.MapStatement):
            return known | {
                v.identifier
                for v, rng in zip(node.variables, node.range_expression)
                if affine.const_range(rng) == (0, extent, 1)
            }
        return known

    def walk(node: spir.SpatialNode, scope_dsd: bool, scope_top: bool, known: set) -> bool:
        if node is nest:
            return True
        if isinstance(node, spir.SendStatement):
            # A send builds the whole-array memory DSD of what it sends
            return identity_form or not any(isinstance(n, spir.Identifier) and n == array for n in node.walk())
        if isinstance(node, spir.ArraySlice):
            if node.array != array:
                return True
            if not scope_dsd:
                return True  # an element access outside a map/foreach is plain scalar code
            only = node.indices[0] if len(node.indices) == 1 else None
            index = only.value if isinstance(only, spir.Expression) else None
            return identity_form and isinstance(index, spir.Identifier) and index in known
        if isinstance(node, spir.ReceiveStatement):
            local = node.local_array
            local = local.array if isinstance(local, spir.ArraySlice) else local
            if local == array:
                return scope_top  # a top-level receive does not build a memory DSD
            return True
        if isinstance(node, spir.Identifier):
            return node != array or identity_form
        child_known = identity_vars(node, known)
        child_dsd = scope_dsd or isinstance(node, (spir.MapStatement, spir.ForeachStatement))
        child_top = scope_top and not isinstance(node, (spir.ForStatement, spir.MapStatement, spir.ForeachStatement))
        return all(walk(c, child_dsd, child_top, child_known) for c in node.iter_child_nodes())

    return walk(compute, False, True, set())
