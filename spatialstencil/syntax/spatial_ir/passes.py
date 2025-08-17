import copy
from spatialstencil.syntax.spatial_ir import irnodes as spa
from spatialstencil.syntax.stencil_ir.type_inference import _result_type_of


class Concretizer(spa.NodeTransformer):

    def __init__(self, parameters: dict[str, int]):
        super().__init__()
        self.params = parameters

    def visit_Kernel(self, node: spa.Kernel):
        new_params = []
        for p in node.parameters:
            if p.name in self.params:
                continue
            new_params.append(p)
        node.parameters = new_params
        return self.generic_visit(node)

    def visit_Identifier(self, node: spa.Identifier):
        if node.name in self.params:
            return spa.ConstantLiteral(self.params[node.name], spa.ScalarType.i32)
        return self.generic_visit(node)


class FindAndReplace(spa.NodeTransformer):
    """
    A node transformer that replaces old nodes with new nodes.
    """

    def __init__(self, replacements: dict[spa.SpatialNode, spa.SpatialNode]):
        super().__init__()
        self.replacements = replacements

    def visit(self, node: spa.SpatialNode):
        try:
            if node in self.replacements:
                return copy.deepcopy(self.replacements[node])
        except TypeError:
            # If the node is not hashable, we cannot use it as a key in a dict.
            # This is the case for some complex nodes like expressions.
            pass
        return super().visit(node)


def concretize_parameters(kernel: spa.Kernel, **parameters: int) -> spa.Kernel:
    """
    Specialize the given parameters to concrete values in the input kernel.
    Modifies the kernel in-place.

    :param kernel: The kernel to specialize.
    :param parameters: The parameter names and values to set. For example,
                       ``concretize_parameters(kernel, I=128, J=128, K=80)``.
    """
    param_names = [p.name for p in kernel.parameters]
    for param in parameters.keys():
        if param not in param_names:
            raise NameError(f'Parameter {param} is not a parameter of kernel {kernel.name}')

    return Concretizer(parameters).visit(kernel)


class ConstExprPropagation(spa.NodeTransformer):
    """
    Propagates constant expressions throughout IR expressions in the code.
    These include parameters with values and constant literals.
    """

    def visit_Parameter(self, node: spa.Parameter):
        if node.value is not None:
            return spa.ConstantLiteral(node.value, spa.ScalarType.i32)
        return node

    def visit_UnaryOperator(self, node: spa.UnaryOperator):
        value: spa.Expression = self.generic_visit(node.value)
        if isinstance(value.value, spa.ConstantLiteral):
            restype = _result_type_of(value.value.dtype, optype=node.op)
            if node.op == '+':
                cval = +value.value.value
            elif node.op == '-':
                cval = -value.value.value
            else:
                raise TypeError(f'Unrecognized unary operator "{node.op}"')
            return spa.ConstantLiteral(cval, restype)

        node.value = value
        return node

    def visit_BinaryOperator(self, node: spa.BinaryOperator):
        left: spa.Expression = self.generic_visit(node.left)
        right: spa.Expression = self.generic_visit(node.right)
        if isinstance(left.value, spa.ConstantLiteral) and isinstance(right.value, spa.ConstantLiteral):
            restype = _result_type_of(left.value.dtype, right.value.dtype, optype=node.op)
            if node.op == '+':
                result = left.value.value + right.value.value
            elif node.op == '-':
                result = left.value.value - right.value.value
            elif node.op == '*':
                result = left.value.value * right.value.value
            elif node.op == '/':
                result = left.value.value / right.value.value
            elif node.op == '//':
                result = left.value.value // right.value.value
            elif node.op == '%':
                result = left.value.value % right.value.value
            elif node.op == '==':
                result = left.value.value == right.value.value
            elif node.op == '!=':
                result = left.value.value == right.value.value
            elif node.op == '<':
                result = left.value.value == right.value.value
            elif node.op == '<=':
                result = left.value.value == right.value.value
            elif node.op == '>':
                result = left.value.value == right.value.value
            elif node.op == '>=':
                result = left.value.value == right.value.value
            else:
                raise TypeError(f'Unrecognized binary operator "{node.op}"')
            return spa.ConstantLiteral(result, restype)

        node.left = left
        node.right = right
        return node

    def visit_TernaryOperator(self, node: spa.TernaryOperator):
        cond: spa.Expression = self.generic_visit(node.cond)
        iftrue: spa.Expression = self.generic_visit(node.if_true)
        iffalse: spa.Expression = self.generic_visit(node.if_false)
        if (isinstance(cond.value, spa.ConstantLiteral) and isinstance(iftrue.value, spa.ConstantLiteral) and
                isinstance(iffalse.value, spa.ConstantLiteral)):
            restype = _result_type_of(iftrue.value.dtype, iffalse.value.dtype, optype=None)
            result = iftrue.value.value if cond.value.value else iffalse.value.value
            return spa.ConstantLiteral(result, restype)

        node.cond = cond
        node.if_true = iftrue
        node.if_false = iffalse
        return node


def constexpr_propagation(kernel: spa.Kernel) -> spa.Kernel:
    """
    Evaluates constant expressions in a kernel.
    """
    return ConstExprPropagation().visit(kernel)
