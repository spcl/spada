from spatialstencil.syntax.spatial_ir import irnodes as spa


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
