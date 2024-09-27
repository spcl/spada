"""
Provides basic AST/IR utilities for traversal and transformation.

Walking and transformation are based on Python's ``ast`` module, but introduce additional
functionality such as IR language testing and dataclass support.
"""
from typing import Generic, TypeVar, Sequence
from spatialstencil.syntax.common.basenode import BaseNode

# Create a generic type T that extends the base node type
BaseNodeT = TypeVar('BaseNodeT', bound=BaseNode)


class IRNodeVisitor(Generic[BaseNodeT]):
    """
    A node visitor base class that walks the AST/IR node tree and visits children
    using named functions. See ``ast.NodeVisitor`` for more information.

    The class also performs optional language validation checks by passing in the base IR node class.
    """

    def __init__(self, ir_node_class: type[BaseNodeT] = BaseNode, reverse: bool = False):
        self.base_node = ir_node_class
        self.reverse = reverse

    def _validate_node_type(self, node: BaseNodeT):
        if self.base_node is not BaseNode and isinstance(node, BaseNode) and not isinstance(node, self.base_node):
            raise TypeError(f'Node {node} does not match the IR language with base node {self.base_node}')

    def visit(self, node: BaseNodeT):
        """
        Visit a node.
        If you have a node with a ``visit_NodeName`` method
        where ``NodeName`` is the name of the node class, it will be called.
        """
        self._validate_node_type(node)
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: BaseNodeT):
        """
        Recursively visit all children of the node.
        Called if no explicit visitor function exists for a node.
        If a visitor function exists for a given
        node type, it MUST call ``self.generic_visit(node)`` to visit its children.
        """
        if isinstance(node, BaseNode):
            for _, value in node.iter_fields():
                if isinstance(value, (list, tuple)):
                    self.generic_visit_sequence(value)
                elif isinstance(value, BaseNode):
                    self.visit(value)

    def generic_visit_sequence(self, sequence: Sequence[BaseNodeT]):
        if self.reverse:
            sequence = reversed(sequence)
        for item in sequence:
            if isinstance(item, (list, tuple)):
                self.generic_visit_sequence(item)
            elif isinstance(item, BaseNode):
                self.visit(item)


class ScopedIRNodeVisitor(IRNodeVisitor[BaseNodeT]):
    """
    A scoped version of the IR node visitor that contains methods to locate the current scope,
    add pre/post visitors for each scope, as well as a visitor for the contents of each scope.

    Extending a scope visitor can be done by extending ``visit_<NODE TYPE>`` normally, but for
    each scope node type, more methods are exposed for scope analysis: ``pre_visit_<NODE TYPE>``,
    ``do_visit_<NODE TYPE>``, and ``post_visit_<NODE TYPE>``. For proper scope management,
    only override ``do_visit_<NODE TYPE>`` for scope node types.
    """

    _current_scope: list[BaseNodeT]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_scope = []

    def _do_nothing(self, node):
        pass

    def _visit_ScopeNode(self, node: BaseNodeT):
        """
        A generic method that visits a scope of a specific type.
        """
        typestr = type(node).__name__

        # Pre-visit, if exists
        getattr(self, f'pre_visit_{typestr}', self._do_nothing)(node)
        # Setup scope
        self.push_scope(node)
        # Visit scope contents
        result = getattr(self, f'do_visit_{typestr}', self.generic_visit)(node)
        # Pop scope
        self.pop_scope()
        # Post-visit, if exists
        getattr(self, f'post_visit_{typestr}', self._do_nothing)(node)

        # Return result from visiting scope contents
        return result

    def get_scope(self) -> BaseNodeT:
        return self._current_scope[-1]

    def push_scope(self, scope) -> None:
        self._current_scope.append(scope)

    def pop_scope(self) -> BaseNodeT:
        return self._current_scope.pop()

    def get_scope_with_type(self, type_var) -> BaseNodeT | None:
        """
        Returns the closest enclosing scope with the given type or None if not found.
        :param type_var:
        :return:
        """
        for scope in reversed(self._current_scope):
            if isinstance(scope, type_var):
                return scope
        return None


class IRNodeTransformer(IRNodeVisitor[BaseNodeT]):
    """
    A ``NodeVisitor`` subclass that walks the AST/IR and allows modification of nodes.
    See ``ast.NodeTransformer`` for more information.
    """

    def generic_visit(self, node: BaseNodeT):
        for field, old_value in node.iter_fields():
            if isinstance(old_value, (list, tuple)):
                new_values = self.generic_visit_sequence(old_value)
                setattr(node, field, new_values)
            elif isinstance(old_value, BaseNode):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def generic_visit_sequence(self, sequence: Sequence[BaseNodeT]):
        new_values = []
        for value in sequence:
            if isinstance(value, BaseNode):
                value = self.visit(value)
                if value is None:
                    continue
                elif isinstance(value, (list, tuple)):
                    new_values.extend(value)
                    continue
            elif isinstance(value, (list, tuple)):
                value = self.generic_visit_sequence(value)
            new_values.append(value)

        if isinstance(sequence, tuple):
            return tuple(new_values)
        return new_values
