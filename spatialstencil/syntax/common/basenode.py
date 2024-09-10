"""
Provides a base class for AST/IR trees. Includes children queries and schema validation.
"""
from dataclasses import dataclass
from collections import deque
import pprint


@dataclass
class BaseNode:
    """
    Base class for AST/IR nodes.

    The rules for an appropriate structure are:

      * All children must be provided as dataclass fields (e.g., ``child: ChildType``). Otherwise,
        they will not be traversed.
      * Order must be deterministic (this prohibits the use of containers such as unordered dicts)
      * Child types must be one of three:
        * Terminator type (i.e., that cannot contain another node subclass). This is usually a
          constant or literal, such as int/float/str/None.
        * Another AST/IR node.
        * An ordered sequence of AST/IR nodes or terminator types. This is usually limited to a list
          sequence.
      * There cannot be multiply nested sequences (e.g., ``list[list[Node]])``). For that, use
        another AST/IR node in between (e.g., ``list[NodeSequence]`` and ``NodeSequence := list[Node]``).

    Basic rule validation occurs in ``validate_schema``.
    """

    @classmethod
    def validate_schema(cls, visited: set[type['BaseNode']] = None) -> None:
        """
        Validates that the node type and all its child node types abide by
        the rules defined on ``BaseNode``.

        Raises exceptions on failure.
        """
        visited = visited or set()
        if cls in visited:  # Skip type cycles
            return
        visited.add(cls)

        for field_name, field in cls.__dataclass_fields__.items():
            if issubclass(field.type, BaseNode):
                # Traverse
                field.type.validate_schema(visited)
            elif hasattr(field.type, '__origin__') and issubclass(field.type.__origin__, (tuple, list)):  # Sequence
                # Check that contents must be either a terminator or a node
                for subtype in field.type.__args__:
                    if issubclass(subtype, BaseNode):
                        subtype.validate_schema(visited)
                    if not issubclass(subtype, (BaseNode, int, float, str, type(None))):
                        raise TypeError(f'Unsupported sequence content "{subtype}" for field {field_name} of {cls}')
            else:
                if not issubclass(field.type, (int, float, str, type(None))):
                    raise TypeError(f'Unsupported terminator type "{field.type}" for field {field_name} of {cls}')

    def validate(self) -> None:
        """
        Validates the contents of this specific node. This function is to be overridden
        by subclasses.
        """
        pass  # Nothing to validate

    def __post_init__(self):
        self.validate()

    def pretty(self) -> str:
        """
        Pretty-prints the contents of this node.
        """
        return pprint.pformat(self)

    def iter_fields(self):
        """
        Yield a tuple of ``(fieldname, value)`` for each field in the dataclass.
        """
        for field in self.__dataclass_fields__:
            try:
                yield field, getattr(self, field)
            except AttributeError:
                pass

    def iter_child_nodes(self, ir_node_class: type['BaseNode'] = None):
        """
        Yield all direct child AST/IR nodes of node.
        """
        ir_node_class = ir_node_class or BaseNode
        for _, field in self.iter_fields():
            if isinstance(field, BaseNode):
                yield field
            elif isinstance(field, (tuple, list)):
                for item in field:
                    if isinstance(item, BaseNode):
                        yield item

    def walk(self):
        """
        Recursively yield all descendant nodes in the tree starting at ``self``
        (including the node itself), in breadth-first order. This function is
        based on ``ast.walk``.
        """
        todo = deque([self])
        while todo:
            node = todo.popleft()
            todo.extend(node.iter_child_nodes())
            yield node
