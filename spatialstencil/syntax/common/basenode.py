"""
Provides a base class for AST/IR trees. Includes children queries and schema validation.
"""
import inspect
import types
import typing
import warnings
from dataclasses import dataclass
from collections import deque
import pprint
from enum import Enum
from typing import Generic



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
    def validate_schema(cls, visited: set[type['BaseNode']] = None):
        """
        Validates that the node type and all its child node types abide by
        the rules defined on ``BaseNode``.

        Raises exceptions on failure.
        """
        visited = visited or set()
        if cls in visited:  # Skip type cycles
            return True
        visited.add(cls)

        def _check_sequence(sequence, f_name):

            for item in typing.get_args(sequence):
                if isinstance(item, str):
                    module = inspect.getmodule(cls)
                    if not hasattr(module, item):
                        warnings.warn(f"Could not validate schema for field {f_name} of {cls} due to forward reference")
                        continue
                    item = getattr(module, item)  # Try to obtain class from the same module

                if typing.get_origin(item) is types.UnionType or typing.get_origin(item) is typing.Union:
                    _check_union(item, field_name)
                elif issubclass(item, BaseNode):
                    item.validate_schema(visited)
                elif not isinstance(item, type) or not issubclass(item, (int, float, str, type(None), Enum)):
                    raise TypeError(f'Unsupported sequence content {item} for field {f_name} of {cls}')

        def _check_union(union, f_name):
            for subtype in typing.get_args(union):
                # Handle Literal or None types
                if typing.get_origin(subtype) is typing.Literal or subtype is type(None):
                    continue
                if isinstance(subtype, str):
                    warnings.warn(f"Could not validate schema for field {f_name} of {cls} due to forward reference")
                    continue
                if isinstance(typing.get_origin(subtype), (list, type)):
                    _check_sequence(subtype, f_name)
                # Handle BaseNode subclasses
                elif isinstance(subtype, type) and issubclass(subtype, BaseNode):
                    subtype.validate_schema(visited)
                # Raise error for unsupported types
                elif not isinstance(subtype, type) or not issubclass(subtype,
                                                                   (BaseNode, int, float, str, type(None), Enum)):
                    raise TypeError(f'Unsupported union type {subtype} for field {f_name} of {cls}')

        # Use get_type_hints to resolve forward references
        type_hints = typing.get_type_hints(cls)

        for field_name, field in cls.__dataclass_fields__.items():
            # Resolve the field's type using get_type_hints (handling forward references)
            field_type = type_hints[field_name]
            origin = typing.get_origin(field_type)
            if origin is types.UnionType or origin is typing.Union:
                _check_union(field_type, field_name)
            elif isinstance(field_type, type) and issubclass(field_type, BaseNode):
                # Traverse if it’s a BaseNode subclass
                field_type.validate_schema(visited)
            elif origin and issubclass(origin, (tuple, list)):  # Sequence
                # Check contents of sequences
                _check_sequence(field_type, field_name)
            else:
                if not isinstance(field_type, type) or not issubclass(field_type, (int, float, str, type(None), Enum)):
                    raise TypeError(f'Unsupported terminator type {field_type} for field {field_name} of {cls}')

    def validate(self) -> None:
        """
        Validates the contents of this specific node. This function is to be overridden
        by subclasses.
        """
        pass  # Nothing to validate

    def __post_init__(self):
        # Check if there are any wildcards in the node, if so, skip validation
        for _, field in self.iter_fields():
            if isinstance(field, Wildcard):
                return

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

    def iter_child_nodes(self):
        """
        Yield all direct child AST/IR nodes of node.
        """
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


T = typing.TypeVar('T', bound=typing.Any)


class Wildcard(Generic[T], BaseNode):
    """
    Represents a wildcard in the tree.
    """

    def __init__(self, name: str = ''):
        self.name = name

    def __str__(self):
        return f'Wildcard[{self.get_type().__name__}]({self.name})'

    def __repr__(self):
        return str(self)

    def bind(self) -> typing.Any:
        """
        Masks the wildcard type, allowing it to be used
        as part of the base node structure.

        :return:
        """
        return self

    def __call__(self, *args, **kwargs) -> typing.Any:
        """
        Masks the wildcard type, allowing it to be used
        as part of the base node structure.

        :return:
        """
        return self

    def get_type(self):
        """
        Gets the type restriction of the wildcard.
        If no type restriction is provided, defaults to Any.

        :return: The type restriction of the wildcard.
        """

        if hasattr(self, '__orig_class__'):
            return self.__orig_class__.__args__[0]
        else:
            return typing.Any  # Fallback to Any if no type argument is provided
