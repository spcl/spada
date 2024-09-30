from collections import deque
from dataclasses import dataclass
from typing import List, Union, Deque, TypeVar, Any

from spatialstencil.syntax.common.basenode import BaseNode

import spatialstencil.syntax.common.basenode as syntax

V = TypeVar('V')


# Abstract Tree class
class MatchTree[V]:
    pass


# Node class representing a node in the tree
class TreeNode(MatchTree[V]):
    def __init__(self, label: V, children: List[MatchTree[V]]):
        self.label = label
        self.children = children

    def get_label(self) -> V:
        return self.label

    def get_children(self) -> List[MatchTree[V]]:
        return self.children

    def __str__(self):
        if not self.children:
            return self.label

        subtrees = ",".join(str(child) for child in self.children)
        node = f"{self.label}({subtrees})"
        return node

    def match_string(self, match: dict['TreeNode', bool]):
        if not self.children:
            return self.label

        subtrees = ",".join(child.match_string(match) for child in self.children)
        node = f"{self.label}({subtrees})"
        return f"[{node}]" if self in match else node


# Wildcard class representing a wildcard in the tree
class TreeWildcard(MatchTree):
    def __str__(self):
        return "_"


NVar = TypeVar('NVar', bound=BaseNode)


class MatchingBaseNode[NVar](TreeNode[str]):

    # The base node that this match tree represents
    base_node: NVar = None

    def __init__(self, label: str, children: List[MatchTree[str]], base_node: NVar | None = None):
        super().__init__(label, children)
        self.base_node = base_node

    @staticmethod
    def from_base_node(node: NVar) -> TreeNode[str]:
        # Iterate over all fields of the node
        # for each string, int, or enum field, create a child node directly
        # for each basenode, create a child node for each field recursively
        # for each list of basenodes, create a child node for each element recursively
        # for each list of strings, ints, or enums, create a child node for each element directly
        # the label of the node is the name of the class of the node

        def _create_wildcard_node(_wildcard: syntax.Wildcard) -> TreeNode[str]:
            _grandchild = TreeWildcard()
            _type = _wildcard.get_type()
            if _type != Any:
                _type_label = _type.__name__
                _child = MatchingBaseNode(f"{_type_label}", [_grandchild])
            else:
                _child = _grandchild
            return _child

        def _create_primitive_node(_value: Any) -> TreeNode[str]:
            _class_name = _value.__class__.__name__
            _grandchild = TreeNode(str(_value), [])
            return TreeNode(_class_name, [_grandchild])

        label = node.__class__.__name__
        children = []
        for f in node.iter_fields():
            if isinstance(f[1], syntax.Wildcard):
                child = _create_wildcard_node(f[1])
                children.append(child)
            elif isinstance(f[1], BaseNode):
                children.append(MatchingBaseNode.from_base_node(f[1]))
            elif isinstance(f[1], (list, tuple)):
                for i, elem in enumerate(f[1]):
                    # Make sure the order is respected
                    if isinstance(elem, syntax.Wildcard):
                        grandchild = _create_wildcard_node(elem)
                    elif isinstance(elem, BaseNode):
                        grandchild = MatchingBaseNode.from_base_node(elem)
                    else:
                        grandchild = _create_primitive_node(elem)
                    child = MatchingBaseNode(str(i), [grandchild])
                    children.append(child)
            else:
                child = _create_primitive_node(f[1])
                children.append(child)

        return MatchingBaseNode[NVar](label, children, node)


# Abstract base class for Symbol
class Symbol[V]:
    pass


# Index class representing an index symbol in the automaton
@dataclass(frozen=True)
class Index(Symbol[V]):
    value: int

    def __str__(self):
        return str(self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, Index) and self.value == other.value


# Label class representing a label symbol in the automaton
@dataclass(frozen=True)
class Label(Symbol[V]):
    value: V

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, Label) and self.value == other.value


# Wildcard class representing a wildcard in the tree
class SymbolWildcard(Symbol):
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, SymbolWildcard)

    def __hash__(self):
        return hash('Wildcard')

    def __str__(self):
        return '_'


# Helper function to collect paths from root to leaf
def _root_to_leaf(root: MatchTree, acc: Deque[Union[Label, Index]],
                  paths: List[List[Union[Label, Index]]]):
    if isinstance(root, TreeNode):
        # if the node is a leaf, collect it
        if not root.get_children():
            path = deque(acc)
            path.appendleft(Label(root.get_label()))
            paths.append(list(reversed(path)))
            return

        # collect paths down children depth-first
        for i, child in enumerate(root.get_children()):
            acc.appendleft(Label(root.get_label()))
            acc.appendleft(Index(i))
            _root_to_leaf(child, acc, paths)
            acc.popleft()
            acc.popleft()

    elif isinstance(root, TreeWildcard):
        # wildcards are leafs, collect up to them
        path = deque(acc)
        paths.append(list(reversed(path)))


# Wrapper function to collect all root-to-leaf paths
def root_to_leaf_paths(root: MatchTree) -> List[List[Union[Label, Index]]]:
    paths = []
    _root_to_leaf(root, deque(), paths)
    return paths
