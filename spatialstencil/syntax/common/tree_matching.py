from dataclasses import dataclass
from typing import TypeVar, Generic

from spatialstencil.syntax.common.basenode import BaseNode
from spatialstencil.syntax.common.match_tree import root_to_leaf_paths, TreeNode, Symbol, Index, Label, MatchingBaseNode
from spatialstencil.syntax.common.trie import TrieBuilder, TrieNode, Trie
from collections import deque, defaultdict


BaseNodeT = TypeVar('BaseNodeT', bound=BaseNode)


class PatternMatcher(Generic[BaseNodeT]):

    def __init__(self, pattern: BaseNodeT):
        pattern_tree = MatchingBaseNode.from_base_node(pattern)
        print(pattern_tree)
        trie, paths = _build_trie(pattern_tree)
        self.trie = trie
        self.paths = paths

    def match_pattern(self, subject: BaseNode) -> set[BaseNode]:
        matches = self._match_pattern(subject)
        return [m.base_node for m in matches]

    def _match_pattern(self, subject: BaseNode) -> set[TreeNode]:
        subject_tree = MatchingBaseNode.from_base_node(subject)
        print(subject_tree)
        matches = _match_pattern(None, subject_tree, self.paths, self.trie)
        return matches


def _match_pattern(pattern: TreeNode | None, subject: TreeNode, paths=None, trie=None) -> set[TreeNode]:
    """
    Math a pattern tree to a subject tree using the approach by Hoffmann and O’Donnell
    described in "Pattern Matching in Trees".

    It relies on a Aho-Corasick automaton to match the pattern tree to the subject tree.
    It returns a set of sub-tree nodes in the subject tree that match the pattern tree.

    :param pattern:
    :param subject:
    :return:
    """

    # Pattern must be rooted at a labeled node for initial transition
    assert isinstance(subject, TreeNode)

    if trie is None:
        # Build Aho-Corasick automaton
        trie, paths = _build_trie(pattern)

    # Algorithm D stack entry for pre-order book-keeping
    stack = deque[Entry]()

    counter: dict[TreeNode, int] = defaultdict(int)
    has_match: dict[TreeNode, bool] = defaultdict(bool)

    # Tabulate update counters and registers matches
    # Tabulate update counters and register matches
    def tabulate(state: TrieNode[Symbol]):
        for output in state.get_outputs():
            # Inefficient, should be precomputed and stored as match length
            match = [p for p in output if isinstance(p, Label)]
            entry = stack[-len(match)]
            node = entry.node
            counter[node] = counter[node] + 1
            has_match[node] = (counter[node] == len(paths))
            # If this node is a match, print the label of the root node
            if has_match[node]:
                print(f"Match found at root: {node.get_label()}")

    # Populate stack with initial transition
    subject_root = subject
    next_state = trie.get_root().goto_on(Label(subject_root.get_label()))
    stack.append(Entry(subject_root, next_state, -1))
    tabulate(next_state)

    # Process all subtrees
    while stack:
        top = stack[-1]
        this_node = top.node
        this_state = top.state
        visited = top.visited

        # Visited all children
        if visited >= len(this_node.get_children()) - 1:
            stack.pop()
            continue

        # Increase visitation index, initially -1 for all entries
        top.visited = visited + 1

        # Follow child subtree's index symbol
        int_state = this_state.goto_on(Index(top.visited))
        tabulate(int_state)

        # Follow child subtree, pushing it to the stack
        next_node = this_node.get_children()[top.visited]
        assert isinstance(next_node, TreeNode)
        next_state = int_state.goto_on(Label(next_node.get_label()))
        stack.append(Entry(next_node, next_state, -1))
        tabulate(next_state)

    return {t for t in counter.keys() if has_match[t]}

def _build_trie(pattern: TreeNode) -> tuple[Trie[Symbol], list[list[Symbol]]]:

    assert isinstance(pattern, TreeNode)
    assert isinstance(pattern, TreeNode)
    # Construct Aho-Corasick automaton from pattern tree
    paths = root_to_leaf_paths(pattern)
    builder = TrieBuilder[Symbol]()
    for path in paths:
        print([str(p) for p in path])
        assert all(isinstance(p, Symbol) for p in path)
        builder.add(path)

    # Build Aho-Corasick automaton
    trie = builder.build()
    return trie, paths

@dataclass
class Entry:
    node: TreeNode
    state: TrieNode[Symbol]
    visited: int

