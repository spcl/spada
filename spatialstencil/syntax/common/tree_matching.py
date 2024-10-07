import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic

from spatialstencil.syntax.common.basenode import BaseNode, Wildcard
from spatialstencil.syntax.common.match_tree import root_to_leaf_paths, TreeNode, Symbol, Index, Label, MatchingBaseNode
from spatialstencil.syntax.common.trie import TrieBuilder, TrieNode, Trie
from collections import deque, defaultdict


BaseNodeT = TypeVar('BaseNodeT', bound=BaseNode)
BaseNodeK = TypeVar('BaseNodeK', bound=BaseNode)


@dataclass(frozen=True)
class Match:
    """
    A match contains the root node of the match and the wildcards that were matched in the subtree.
    """
    root: BaseNode
    wildcards: dict[str, BaseNode]


class PatternMatcher(Generic[BaseNodeT]):
    """
    Pattern matcher for trees.

    The pattern matcher can be used to match a pattern tree to a subject tree.
    All subtrees in the subject tree that match the pattern tree are returned.
    Matches check for the structure and the labels of the nodes in the pattern tree.
    """
    def __init__(self, pattern: BaseNodeT):
        assert not isinstance(pattern, Wildcard), "Root node cannot be a wildcard (for now)"

        pattern_tree = MatchingBaseNode.from_base_node(pattern)
        trie, paths = _build_trie(pattern_tree)
        self.trie = trie
        self.paths = paths
        self.pattern = pattern

    def match_pattern(self, subject: BaseNodeT) -> list[Match]:
        """
        Return a list of matches for the pattern in the subject tree.
        All subtrees in the subject tree that match the pattern tree are returned,
        Matches check for the structure and the labels of the nodes in the pattern tree.
        The match contains the root node of the match and the wildcards
        that were matched, which are stored in a dictionary.

        :param subject: the subject tree
        :return: list of matches
        """
        matches = self._match_pattern(subject)

        result = []
        for match in matches:
            wildcard_matches = self._collect_wildcards(self.pattern, match.base_node)
            result.append(Match(root=match.base_node, wildcards=wildcard_matches))

        return result

    def _collect_wildcards(self, pattern_node: BaseNodeT, subject_node: BaseNodeT) -> dict[str, BaseNodeT]:
        """
        Collect named wildcards from a pattern and a subject node.

        :param pattern_node:
        :param subject_node:
        :return:
        """
        wildcard_matches = {}
        self._collect_named_wildcards(pattern_node, subject_node, wildcard_matches)
        return wildcard_matches

    def _collect_named_wildcards(self,
                                 pattern_node: BaseNodeT | typing.Sequence[BaseNodeT],
                                 subject: BaseNodeT | float | int | str | bool | list | tuple,
                                 wildcard_matches: dict) -> None:
        """
        Recursively collect named wildcards from a pattern and a subject node.

        :param pattern_node:
        :param subject:
        :param wildcard_matches:
        :return:
        """
        if isinstance(pattern_node, Wildcard):
            wildcard_name = pattern_node.name
            if wildcard_name and len(wildcard_name):
                wildcard_matches[wildcard_name] = subject
        elif isinstance(subject, BaseNode):
            # Collect all the fields of the pattern and subject:
            subject_dict = {field_name: field for field_name, field in subject.iter_fields()}
            for field_name, pattern_field in pattern_node.iter_fields():
                if field_name in subject_dict:
                    subject_field = subject_dict[field_name]
                    self._collect_named_wildcards(pattern_field, subject_field, wildcard_matches)
        elif isinstance(subject, (list, tuple)):
            assert isinstance(pattern_node, typing.Sequence)
            # Collect wildcards from each element of the sequence
            for pattern_field, subject_field in zip(pattern_node, subject):
                self._collect_named_wildcards(pattern_field, subject_field, wildcard_matches)

    def _match_pattern(self, subject: BaseNodeT) -> list[MatchingBaseNode]:
        """
        Wraps the pattern matching algorithm and returns the matching nodes.

        :param subject:
        :return:
        """
        subject_tree = MatchingBaseNode.from_base_node(subject)
        matches = _match_pattern(None, subject_tree, self.paths, self.trie)
        return matches  # type: ignore


ContextT = TypeVar('ContextT')


class PatternTransformer(Generic[BaseNodeT, BaseNodeK, ContextT], ABC):
    """
    Abstract base class for pattern transformers.

    The transformer can be used to apply a transformation to a tree based
    on a pattern match. The pattern match is determined by the pattern
    tree that is passed to the transformer. The transformer can be used
    to apply the transformation to the first match or all matches.
    """

    context: ContextT

    def __init__(self, patterns: list[BaseNodeT]):
        self.patterns = patterns
        self.matchers = [PatternMatcher(pattern) for pattern in patterns]

    def set_context(self, context: ContextT) -> None:
        """
        Set the context for the transformer. This can be used to pass
        additional information to the transform method.

        :param context:
        :return:
        """
        self.context = context

    def get_context(self) -> ContextT:
        """
        Get the context for the transformer.

        :return:
        """
        return self.context

    def first(self, subject: BaseNodeT) -> list[BaseNodeK]:
        """
        Apply the first pattern that matches the subject

        :param subject: match against this subject
        :return: transformed nodes (if any, otherwise empty list)
        """
        for matcher in self.matchers:
            matches = matcher.match_pattern(subject)
            if matches:
                return self.transform(matches[0].root, **matches[0].wildcards)
        return []

    def match(self, subject: BaseNodeT) -> list[Match]:
        """
        Match all patterns that match the subject, returning the nodes
        at which a transformation is possible.

        :param subject:
        :return: list of matches
        """

        matches = []
        for matcher in self.matchers:
            matches.extend(matcher.match_pattern(subject))
        return matches

    def apply(self, subject: BaseNodeT) -> list[list[BaseNodeK]]:
        """
        Apply all patterns that match the subject

        :param subject:
        :return: list of transformed nodes
        """

        matches = []

        for matcher in self.matchers:
            matches.extend(matcher.match_pattern(subject))
        # Create a PatternMatcher and match the pattern

        # For each match, call the transform function
        result = []
        for match in matches:
            result.append(self.transform(match.root, **match.wildcards))
        return result

    def transform(self, root: BaseNodeT, **wildcards: BaseNodeT) -> list[BaseNodeK]:
        """
        This method must be implemented by subclasses to provide
        specific transformations for the pattern.

        :param root: the root node of the match
        :param wildcards: the wildcards that were matched. Each argument is named after the wildcard name
        and contains the matched node. This provided a convenient way to access the matched nodes.
        :return: the transformed nodes, if any, or an empty list if no transformation is applied.
        """
        pass


def _match_pattern(pattern: TreeNode | None, subject: TreeNode, paths=None, trie=None) -> list[TreeNode]:
    """
    Match a pattern tree to a subject tree using the approach by Hoffmann and O’Donnell
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

    return [t for t in subject.walk_tree() if has_match[t]]


def _build_trie(pattern: TreeNode) -> tuple[Trie[Symbol], list[list[Symbol]]]:
    """
    Build a trie of all root-to-leaf paths and an associated Aho-Corasick automaton from a pattern tree.
    It returns the trie (which encapsulates an Aho-Corasick automaton) and the paths from the root to the leaf nodes.

    :param pattern:
    :return:
    """

    assert isinstance(pattern, TreeNode)
    assert isinstance(pattern, TreeNode)
    # Construct Aho-Corasick automaton from pattern tree
    paths = root_to_leaf_paths(pattern)
    builder = TrieBuilder[Symbol]()
    for path in paths:
        assert all(isinstance(p, Symbol) for p in path)
        builder.add(path)

    # Build Aho-Corasick automaton
    trie = builder.build()
    return trie, paths


@dataclass
class Entry:
    """
    Matching algorithm stack entry for internal pre-order book-keeping.
    """
    node: TreeNode
    state: TrieNode[Symbol]
    visited: int
