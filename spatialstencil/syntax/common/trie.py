from typing import List, Dict, Optional, TypeVar, Generic
from collections import deque

T = TypeVar('T')


class TrieNode(Generic[T]):
    """
    A node in the Trie data structure.
    """
    def __init__(self, parent: Optional['TrieNode[T]'], identifier: int = None):
        self.arrows: Dict[T, 'TrieNode[T]'] = {}
        self.parent: Optional['TrieNode[T]'] = parent
        self.suffix: Optional['TrieNode[T]'] = None
        self.output: Optional['TrieNode[T]'] = None
        self.pattern: Optional[List[T]] = None
        self.identifier = identifier

    def add(self, symbol: T, target: 'TrieNode[T]'):
        assert isinstance(target, TrieNode)
        self.arrows[symbol] = target

    def goto_on(self, symbol: T) -> 'TrieNode[T]':
        node = self
        while symbol not in node.arrows:
            node = node.suffix
            if node.parent is None:
                break
        return node.arrows.get(symbol, node)

    def get_outputs(self) -> List[List[T]]:
        outputs = []
        if self.pattern is not None:
            outputs.append(self.pattern)

        link = self.output
        while link is not None:
            if link.pattern is not None:
                outputs.append(link.pattern)
            link = link.output

        return outputs

    def __str__(self):
        return f'{self.identifier} : {[(str(a), b.identifier) for a, b in self.iterate_arrows(self.arrows)]}'


class Trie(Generic[T]):
    """
    A Trie data structure.

    Supports the Aho-Corasick automaton construction for string matching.
    By representing trees as strings, we can use it to match trees as well.
    """

    _id_count: int

    def __init__(self):
        self.root = TrieNode[T](None, 0)
        self._id_count = 0

    def get_root(self) -> TrieNode[T]:
        return self.root

    def add(self, pattern: List[T]):
        node = self.root
        for s in pattern:
            if s not in node.arrows:
                node.arrows[s] = TrieNode(node, self._id_count)
                self._id_count += 1
            node = node.arrows[s]
            assert node is not None
        node.pattern = pattern[:]

    @staticmethod
    def iterate_sorted(dictionary: Dict[T, 'TrieNode[T]']):
        for k, v in sorted(dictionary.items(), key=lambda x: x[0]):
            yield k, v

    def compute_automaton_links(self):
        q = deque[T, TrieNode[T]]()

        # root and its children have root as their suffix link
        self.root.suffix = self.root
        for initial, child in self.iterate_sorted(self.root.arrows):
            child.suffix = self.root
            for symbol, grandchild in self.iterate_sorted(child.arrows):
                q.append((symbol, grandchild))

        while q:
            symbol, node = q.popleft()

            suffix: TrieNode = node.parent.suffix
            while symbol not in suffix.arrows:
                suffix = suffix.suffix
                if suffix == self.root:
                    break

            node.suffix = suffix.arrows.get(symbol, self.root)
            node.output = node.suffix if node.suffix.pattern is not None else node.suffix.output

            for sym, child in self.iterate_sorted(node.arrows):
                q.append((sym, child))

    def __str__(self):
        return str(self.root)


class TrieBuilder(Generic[T]):
    """
    Wrapper class to build a Trie data structure and its associated Aho-Corasick automaton.
    """
    def __init__(self):
        self.trie = Trie[T]()

    def add(self, pattern: List[T]) -> 'TrieBuilder[T]':
        self.trie.add(pattern)
        return self

    def build(self) -> Trie[T]:
        self.trie.compute_automaton_links()
        return self.trie
