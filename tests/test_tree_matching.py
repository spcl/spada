from dataclasses import dataclass

import unittest
from spatialstencil.syntax.common.match_tree import TreeNode, TreeWildcard, MatchTree, MatchingBaseNode
from spatialstencil.syntax.common.tree_matching import _match_pattern, PatternMatcher, PatternTransformer
from typing import Tuple, List, TypeVar, Generic

import spatialstencil.syntax.stencil_ir.irnodes as sast
from spatialstencil.syntax.common.basenode import Wildcard
from spatialstencil.syntax.common.types import ScalarType


# Assume Tree, Node, Wildcard classes are already defined from previous translations.
class Parser:
    @staticmethod
    def parse(input_str: str) -> MatchTree[str]:
        # Strip whitespace for simplicity
        input_str = input_str.replace(" ", "")
        return Parser.parse_expression(input_str, 0)[0]

    @staticmethod
    def parse_expression(input_str: str, i: int) -> Tuple[MatchTree[str], int]:
        # Return None if input is exhausted
        if i >= len(input_str) - 1:
            return None, i

        while i < len(input_str):
            current = input_str[i]

            # Are we parsing an identifier?
            if current.isalpha():
                builder = [current]

                # Consume rest of the identifier
                j = i + 1
                while j < len(input_str) and input_str[j].isalpha():
                    builder.append(input_str[j])
                    j += 1
                i = j

                # Lookahead to see if we can expect a list of expressions
                if i < len(input_str) and input_str[i] == '(':
                    i += 1

                    # Parse a non-empty list of subtree expressions
                    children, i = Parser.parse_list(input_str, i)
                    if i < len(input_str) and input_str[i] == ')':
                        return TreeNode(''.join(builder), children), i + 1

                # No subtrees, return leaf node
                return TreeNode(''.join(builder), []), i

            elif current == '_':
                return TreeWildcard(), i + 1

        # Shouldn't reach here, return None
        return None, i

    @staticmethod
    def parse_list(input_str: str, i: int) -> Tuple[List[MatchTree[str]], int]:
        children = []

        # Parse the first expression
        first_expr, i = Parser.parse_expression(input_str, i)
        children.append(first_expr)

        # Keep parsing the rest of the list
        while i < len(input_str) and input_str[i] == ',':
            i += 1
            next_expr, i = Parser.parse_expression(input_str, i)
            children.append(next_expr)

        return children, i


class IdentifierIncrementerTransformer(PatternTransformer[sast.BaseNode, sast.Identifier]):

    def __init__(self):
        pattern = sast.Expression(sast.Identifier(Wildcard[str]()(), Wildcard[int]("version")()))
        super().__init__([pattern])

    def transform(self, root: sast.Expression, version: int = None, **wildcards) -> sast.Identifier:
        assert version is not None
        return sast.Identifier(root.value.name, version + 1)


class TestTreeMatching(unittest.TestCase):

    def test_match(self):
        pattern = "a(a(b, _), c)"
        subject = "f(a(a(b, a(a(b, a(a(b, y), c)), c)), c), z)"
        # Parse pattern and subject trees into trees for matching
        print(pattern)
        print(subject)
        pattern_tree = Parser.parse(pattern)
        subject_tree = Parser.parse(subject)
        has_match = _match_pattern(pattern_tree, subject_tree)

        # Print the match
        self._print_match(subject_tree, has_match)

    def _print_match(self, subject_tree, has_match):
        colours = ["\u001B[31m", "\u001B[33m", "\u001B[32m", "\u001B[33m", "\u001B[36m", "\u001B[35m"]
        depth = -1
        subject_str = subject_tree.match_string(has_match)
        match_count = 0
        for c in subject_str:
            if c == '[':
                depth += 1
                match_count += 1
            elif c == ']':
                depth -= 1
            color = colours[depth % len(colours)] if depth >= 0 else "\033[0m"
            print(f"{color}{c}", end="")
        print("\u001B[0m")
        return match_count

    def test_expression_matching(self):

        e = sast.Expression(sast.BinaryOperator(sast.Expression(sast.Identifier("a", 0)),
                                                "+",
                                                sast.Expression(sast.Identifier("b", 1))))

        pattern = sast.BinaryOperator(Wildcard("left")(), "+", Wildcard("right")())

        matcher = PatternMatcher(pattern)

        match = matcher.match_pattern(e)

        assert len(match) == 1

        e = sast.Expression(sast.BinaryOperator(sast.Expression(sast.Identifier("a", 0)),
                                                "+",
                                                sast.Expression(
                                                    sast.BinaryOperator(sast.Expression(sast.Identifier("b", 0)),
                                                                        "+",
                                                                        sast.Expression(1)))))

        match = matcher.match_pattern(e)

        assert len(match) == 2

    def test_expression_wildcard(self):

        e = sast.Expression(sast.BinaryOperator(sast.Expression(sast.Identifier("a", 0)),
                                                "+",
                                                sast.Expression(
                                                    sast.BinaryOperator(sast.Expression(sast.Identifier("b", 0)),
                                                                        "+",
                                                                        sast.Expression(1)))))

        pattern = sast.BinaryOperator(sast.Expression(sast.Identifier("b", 0)),
                                      "+",
                                      sast.Expression(Wildcard[int]("right").bind()))

        matcher = PatternMatcher(pattern)

        match = matcher.match_pattern(e)

        assert len(match) == 1

        assert "right" in match[0].wildcards
        assert match[0].wildcards["right"] == 1

        pattern = sast.BinaryOperator(sast.Expression(Wildcard("left").bind()),
                                      "+",
                                      sast.Expression(Wildcard[int]("right").bind()))

        print(pattern)

        matcher = PatternMatcher(pattern)

        match = matcher.match_pattern(e)

        assert len(match) == 1

        assert "right" in match[0].wildcards
        assert match[0].wildcards["right"] == 1

        assert "left" in match[0].wildcards
        assert match[0].wildcards["left"].name == "b"
        assert match[0].wildcards["left"].version == 0

        pattern = sast.Identifier(Wildcard("id").bind(), 0)

        matcher = PatternMatcher(pattern)

        match = matcher.match_pattern(e)

        assert len(match) == 2

        for m in match:
            assert m.wildcards["id"] == "a" or m.wildcards["id"] == "b"

    def test_expression_transform(self):

        e = sast.Expression(sast.BinaryOperator(sast.Expression(sast.Identifier("a", 0)),
                                                "+",
                                                sast.Expression(
                                                    sast.BinaryOperator(sast.Expression(sast.Identifier("b", 4)),
                                                                        "+",
                                                                        sast.Expression(1)))))

        transformer = IdentifierIncrementerTransformer()

        result = transformer.apply(e)

        print(result)

        for r in result:
            assert r.name == "a" or r.name == "b"
            if r.name == "a":
                assert r.version == 1
            else:
                assert r.version == 5

    def test_expresison_ordering(self):

        e = sast.Expression(sast.BinaryOperator(sast.Expression(sast.Identifier("a", 0)),
                                                "+",
                                                sast.Expression(sast.Identifier("b", 1))))

        pattern = sast.BinaryOperator(sast.Expression(sast.Identifier("b", 0)),
                                      "+",
                                      sast.Expression(Wildcard[float]("right").bind()))

        matcher = PatternMatcher(pattern)

        match = matcher.match_pattern(e)

        assert len(match) == 0

        pattern = sast.BinaryOperator(Wildcard("left").bind(), "+", sast.Expression(sast.Identifier("b", 0)))

        matcher = PatternMatcher(pattern)

        match = matcher.match_pattern(e)

        assert len(match) == 0

    def test_assign_bind(self):

        pattern = sast.AssignOp(Wildcard("dst")(), sast.Expression(1), Wildcard()())

        e = sast.AssignOp(sast.Identifier("a", 0), sast.Expression(1))

        matcher = PatternMatcher(pattern)

        match = matcher.match_pattern(e)

        assert len(match) == 1

        assert "dst" in match[0].wildcards
        assert match[0].wildcards["dst"].name == "a"

        e = sast.Expression(
            sast.BinaryOperator(
                sast.Expression(sast.UnaryOperator(Wildcard("u_op")(), sast.Expression(Wildcard[float]("value")()))),
                Wildcard("op")(),
                sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0])),
            ))
        pattern = sast.AssignOp(Wildcard("dst")(), e, Wildcard()())

        e = sast.AssignOp(result=sast.Identifier(name='a', version=0), value=sast.Expression(
            value=sast.BinaryOperator(
                left=sast.Expression(value=sast.UnaryOperator(op='-', value=sast.Expression(value=4.0))), op='*',
                right=sast.Expression(
                    value=sast.Subscript(value=sast.Identifier(name='in', version=0), subscript=[0, 0, 0])))),
                          operation_type=sast.OperationType([ScalarType.f32], destination=None))

        matcher = PatternMatcher(pattern)

        match = matcher.match_pattern(e)

        assert len(match) == 1
        assert "dst" in match[0].wildcards
        assert "u_op" in match[0].wildcards
        assert "op" in match[0].wildcards
        assert "src" in match[0].wildcards
        assert "value" in match[0].wildcards

        assert match[0].wildcards["dst"].name == "a"
        assert match[0].wildcards["dst"].version == 0
        assert match[0].wildcards["u_op"] == "-"
        assert match[0].wildcards["op"] == "*"
        assert match[0].wildcards["src"].name == "in"
        assert match[0].wildcards["src"].version == 0
        assert match[0].wildcards["value"] == 4.0

    def test_patterns(self):

        return_pattern = sast.ReturnOp(
            [sast.Expression(sast.Identifier(Wildcard[str]("dest_name").bind(), Wildcard[str]("dest_version").bind()))]
        )

        print(return_pattern)

        pattern_matcher = PatternMatcher(return_pattern)

        assign_pattern = sast.AssignOp(
            sast.Identifier(Wildcard('dest_name').bind(), Wildcard[int]("dest_version").bind()),
            sast.Expression(
                sast.BinaryOperator(
                    sast.Expression(sast.Identifier(Wildcard("source_name").bind(), Wildcard("source_version").bind())),
                    Wildcard("operator").bind(),
                    sast.Expression(Wildcard[int]("int_literal").bind()),
                )
            ))

        pattern_matcher = PatternMatcher(assign_pattern)

        print(assign_pattern)


if __name__ == '__main__':
    unittest.main()
