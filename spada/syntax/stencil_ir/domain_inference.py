import itertools
import warnings
from typing import Sequence, Collection

import spada.syntax.stencil_ir.irnodes as sast
import copy

from spada.syntax.stencil_ir import def_use_analysis
from spada.syntax.stencil_ir.def_use_analysis import ScopedUse, ScopedDefinition


def infer_field_domains(program: sast.Program,
                        domain: sast.Cartesian | None = None,
                        def_use: dict[sast.Identifier, list[ScopedUse]] | None = None,
                        use_def: dict[sast.Identifier, list[ScopedDefinition]] | None = None):
    """
    Infers the domain size of a Stencil IR program by traversing it backwards.
    Operates in-place.

    :param program: The Stencil IR program to traverse.
    :param domain: An optional 3-tuple representing domain size (x, y, z). If not given, existing domain size will
                   be used or "?" will remain.
    :param def_use: A dictionary of field names to a set of uses (field type objects).
    """
    if domain is None:  # Nothing to do
        return

    if def_use is None or use_def is None:
        def_use = dict()
        use_def = dict()
        def_use_analysis.DefUseAnalysis(def_use, use_def).visit(program)

    dom_inference = DomainInference(def_use, use_def, domain)
    dom_inference.visit(program)


class DomainInference(sast.ScopedNodeVisitor):

    def __init__(self,
                 def_use: dict[sast.Identifier, list[ScopedUse]],
                 use_def: dict[sast.Identifier, list[ScopedDefinition]],
                 result_domain: sast.Cartesian):
        super().__init__(reverse=True)
        self.domain = result_domain
        self.def_use = def_use
        self.use_def = use_def

    def __post_init__(self):
        assert self.reverse

    def visit_Program(self, program: sast.Program):
        self.push_scope(program)
        # Start with outputs. Use halo for extents.
        assert isinstance(program.operation_type.destination, list)
        for field, dtype in zip(program.outputs, program.operation_type.destination):
            if dtype.domain.is_unknown():
                dtype.domain = self.domain
        self.generic_visit(program)
        # Set the input domains to the union of the domains of their uses
        for inp, inptype in zip(program.inputs, program.operation_type.source):
            if isinstance(inptype, sast.ScalarType):
                continue
            in_domains = self._domains_of_uses_in_scope(program, inp)
            in_domain = _union_domains(in_domains)
            if inptype.domain.is_unknown():
                inptype.domain = in_domain
        self.pop_scope()

    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        computation = self.get_scope_with_type(sast.ComputationBlock)
        assert computation
        assert isinstance(computation, sast.ComputationBlock)
        # Initialize materialize op types with the domain of the result
        domain = self._union_of_domains_of_uses_in_scope(computation, node.result)
        node.operation_type.destination[0].domain = copy.deepcopy(domain)

        # The input is given similarly as for a statement that accesses the offsets in the materialize op
        def_extents = self._union_of_offsets_of_in_scope_definitions(node.value, computation)

        # Compute extents that are relative to the def_extents.
        relative = sast.Extent(_offsets_relative_to_defining_offsets(node.operation_type.destination[0].extent.extents,
                                                                     def_extents))

        node.operation_type.source[0].domain = _infer_domain_from_extents(domain,
                                                                          relative)

    def visit_ReturnOp(self, node: sast.ReturnOp):
        scope = self.get_scope()
        if isinstance(scope, sast.ComputationBlock):
            # Initialize return op types with the domain of the computation block's result
            for i in range(len(node.values)):
                node.operation_type.source[i].domain = copy.deepcopy(scope.operation_type.destination[i].domain)

        elif isinstance(scope, sast.Program):
            for i in range(len(node.values)):
                node.operation_type.source[i].domain = copy.deepcopy(self.domain)

    def visit_StatementBlock(self, node: sast.StatementBlock):
        computation = self.get_scope()

        assert computation
        assert isinstance(computation, sast.ComputationBlock)
        # The output domains are given by the union of the domains of their uses
        out_domains = []
        for out, outtype in zip(node.outputs, node.operation_type.destination):
            out_domains.extend(self._domains_of_uses_in_scope(computation, out))
        out_domain = _union_domains(out_domains)
        # Initialize output domains with the domain of the result
        for outtype in node.operation_type.destination:
            outtype.domain = copy.deepcopy(out_domain)

        # Compute input domains based on extents and output
        for inp, inptype in zip(node.inputs, node.operation_type.source):
            if isinstance(inptype, sast.ScalarType):
                continue

            # Determine the offsets of the definition of the input
            in_extents = self._union_of_offsets_of_in_scope_definitions(inp, computation)

            # Compute inptype extents that are relative to the in_extents.
            relative_extent = sast.Extent(_offsets_relative_to_defining_offsets(inptype.extent.extents, in_extents))

            # Compute the domain of the input based on the extents and the output domain
            new_domain = _infer_domain_from_extents(out_domain,
                                                    relative_extent)

            # Take max value from current domain if in dictionary
            inptype.domain = inptype.domain.union(new_domain)

    def visit_ComputationBlock(self, node: sast.ComputationBlock):
        # Initialize output domains with the domain of the result intersected with the interval
        for out, outtype in zip(node.outputs, node.operation_type.destination):
            if outtype.domain.is_unknown():
                outtype.domain = self.domain.intersect_with_ranges(node.interval)

        self.push_scope(node)
        for child in reversed(node.body):
            self.visit(child)
        # Initialize input domains as the union of the domains of their uses
        for inp, inptype in zip(node.inputs, node.operation_type.source):
            if isinstance(inptype, sast.ScalarType):
                continue
            in_domains = self._domains_of_uses_in_scope(node, inp)
            in_domain = _union_domains(in_domains)
            if inptype.domain.is_unknown():
                inptype.domain = copy.deepcopy(in_domain)
        self.pop_scope()

    def visit_IfBlock(self, node: sast.IfBlock):
        # We need to infer the domain of the outputs based on the uses of the outputs
        # We will compute the output for the union of the uses.
        computation = self.get_scope_with_type(sast.ComputationBlock)
        assert computation
        assert isinstance(computation, sast.ComputationBlock)

        domains = []
        for out, outtype in zip(node.outputs, node.operation_type.destination):
            out_domain = self._domains_of_uses_in_scope(computation, out)
            domains.extend(out_domain)

        out_domain = _union_domains(domains)
        for outtype in node.operation_type.destination:
            outtype.domain = copy.deepcopy(out_domain)

        # Visit the condition, body, and else_ifs
        self.generic_visit(node)

        # The input domain must match the output domain
        # as it serves as a mask for which output to use
        conditions = [node.condition]
        for elifblock in node.else_ifs:
            conditions.append(elifblock.condition)

        for inp, inptype in zip(conditions, node.operation_type.source):
            inptype.domain = out_domain

    def _in_scope_definitions(self, value: sast.Identifier, scope: sast.ComputationBlock) -> list[ScopedDefinition]:
        """
        Get the definitions of a field in the current scope.

        :param value: The identifier of the field
        :param scope: The current scope (computation block or program)
        :return: A list of scoped definitions
        """
        return [d for d in self.use_def[value] if d.definition_scope == scope]

    def _union_of_offsets_of_in_scope_definitions(self,
                                                  value: sast.Identifier,
                                                  scope: sast.ComputationBlock) -> list[sast.Offset]:
        """
        Get the offsets of the definitions of a field in the current scope.

        :param value: The identifier of the field
        :param scope: The current scope (computation block or program)
        :return: A list of offsets
        """
        return list(itertools.chain.from_iterable(d.field_type.extent.extents
                                                  for d in self._in_scope_definitions(value, scope)))

    def _union_of_domains_of_uses_in_scope(self,
                                          computation: sast.ComputationBlock,
                                          identifier: sast.Identifier) -> sast.Cartesian:
        return _union_domains(self._domains_of_uses_in_scope(computation, identifier))

    def _domains_of_uses_in_scope(self,
                                  computation: sast.ComputationBlock | sast.Program,
                                  identifier: sast.Identifier) -> list[sast.Cartesian]:
        """
        Get the offsets of the uses of a field in the current scope.

        :param uses: The uses dictionary
        :param computation: The current computation block
        :param identifier: The identifier
        :return: A list of offsets
        """
        assert isinstance(identifier, sast.Identifier)

        uses_of_result = self.def_use.get(identifier) or []
        uses_of_result = [u for u in uses_of_result if u.definition_scope == computation]
        # Concatenate all the extents from uses_of_result
        use_domains = []
        for use in uses_of_result:
            # What if the use is of an input field??
            use_domains.append(use.field_type.domain)
        if len(use_domains) == 0:
            print(f"WARNING: No uses of {identifier.as_ir()} found within current scope.")

        # assert no unknown domains
        assert all(not o.is_unknown()
                   for o in use_domains), f"Domain of uses of {identifier.name} must be known before inference"

        return use_domains


def _union_domains(domains: list[sast.Domain]) -> sast.Cartesian:
    """
    Given a list of domains, returns the union of all domains.
    """
    assert len(domains) > 0
    dom = domains[0]
    for d in domains[1:]:
        dom = dom.union(d)
    return dom


def _infer_domain_from_extents(output_domain: sast.Cartesian,
                               extents: sast.Extent) -> sast.Cartesian:
    """
    Given the output domain and extents, infers the domain size of the input field.
    Assuming that the output is accessed at offset (0, 0, 0).
    # TODO Double-Check for non-zero output offsets

    :param output_domain: The domain of the output field
    :param extents: The extents of the input field
    :return: The inferred domain of the input field
    """
    current_domain = copy.deepcopy(output_domain)

    for e in extents.extents:
        # Expand the domain with the extent values
        extent_domain = output_domain.add(e.values)

        current_domain = current_domain.union(extent_domain)

    return current_domain


def _offsets_relative_to_defining_offsets(offsets: list[sast.Offset],
                                          defining_offsets: list[sast.Offset]) -> list[sast.Offset]:
    """
    Given a list of offsets and a list of defining offsets, returns the offsets relative to the defining offsets.
    That is, the closest defining offset is subtracted from the offset.
    If multiple defining offsets are equally close, the first one is chosen.
    Any unknown defining offsets are ignored.

    For example, if we have defining offsets [(0, 0, 0), (1, -1, 0), (?, ?, ?)] and offests [(0, 0, 0), (1, -2, 0)]
    the result is [(0, 0, 0), (0, 1, 0)]

    :param offsets: The offsets to make relative
    :param defining_offsets: The offsets of the defining variable.
    """
    relative_offsets = []
    for offset in offsets:
        min_offset = offset
        min_distance = float('inf')
        for def_offset in defining_offsets:
            if def_offset.is_unknown():
                continue
            relative_offset = offset - def_offset
            distance = relative_offset.l1_norm()
            if distance < min_distance:
                min_offset = relative_offset
                min_distance = distance
        relative_offsets.append(min_offset)
    return relative_offsets
