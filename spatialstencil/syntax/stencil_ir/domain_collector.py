from dataclasses import dataclass

import spatialstencil.syntax.stencil_ir.irnodes as sast
from spatialstencil.syntax.stencil_ir.irnodes import ComputationBlock, Program


@dataclass
class ScopedDomain:
    definition_scope: sast.ComputationBlock | sast.Program
    domain: sast.Cartesian


class DomainCollector(sast.ScopedNodeVisitor):
    """
    Collects the domains of all the identifiers in the program which
    potentially require storage.

    This includes input fields, output fields, and fields that are materialized.
    """

    domains: dict[str, list[ScopedDomain]]
    _union_domain: sast.Cartesian | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domains = {}
        self._union_domain = None

    def get_union_domain(self) -> sast.Cartesian:
        """
        Compute the smallest and largest values for each dimension in the domain,
        for all the domains collected.

        :return: a domain that covers all the domains collected
        """
        if self._union_domain:
            return self._union_domain

        domain = sast.Cartesian(x=sast.Interval(0, 0), y=sast.Interval(0, 0), z=sast.Interval(0, 0))
        for domains in self.domains.values():
            for sd in domains:
                domain = domain.union(sd.domain)
        self._union_domain = domain
        return domain

    def get_shift(self) -> tuple[int, int, int]:
        union_domain = self.get_union_domain()
        return self.shift_for_domain(union_domain)

    @staticmethod
    def shift_for_domain(domain: tuple[int, int, int]) -> tuple[int, int, int]:
        shift_x = -domain.x[0] if domain.x[0] < 0 else 0
        shift_y = -domain.y[0] if domain.y[0] < 0 else 0
        shift_z = -domain.z[0] if domain.z[0] < 0 else 0
        
        return shift_x, shift_y, shift_z
    
    
    def get_shifted_domain(self, identifier: sast.Identifier, scope: sast.Program | sast.ComputationBlock) -> sast.Cartesian | None:
        """
        Get the domain of an identifier in a given scope, where the negative values have been shifted to 0.

        :param identifier:
        :param scope:
        :return:
        """
        domain = self.get_domain(identifier, scope)
        if domain is None:
            return None

        return domain.add(self.get_shift())


    def get_domain(self, identifier: sast.Identifier,
                   scope: sast.ComputationBlock | sast.Program) -> sast.Cartesian | None:
        """
        Get the domain of an identifier in a given scope.
        :param identifier:
        :param scope:
        :return:
        """
        if identifier.name in self.domains:
            for sd in self.domains[identifier.name]:
                if sd.definition_scope == scope:
                    return sd.domain
        return None

    def _add_domain(self, identifier: sast.Identifier, domain: sast.Cartesian):
        self._union_domain = None
        if identifier.name not in self.domains:
            self.domains[identifier.name] = []
        scope = self.get_scope()
        assert isinstance(scope, sast.ComputationBlock) or isinstance(scope, sast.Program)
        # If the domain is already in the list of domains for this scope, do nothing
        if any(sd.definition_scope == scope for sd in self.domains[identifier.name]):
            return

        self.domains[identifier.name].append(ScopedDomain(scope, domain))

    def do_visit_ComputationBlock(self, computation: ComputationBlock):
        # Add all the inputs to the domain collector
        assert isinstance(self.get_scope(), sast.ComputationBlock)
        for inp, inp_t in zip(computation.inputs, computation.operation_type.source):
            if isinstance(inp_t, sast.ViewType):
                assert isinstance(inp_t.domain, sast.Cartesian)
                self._add_domain(inp, inp_t.domain)

        self.generic_visit(computation)

    def do_visit_Program(self, program: Program):
        # Add all the inputs to the domain collector
        for inp, inp_t in zip(program.inputs, program.operation_type.source):
            if isinstance(inp_t, sast.FieldType):
                assert isinstance(inp_t.domain, sast.Cartesian)
                self._add_domain(inp, inp_t.domain)

        for out, out_t in zip(program.outputs, program.operation_type.source):
            if isinstance(out_t, sast.FieldType):
                assert isinstance(out_t.domain, sast.Cartesian)
                self._add_domain(out, out_t.domain)

        self.generic_visit(program)

    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        # Add the domain of the result to the domain collector
        self._add_domain(node.result, node.operation_type.destination[0].domain)

