import unittest
from spatialstencil.syntax.gt4py import parser
from spatialstencil.syntax.stencil_ir import irnodes as sast, analysis
from spatialstencil.lowering import gt4py_to_stencil_ir
import numpy as np
import os

Field3D = np.ndarray


class TestStencilIRParser(unittest.TestCase):

    def setUp(self):
        self.gtfuncs = parser.parse_file(__file__)  # Parse this file

    def test_lower_gt4py_intermediates(self):
        program = self.gtfuncs['simple']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)
        assert irprogram.name == 'simple'
        assert [inp.name for inp in irprogram.inputs] == ['inp']
        assert [out.name for out in irprogram.outputs] == ['out']
        assert len(irprogram.computations) == 2

        # Test computation
        comp = irprogram.computations[0]
        assert comp.schedule == sast.ComputationType.PARALLEL
        assert comp.interval == [sast.Interval(0, None), sast.Interval(0, None), sast.Interval(0, None)]

        # Test input/output collection
        assert [inp.name for inp in comp.inputs] == ['inp']
        assert [out.name for out in comp.outputs] == ['out']
        extents = analysis.collect_extents(comp)
        assert next(iter(extents['inp'].keys())) == (0, None, 0, None, 0, None)
        assert next(iter(extents['inp'].values())) == {(0, 0, 0)}
        assert next(iter(extents['out'].values())) == {(0, 0, 0)}

    def test_lower_gt4py_intermediates_2(self):
        program = self.gtfuncs['unused']
        with self.assertWarns(UserWarning) as wrn:
            irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program, materialize=False)
        assert 'No uses of unused' in str(wrn.warning)
        assert irprogram.name == 'unused'
        assert [inp.name for inp in irprogram.inputs] == ['inp']
        assert [out.name for out in irprogram.outputs] == ['out']
        assert len(irprogram.computations) == 3

        # Test computation
        comps = irprogram.computations
        assert comps[0].schedule == sast.ComputationType.PARALLEL
        assert comps[0].interval == [sast.Interval(0, None), sast.Interval(0, None), sast.Interval(0, None)]

        # Test input/output collection
        assert [inp.name for inp in comps[0].inputs] == ['inp']
        assert [out.name for out in comps[0].outputs] == ['used']
        assert [inp.name for inp in comps[1].inputs] == ['used']
        assert [out.name for out in comps[1].outputs] == ['out']

    def test_lower_gt4py_intermediates_3(self):
        program = self.gtfuncs['intermediates_versioning']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program, materialize=False)
        assert irprogram.name == 'intermediates_versioning'
        assert [inp.name for inp in irprogram.inputs] == ['inp']
        assert [out.name for out in irprogram.outputs] == ['out']
        assert len(irprogram.computations) == 3

        # Test computation
        comps = irprogram.computations
        assert comps[0].schedule == sast.ComputationType.PARALLEL
        assert comps[0].interval == [sast.Interval(0, None), sast.Interval(0, None), sast.Interval(0, None)]

        # Test input/output collection
        assert [inp.name for inp in comps[0].inputs] == ['inp']
        assert [out.name for out in comps[0].outputs] == ['interim']
        assert [inp.name for inp in comps[1].inputs] == ['interim']
        assert [out.name for out in comps[1].outputs] == ['out']
        extents = analysis.collect_extents(comps[0])
        assert next(iter(extents['inp'].values())) == {(0, -1, 0)}
        assert next(iter(extents['tmp'].values())) == {(1, 0, 0), (0, 0, 0)}

    def test_lower_gt4py_intermediates_4(self):
        program = self.gtfuncs['intermediates_versioning_2']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program, materialize=False)
        assert irprogram.name == 'intermediates_versioning_2'
        assert [inp.name for inp in irprogram.inputs] == ['inp']
        assert [out.name for out in irprogram.outputs] == ['out']
        assert len(irprogram.computations) == 3

        # Test computation
        comps = irprogram.computations
        assert comps[0].schedule == sast.ComputationType.PARALLEL
        assert comps[0].interval == [sast.Interval(0, None), sast.Interval(0, None), sast.Interval(0, None)]

        # Test input/output collection
        assert [inp.name for inp in comps[0].inputs] == ['inp']
        assert [out.name for out in comps[0].outputs] == ['interim', 'tmp']
        assert [inp.name for inp in comps[1].inputs] == ['interim', 'tmp']
        assert [out.name for out in comps[1].outputs] == ['out', 'tmp']
        extents = analysis.collect_extents(comps[0])
        assert next(iter(extents['inp'].values())) == {(0, -1, 0)}
        assert next(iter(extents['tmp'].values())) == {(1, 0, 0), (0, 0, 0)}

    def test_lower_gt4py_intermediates_5(self):
        program = self.gtfuncs['intermediates_versioning_3']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program, materialize=False)
        assert irprogram.name == 'intermediates_versioning_3'
        assert [inp.name for inp in irprogram.inputs] == ['inp']
        assert [out.name for out in irprogram.outputs] == ['out']
        assert len(irprogram.computations) == 3

        # Test computation
        comps = irprogram.computations
        assert comps[0].schedule == sast.ComputationType.PARALLEL
        assert comps[0].interval == [sast.Interval(0, None), sast.Interval(0, None), sast.Interval(0, None)]

        # Test input/output collection
        assert comps[0].inputs == [sast.Identifier('inp')]
        assert comps[0].outputs == [sast.Identifier('tmp', version=0)]
        assert comps[1].inputs == [sast.Identifier('tmp', version=0)]
        assert comps[1].outputs == [sast.Identifier('out', version=0), sast.Identifier('tmp', version=1)]

    def test_versioning_in_ifelse(self):
        program = self.gtfuncs['versioning_in_ifelse']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program, materialize=False)

        comp = irprogram.computations[0]
        assert comp.outputs == [sast.Identifier('out', version=1)]
        
        # If block (all version 0)
        assert isinstance(comp.body[1], sast.IfBlock)
        ifblock = comp.body[1]
        assert ifblock.outputs == [sast.Identifier('out', version=2)]
        stmt = ifblock.body[0]
        assert stmt.outputs == [sast.Identifier('out', version=0)]
        assert len(ifblock.else_ifs) == 1
        ifelse_block = ifblock.else_ifs[0]
        stmt2 = ifelse_block.body[0]
        assert stmt2.outputs == [sast.Identifier('out', version=1)]
        
        # Successor to if block (version 1)
        succ = comp.body[2]
        assert isinstance(succ, sast.StatementBlock)
        assert succ.outputs == [sast.Identifier('out', version=3)]

        # Return
        ret = comp.body[3]
        assert isinstance(ret, sast.ReturnOp)
        assert succ.outputs == [sast.Identifier('out', version=3)]

    def test_lower_mathcall(self):
        program = self.gtfuncs['simple_mathcall']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)
        comp = irprogram.computations[0]
        assert [inp.name for inp in comp.inputs] == ['inp']
        assert [out.name for out in comp.outputs] == ['out']

    def test_output_overwrite(self):
        program = self.gtfuncs['output_overwrite']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)
        comp = irprogram.computations[0]
        assert comp.inputs == [sast.Identifier('inp'), sast.Identifier('out')]
        assert comp.outputs == [sast.Identifier('out', version=1)]

    def test_lower_gt4py_if(self):
        program = self.gtfuncs['satadjust_specific_humidity']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)
        comp = irprogram.computations[0]
        assert comp.outputs == [sast.Identifier('rh')]
        assert isinstance(comp.body[2], sast.IfBlock)
        ifblock = comp.body[2]
        assert len(ifblock.else_ifs) == 2
        assert ifblock.outputs == [sast.Identifier('qstar', version=3)]

    def test_lower_gt4py_nested_if(self):
        program = self.gtfuncs['satadjust_specific_humidity_nestedif']
        with self.assertRaises(SyntaxError):
            gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)

    def test_domain_inference(self):
        # Parse stencil samples file
        gtfuncs = parser.parse_file(os.path.join(os.path.dirname(__file__), '..', 'samples', 'stencils.py'))
        program = gtfuncs['horizontal_diffusion']

        # Lower without a domain
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)
        assert irprogram.inputs[1].name == 'in_field'
        assert irprogram.operation_type.source[1].domain == sast.Cartesian()

        # Lower with a domain
        domain = (128, 128, 80)
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program, domain=domain)
        # Check input domain
        result = sast.Cartesian(sast.Interval(-2, 130), sast.Interval(-2, 130), sast.Interval(0, 80))
        assert irprogram.operation_type.source[1].domain == result

    def test_domain_inference_vertical(self):
        # Parse stencil samples file
        gtfuncs = parser.parse_file(os.path.join(os.path.dirname(__file__), '..', 'samples', 'stencils.py'))
        program = gtfuncs['vertical_advection']

        # Lower without a domain
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)
        assert irprogram.inputs[-1].name == 'wcon'
        assert irprogram.operation_type.source[-1].domain == sast.Cartesian()
        assert irprogram.outputs[0].name == 'utens_stage'
        assert irprogram.operation_type.destination[0].domain == sast.Cartesian()

        # Lower with a domain
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program, domain=(128, 128, 80))
        assert irprogram.inputs[-1].name == 'wcon'
        assert irprogram.operation_type.source[-1].domain == sast.Cartesian.from_sequence((0, 129, 0, 128, 0, 80))
        assert irprogram.outputs[0].name == 'utens_stage'
        assert irprogram.operation_type.destination[0].domain == sast.Cartesian.from_sequence((0, 128, 0, 128, 0, 80))


# GT4Py stencils for the test
def simple(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        tmp = inp + 1
        out = tmp + 1


def unused(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        used = inp + 1
        unused = inp + 2

    with computation(PARALLEL), interval(...):
        out = used + 1


def intermediates_versioning(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        tmp = inp[0, -1, 0] + 1
        interim = tmp[1, 0, 0] + 1

    with computation(PARALLEL), interval(...):
        tmp = interim + 1
        out = tmp + 1


def intermediates_versioning_2(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        tmp = inp[0, -1, 0] + 1
        interim = tmp[1, 0, 0] + 1

    with computation(PARALLEL), interval(...):
        tmp = interim + tmp
        out = tmp + 1


def intermediates_versioning_3(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        tmp = inp + 1
        tmp = tmp + 1

    with computation(PARALLEL), interval(...):
        tmp = tmp + 1
        out = tmp + 1


def simple_mathcall(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        out = sqrt(inp[1, 0, 0] + 1)


def output_overwrite(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        tmp = inp + out
        out = tmp + 1

def versioning_in_ifelse(inp: Field3D,
                              out: Field3D):
    with computation(PARALLEL), interval(...):
        pred1 = inp < 233.16
        if pred1:
            out = inp
        else:
            out = inp + 1
        out = out + inp


# Adapted saturation adjustment subset code from PyFV3, see:
# https://github.com/NOAA-GFDL/PyFV3/blob/fcaf36e6b3101ec655d4c728d573267c52d42132/pyFV3/stencils/saturation_adjustment.py#L883


# Simplified version without nested conditionals
def satadjust_specific_humidity(tin: Field3D, iqs1: Field3D, wqs1: Field3D, q_cond: Field3D, q_sol: Field3D,
                                qpz: Field3D, rh: Field3D):
    with computation(PARALLEL), interval(...):
        pred1 = tin < 233.16
        pred2 = tin >= 273.16
        if pred1:
            qstar = iqs1
        elif pred2:
            qstar = wqs1
        else:
            rqi = q_sol / q_cond
            qstar = rqi * iqs1 + (1.0 - rqi) * wqs1

        rh = qpz / qstar


def satadjust_specific_humidity_nestedif(tin: Field3D, iqs1: Field3D, wqs1: Field3D, q_cond: Field3D, q_sol: Field3D,
                                         qpz: Field3D, rh: Field3D):
    with computation(PARALLEL), interval(...):
        pred1 = tin < 233.16
        pred2 = tin >= 273.16
        if pred1:  # Homogeneous freezing temperature
            # ice phase
            qstar = iqs1
        elif pred2:  # Freezing temperature
            qstar = wqs1
        else:
            pred3 = q_cond > 1e-6
            if pred3:
                rqi = q_sol / q_cond
            else:
                rqi = (273.16 - tin) / -40.0
            qstar = rqi * iqs1 + (1.0 - rqi) * wqs1

        rh = qpz / qstar


if __name__ == '__main__':
    unittest.main()
