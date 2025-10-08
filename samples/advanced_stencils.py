# Stencils in GT4Py syntax
from math import sqrt
import numpy as np
from gt4py import computation, interval, PARALLEL, FORWARD, BACKWARD

Field3D = np.ndarray


# See https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L194
def horizontal_diffusion(in_field: Field3D, out_field: Field3D, coeff: Field3D):
    with computation(PARALLEL), interval(...):
        lap_field = 4.0 * in_field[0, 0, 0] - (
            in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0])
        res = lap_field[1, 0, 0] - lap_field[0, 0, 0]
        flx_field = 0 if (res * (in_field[1, 0, 0] - in_field[0, 0, 0])) > 0 else res
        res = lap_field[0, 1, 0] - lap_field[0, 0, 0]
        fly_field = 0 if (res * (in_field[0, 1, 0] - in_field[0, 0, 0])) > 0 else res
        out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
            flx_field[0, 0, 0] - flx_field[-1, 0, 0] + fly_field[0, 0, 0] - fly_field[0, -1, 0])


# See https://github.com/spcl/open-earth-compiler/blob/1e48dee6a1a021bc11d6621432450406349b3733/test/Examples/hdiffsa.mlir
def hdiffsa(arg0: Field3D, arg1: Field3D, arg2: Field3D, arg3: Field3D, arg4: Field3D):
    with computation(PARALLEL), interval(...):
        i15 = arg0[0, 0, 0]
        i16 = arg0[-1, 0, 0] + arg0[1, 0, 0]
        i17 = -2.0 * i15
        i18 = i16 + i17
        i23 = arg0[0, 1, 0] - i15
        i24 = arg0[0, -1, 0] - i15
        i25 = i23 * arg3[0, 0, 0]
        i26 = i24 * arg4[0, 0, 0]
        i9 = i25 + i18 + i26

        i15 = i9[1, 0, 0] - i9[0, 0, 0]
        i18 = arg0[1, 0, 0] - arg0[0, 0, 0]
        i19 = i15 * i18
        i10 = 0 if (i19 > 0) else i15

        i15 = i9[0, 1, 0] - i9[0, 0, 0]
        i17 = i15 * arg3[0, 0, 0]
        i20 = arg0[0, 1, 0] - arg0[0, 0, 0]
        i21 = i17 * i20
        i11 = 0 if (i21 > 0) else i17

        # Output
        i15 = i10[-1, 0, 0] - i10[0, 0, 0]
        i18 = i11[0, -1, 0] - i11[0, 0, 0]
        arg2 = arg0[0, 0, 0] + arg1[0, 0, 0] * (i15 + i18)


# See https://github.com/spcl/open-earth-compiler/blob/1e48dee6a1a021bc11d6621432450406349b3733/test/Examples/hdiffsa.mlir
def hdiffsa(arg0: Field3D, arg1: Field3D, arg2: Field3D, arg3: Field3D, arg4: Field3D):
    with computation(PARALLEL), interval(...):
        i15 = arg0[0, 0, 0]
        i16 = arg0[-1, 0, 0] + arg0[1, 0, 0]
        i17 = -2.0 * i15
        i18 = i16 + i17
        i23 = arg0[0, 1, 0] - i15
        i24 = arg0[0, -1, 0] - i15
        i25 = i23 * arg3[0, 0, 0]
        i26 = i24 * arg4[0, 0, 0]
        i9 = i25 + i18 + i26

        i15 = i9[1, 0, 0] - i9[0, 0, 0]
        i18 = arg0[1, 0, 0] - arg0[0, 0, 0]
        i19 = i15 * i18
        i10 = 0 if (i19 > 0) else i15

        i15 = i9[0, 1, 0] - i9[0, 0, 0]
        i17 = i15 * arg3[0, 0, 0]
        i20 = arg0[0, 1, 0] - arg0[0, 0, 0]
        i21 = i17 * i20
        i11 = 0 if (i21 > 0) else i17

        # Output
        i15 = i10[-1, 0, 0] - i10[0, 0, 0]
        i18 = i11[0, -1, 0] - i11[0, 0, 0]
        arg2 = arg0[0, 0, 0] + arg1[0, 0, 0] * (i15 + i18)