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


# See https://github.com/spcl/open-earth-compiler/blob/1e48dee6a1a021bc11d6621432450406349b3733/test/Examples/hdiffsmag.mlir
def hdiffsmag(arg0: Field3D, arg1: Field3D, arg2: Field3D, arg3: Field3D, arg4: Field3D, arg5: Field3D, arg6: Field3D,
              arg7: Field3D, arg8: Field3D, arg9: Field3D):
    with computation(PARALLEL), interval(...):
        cst = 1.000000e+00
        cst_0 = 6.371229e+06
        cst_1 = 4.8828125E-4
        cst_2 = 7.32421875E-4
        cst_3 = -2.000000e+00
        i24 = cst / cst_0

        i25 = arg9[0, 0, 0]
        i26 = i25 * cst_2
        i27 = cst_1 * i24
        i28 = arg1[0, -1, 0]
        i29 = arg1[0, 0, 0]
        i30 = i28 - i29
        i31 = i30 * i27
        i32 = arg0[-1, 0, 0]
        i33 = arg0[0, 0, 0]
        i34 = i32 - i33
        i35 = i34 * i26
        i36 = i31 - i35
        i18 = i36 * i36

        i28 = arg1[1, 0, 0]
        i30 = i28 - i29
        i31 = i30 * i26
        i32 = arg0[0, 1, 0]
        i33 = arg0[0, 0, 0]
        i34 = i32 - i33
        i35 = i34 * i27
        i36 = i35 + i31
        i19 = i36 * i36

        i24 = arg0[-1, 0, 0]
        i25 = arg0[1, 0, 0]
        i26 = arg0[0, 0, 0]
        i27 = i24 + i25
        i28 = cst_3 * i26 
        i29 = i27 + i28
        i30 = arg0[0, 1, 0]
        i31 = arg0[0, -1, 0]
        i32 = arg7[0, 0, 0]
        i33 = arg8[0, 0, 0]
        i34 = i30 - i26
        i35 = i31 - i26
        i36 = i34 * i32
        i37 = i35 * i33
        i38 = i36 + i29
        i20 = i38 + i37

        i24 = arg1[-1, 0, 0]
        i25 = arg1[1, 0, 0]
        i26 = arg1[0, 0, 0]
        i27 = i24 + i25
        i28 = cst_3 * i26
        i29 = i27 + i28
        i30 = arg1[0, 1, 0]
        i31 = arg1[0, -1, 0]
        i32 = arg5[0, 0, 0]
        i33 = arg6[0, 0, 0]
        i34 = i30 - i26
        i35 = i31 - i26
        i36 = i34 * i32
        i37 = i35 * i33
        i38 = i36 + i29
        i21 = i38 + i37

        i24 = arg2[0, 0, 0]
        i25 = 2.500000e-02 * i24
        i26 = i18[1, 0, 0]
        i27 = i18[0, 0, 0]
        i28 = i26 + i27
        i29 = i28 * 0.5
        i30 = i19[0, -1, 0]
        i31 = i19[0, 0, 0]
        i32 = i30 + i31
        i33 = i32 * 0.5
        i34 = i29 + i33
        i35 = sqrt(i34)
        i36 = i35 * 1.000000e-02
        i37 = i36 - i25
        i39 = i37 if i37 > 0 else 0
        i41 = i39 if i39 < 0.5 else 0.5
        i42 = i20[0, 0, 0]
        i43 = arg0[0, 0, 0]
        i44 = i41 * i42
        arg3 = i44 + i43

        i24 = arg2[0, 0, 0]
        i25 = 2.500000e-02 * i24
        i26 = i18[0, 1, 0]
        i27 = i18[0, 0, 0]
        i28 = i26 + i27
        i29 = i28 * 0.5
        i30 = i19[-1, 0, 0]
        i31 = i19[0, 0, 0]
        i32 = i30 + i31
        i33 = i32 * 0.5
        i34 = i29 + i33
        i35 = sqrt(i34)
        i36 = i35 * 1.000000e-02
        i37 = i36 - i25
        i39 = i37 if i37 > 0 else 0
        i41 = i39 if i39 < 0.5 else 0.5
        i42 = i21[0, 0, 0]
        i43 = arg1[0, 0, 0]
        i44 = i41 * i42
        arg4 = i44 + i43