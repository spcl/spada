from spatialstencil.syntax.spatial_ir import parser
from spatialstencil.optimizations.optimization_pass import optimization_pass
import os



def _load_ref_file(file) -> list[str]:
    # change the file extension to .ref
    file = file[:-5] + '.ref_tile'
    # read the file and return the lines as a list without the newline character
    with open(file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def _tiling_test(file):
    """
    Tests a roundtrip IR->parse->IR->parse->IR for differences.

    :param file:
    :return:
    """
    program = parser.parse_file(file)
    program_optimized = optimization_pass(program)
    ir_1 = program_optimized.as_ir()

    ir_ref = _load_ref_file(file)
    num_dataflow = ir_ref[-1]
    ir_ref = ir_ref[:-1]

    count_ref = 0
    for line in ir_ref:
        assert ("compute i16 i, i16 j in " + line) in ir_1
        count_ref += 1
    count = 0
    for line in ir_1.splitlines():
        if "compute i16 i, i16 j in " in line:
            count += 1
    assert count == count_ref
    count_dataflow = 0
    for line in ir_1.splitlines():
        if "dataflow i16 i, i16 j in" in line:
            count_dataflow += 1
    assert count_dataflow == int(num_dataflow)
    



def test_simple_bcast():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'collective', 'simple_bcast.sptl')
    _tiling_test(file)

def test_simple_reduce_grid_pipelined_1():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'collective', 'simple_reduce_grid_pipelined_1.sptl')
    _tiling_test(file)

def test_simple_reduce_grid_pipelined_2():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'collective', 'simple_reduce_grid_pipelined_2.sptl')
    _tiling_test(file)

def test_simple_reduce_grid_1():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'collective', 'simple_reduce_grid_1.sptl')
    _tiling_test(file)

def test_simple_reduce_grid_2():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'collective', 'simple_reduce_grid_2.sptl')
    _tiling_test(file)

def test_simple_reduce_snake_pipelined_1():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'collective', 'simple_reduce_snake_pipelined_1.sptl')
    _tiling_test(file)

def test_simple_reduce_snake_1():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'collective', 'simple_reduce_snake_1.sptl')
    _tiling_test(file)

def test_simple_reduce_looped():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'collective', 'simple_reduce_looped.sptl')
    _tiling_test(file)

def test_medium_reduce_grid_1():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'collective', 'medium_reduce_grid_1.sptl')
    _tiling_test(file)    

def test_hard_reduce_1():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'collective', 'hard_reduce_1.sptl')
    _tiling_test(file)

def test_hard_reduce_2():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'collective', 'hard_reduce_2.sptl')
    _tiling_test(file)

def test_hard_reduce_3():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'collective', 'hard_reduce_3.sptl')
    _tiling_test(file)


if __name__ == '__main__':
    test_simple_bcast()
    test_simple_reduce_grid_pipelined_1()
    test_simple_reduce_grid_pipelined_2()
    test_simple_reduce_grid_1()
    test_simple_reduce_grid_2()
    test_simple_reduce_snake_pipelined_1()
    test_simple_reduce_snake_1()
    test_simple_reduce_looped()
    test_medium_reduce_grid_1()
    test_hard_reduce_1()
    test_hard_reduce_2()
    test_hard_reduce_3()