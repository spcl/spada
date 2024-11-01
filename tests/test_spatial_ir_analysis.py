import os
import pytest
import networkx as nx
from spatialstencil.syntax.spatial_ir import irnodes as spa, analysis, parser, canonicalization
from spatialstencil.syntax.csl import tasks


def test_completion_dag_simple():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'spatial', 'add.sptl')
    kernel = parser.parse_file(file)
    assert isinstance(kernel.body[1], spa.ComputeBlock)
    dag = analysis.to_completion_dag(kernel.body[1])
    assert len(dag.nodes) == 8
    assert [n.optype for n in dag.nodes] == ['post', 'wait'] * 4
    for node in list(dag.nodes)[1:]:
        assert dag.in_degree(node) == 1


def test_completion_dag_concurrent():
    ir = '''
    kernel <>(stream<f32>[90, 80, 3] a) {
        compute u16 i, u16 j in [0:90, 0:80] {
            completion comp1 = foreach i16 k, f32 x in [0:80], receive(a[i, j, 0]) {
                a = x + 1
                b = x + 2
            }
            completion comp2 = send(c, a[i, j, 1])
            completion comp3 = send(d, a[i, j, 2])
            completion comp4 = async {
                e = 5
            }
            await comp1
            await comp2
            await comp3
            await comp4
        }
    }
    '''
    kernel = parser.parse_string(ir)
    dag = analysis.to_completion_dag(kernel.body[0])
    assert [n.optype for n in dag.nodes] == ['post'] * 4 + ['wait'] * 4
    for node in list(dag.nodes)[4:]:
        assert dag.in_degree(node) == 2


@pytest.mark.parametrize('awaitall_needed', (False, True))
def test_completion_dag_multiphase(awaitall_needed):
    ir = f'''
    kernel <>(stream<f32>[90, 80, 3] a) {{
        place u16 i, u16 j in [0:90, 0:80] {{
            f32 field
            f32 field2
            f32[3] field3
        }}
        phase {{
            compute u16 i, u16 j in [0:90, 0:80] {{
                completion comp1 = foreach i16 _, f32 x in [0:80], receive(a[i, j, 0]) {{
                    field = field + x
                }}
                for u16 k in [0:5] {{
                    field2 = k + 1
                }}
                {"await" if not awaitall_needed else "completion comp2 ="} map u16 k#1 in [0:3] {{
                    field3[k#1] = k#1
                }}
                {"await comp1" if not awaitall_needed else ""}
                {"await" if not awaitall_needed else "completion comp3 ="} send(field2, a[i, j, 1])
            }}
        }}
        phase {{
            compute u16 i, u16 j in [0:90, 0:80] {{
                await send(field, a[i, j, 2])
            }}
        }}
    }}
    '''
    kernel = parser.parse_string(ir)
    kernel = canonicalization.canonicalize_phases(kernel)
    kernel = canonicalization.inline_phases(kernel)
    assert len(kernel.body) == 2
    assert isinstance(kernel.body[1], spa.ComputeBlock)
    dag = analysis.to_completion_dag(kernel.body[1])

    if awaitall_needed:
        assert len(dag.nodes) == 8
        assert dag.in_degree(list(dag.nodes)[5]) == 3
    else:
        assert len(dag.nodes) == 10  # Should be 11 with awaitall included
        assert max(dag.in_degree(n) for n in dag.nodes) == 2


def test_limit_indegree():
    ir = f'''
    kernel <>(stream<f32>[90, 80, 4] a) {{
        place u16 i, u16 j in [0:90, 0:80] {{
            f32 field
            f32 field2
            f32[3] field3
        }}
        phase {{
            compute u16 i, u16 j in [0:90, 0:80] {{
                completion comp1 = foreach i16 _, f32 x in [0:80], receive(a[i, j, 0]) {{
                    field = field + x
                }}
                for u16 k in [0:5] {{
                    field2 = k + 1
                }}
                completion comp2 = map u16 k#1 in [0:3] {{
                    field3[k#1] = k#1
                }}
                completion comp3 = send(field2, a[i, j, 1])
                completion comp4 = send(field3, a[i, j, 2])
            }}
        }}
        phase {{
            compute u16 i, u16 j in [0:90, 0:80] {{
                await send(field, a[i, j, 3])
            }}
        }}
    }}
    '''
    kernel = parser.parse_string(ir)
    kernel = canonicalization.canonicalize_phases(kernel)
    kernel = canonicalization.inline_phases(kernel)
    assert len(kernel.body) == 2
    assert isinstance(kernel.body[1], spa.ComputeBlock)
    dag = analysis.to_completion_dag(kernel.body[1])

    # Collect toposort before operation
    topo_before = list(nx.topological_sort(dag))

    assert any(dag.in_degree(n) == 4 for n in dag.nodes)
    tasks._limit_indegree(dag)
    assert max(dag.in_degree(n) for n in dag.nodes) == 2

    topo_after = [n for n in nx.topological_sort(dag) if n.statement_id >= 0]
    assert topo_before == topo_after  # Approximate path preservation assertion


if __name__ == '__main__':
    test_completion_dag_simple()
    test_completion_dag_concurrent()
    test_completion_dag_multiphase(False)
    test_completion_dag_multiphase(True)
    test_limit_indegree()
