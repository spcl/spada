import unittest
import numpy as np

from spada.placement.graph import FieldDomain
from spada.placement.partition import FieldPartition


class TestPlacement(unittest.TestCase):
    def test_interleaved_placement(self):
        # 1 by 1
        partition_array = np.array([[0, 0], [0, 0]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        placement = partition.place_interleaved()
        assert np.allclose( placement.strides[:, 0], 1)
        assert np.allclose(placement.strides[:, 1], 1)

        # 2 by 2
        partition_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        placement = partition.place_interleaved()
        assert np.allclose( placement.strides[:, 0], 2)
        assert np.allclose(placement.strides[:, 1], 2)

        # 1 by 2
        partition_array = np.array([[0, 1], [0, 0]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        placement = partition.place_interleaved()
        assert np.allclose( placement.strides[:, 0], 1)
        assert np.allclose(placement.strides[:, 1], 2)

        # 2 by 1
        partition_array = np.array([[0, 0], [1, 0]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        placement = partition.place_interleaved()
        assert np.allclose(placement.strides[:, 0], 2)
        assert np.allclose(placement.strides[:, 1], 1)

    def test_block_placement(self):

        x_length = 2
        y_length = 4
        domain = FieldDomain(np.array([[0, 0, 0], [x_length, y_length, 8]], dtype=np.int32))

        # 1 by 1
        partition_array = np.array([[0, 0], [0, 0]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        placement = partition.place_blocked(domain)
        assert np.allclose( placement.strides[:], 1)
        assert np.allclose(placement.offsets[:, 0], 0)

        # 2 by 2
        partition_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        placement = partition.place_blocked(domain)
        assert np.allclose(placement.strides[:], 1)
        assert np.allclose(placement.offsets[:, 0], [0, x_length, 0, x_length])
        assert np.allclose(placement.offsets[:, 1], [y_length, 0, 0, y_length])

        # 1 by 2
        partition_array = np.array([[0, 1], [0, 0]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        placement = partition.place_blocked(domain)
        assert np.allclose(placement.strides[:], 1)
        assert np.allclose(placement.offsets[:, 0], [0, 0])
        assert np.allclose(placement.offsets[:, 1], [y_length, 0])

        # 2 by 1
        partition_array = np.array([[0, 0], [1, 0]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        placement = partition.place_blocked(domain)
        assert np.allclose(placement.strides[:], 1)
        assert np.allclose(placement.offsets[:, 0], [0, x_length])
        assert np.allclose(placement.offsets[:, 1], [0, 0])


if __name__ == '__main__':
    unittest.main()
