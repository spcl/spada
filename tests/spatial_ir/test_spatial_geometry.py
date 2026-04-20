import copy
import unittest

from spada.syntax.spatial_ir.grid_geometry import Rectangle, intersect_ranges, split_rectangle, split_rectangles, group_rectangles_by_domain

RectWithId = Rectangle[int]


class TestStencilIR(unittest.TestCase):

    @staticmethod
    def check_rect_is_covered(rect: Rectangle, cover: list[Rectangle]):
        # Note this is a slow check, but it is only used for testing
        for x in range(rect.x_range[0], rect.x_range[1], rect.x_range[2]):
            for y in range(rect.y_range[0], rect.y_range[1], rect.y_range[2]):
                found = False
                for r in cover:
                    if r.contains_point(x, y):
                        found = True
                        break
                assert found, f"Point ({x}, {y}) not covered by any rectangle in cover"

    def check_rectangle_split_result(self, rects: list[RectWithId], split_result: list[RectWithId]):
        for r in rects:
            r_with_id = [rect for rect in split_result if rect.metadata == r.metadata]
            self.check_rect_is_covered(r, r_with_id)
            for split in r_with_id:
                assert split.is_subset_of(r), f"Split rectangle {split} is not a subset of original rectangle {r}"

            # Check that there are no duplicate rectangles
            for i in range(len(r_with_id)):
                for j in range(i + 1, len(r_with_id)):
                    assert not r_with_id[i].is_equal(r_with_id[j]), f"Duplicate rectangles found: {r_with_id[i]} and {r_with_id[j]}"

    def test_rectangle_splitting(self):

        rect1 = RectWithId(x_range=(0, 3, 1), y_range=(0, 3, 1), metadata=1)
        rect2 = RectWithId(x_range=(1, 5, 1), y_range=(1, 5, 1), metadata=2)
        rect3 = RectWithId(x_range=(-1, 7, 1), y_range=(-1, 7, 1), metadata=3)
        rect4 = RectWithId(x_range=(2, 5, 1), y_range=(-1, 3, 1), metadata=4)

        rects = [rect1, rect2, rect3, rect4]

        split = split_rectangles(rects)

        self.check_rectangle_split_result(rects, split)

        rect5 = RectWithId(x_range=(0, 3, 1), y_range=(10, 13, 1), metadata=5)
        rect6 = RectWithId(x_range=(0, 3, 1), y_range=(10, 13, 1), metadata=6)
        rect7 = RectWithId(x_range=(0, 3, 1), y_range=(0, 13, 1), metadata=7)

        rects2 = [rect2, rect3, rect4, rect1, rect5, rect6, rect7]
        split2 = split_rectangles(rects2)
        self.check_rectangle_split_result(rects2, split2)


    def test_rectangle_splitting_strided(self):

        rect1 = RectWithId(x_range=(0, 3, 2), y_range=(0, 3, 2), metadata=1)
        rect2 = RectWithId(x_range=(0, 5, 2), y_range=(0, 5, 2), metadata=2)

        rects = [rect1, rect2]

        split = split_rectangles(rects)
        self.check_rectangle_split_result(rects, split)

        rect3 = RectWithId(x_range=(-1, 7, 2), y_range=(-1, 7, 2), metadata=3)
        rect4 = RectWithId(x_range=(2, 5, 2), y_range=(-1, 3, 2), metadata=4)
        rect5 = RectWithId(x_range=(2, 8, 2), y_range=(-1, 5, 2), metadata=5)
        
        rects = [rect1, rect2, rect3, rect4, rect5]
        split = split_rectangles(rects)
        self.check_rectangle_split_result(rects, split)


    def test_rectangle_splitting_strided_2(self):

        rect1 = RectWithId(x_range=(3, 14, 2), y_range=(0, 3, 2), metadata=1)
        rect2 = RectWithId(x_range=(0, 5, 2), y_range=(1, 12, 2), metadata=2)

        rect3 = RectWithId(x_range=(-1, 7, 2), y_range=(-1, 7, 2), metadata=3)
        rect4 = RectWithId(x_range=(2, 3, 2), y_range=(-1, 3, 2), metadata=4)
        rect5 = RectWithId(x_range=(-3, 8, 2), y_range=(-2, 5, 2), metadata=5)
        
        rects = [rect1, rect2]

        split = split_rectangles(rects)
        self.check_rectangle_split_result(rects, split)

        rects = [rect4, rect5, rect1, rect2, rect3]
        split = split_rectangles(rects)
        self.check_rectangle_split_result(rects, split)



    def test_rectangle_splitting_strided_3(self):

        rect1 = RectWithId(x_range=(3, 14, 1), y_range=(0, 3, 2), metadata=1)
        rect2 = RectWithId(x_range=(0, 5, 1), y_range=(1, 12, 2), metadata=2)

        rect3 = RectWithId(x_range=(-1, 7, 1), y_range=(-1, 7, 2), metadata=3)
        rect4 = RectWithId(x_range=(2, 3, 1), y_range=(-1, 3, 2), metadata=4)
        rect5 = RectWithId(x_range=(-3, 8, 1), y_range=(-2, 5, 2), metadata=5)
        
        rects = [rect1, rect2]

        split = split_rectangles(rects)
        self.check_rectangle_split_result(rects, split)

        rects = [rect4, rect5, rect1, rect2, rect3]
        split = split_rectangles(rects)
        self.check_rectangle_split_result(rects, split)
        
        rect1 = RectWithId(x_range=(3, 14, 2), y_range=(0, 3, 1), metadata=1)
        rect2 = RectWithId(x_range=(0, 5, 2), y_range=(1, 12, 1), metadata=2)

        rect3 = RectWithId(x_range=(-1, 7, 2), y_range=(-1, 7, 1), metadata=3)
        rect4 = RectWithId(x_range=(2, 3, 2), y_range=(-1, 3, 1), metadata=4)
        rect5 = RectWithId(x_range=(-3, 8, 2), y_range=(-2, 5, 1), metadata=5)
        
        rects = [rect1, rect2]

        split = split_rectangles(rects)
        self.check_rectangle_split_result(rects, split)

        rects = [rect4, rect5, rect1, rect2, rect3]
        split = split_rectangles(rects)
        self.check_rectangle_split_result(rects, split)


    def test_point_contains(self):
        
        rect1 = RectWithId(x_range=(0, 9, 2), y_range=(0, 1, 1), metadata=0)
        
        assert rect1.contains_point(0, 0)
        assert not rect1.contains_point(1, 0)
        assert rect1.contains_point(8, 0)
        assert not rect1.contains_point(7, 0)
        assert not rect1.contains_point(0, 1)
        assert not rect1.contains_point(5, 1)
        
        
        rect1 = RectWithId(x_range=(0, 9, 2), y_range=(0, 6, 2), metadata=0)
        
        assert rect1.contains_point(0, 0)
        assert not rect1.contains_point(1, 0)
        assert rect1.contains_point(8, 0)
        assert not rect1.contains_point(7, 0)
        assert not rect1.contains_point(0, 1)
        assert not rect1.contains_point(5, 1)
        
        
        rect1 = Rectangle(x_range=(0, 5, 2), y_range=(4, 5, 2), metadata=2)
        assert rect1.contains_point(0, 4)
        
    def test_rectangle_intersection(self):
        
        rect1 = RectWithId(x_range=(0, 5, 2), y_range=(0, 2, 2), metadata=0)
        
        rect2 = RectWithId(x_range=(2, 4, 2), y_range=(0, 2, 2), metadata=1)
        assert rect1.intersects(rect2)
        
        rect2 = RectWithId(x_range=(1, 7, 2), y_range=(0, 2, 2), metadata=1)
        assert not rect1.intersects(rect2)

        rect2 = RectWithId(x_range=(0, 5, 2), y_range=(1, 2, 2), metadata=1)
        assert not rect1.intersects(rect2)
        
        rect2 = RectWithId(x_range=(0, 5, 2), y_range=(1, 3, 1), metadata=1)
        assert not rect1.intersects(rect2)
        
        rect1 = RectWithId(x_range=(0, 5, 2), y_range=(0, 3, 2), metadata=0)
        assert rect1.intersects(rect2)


    # ------------------------------------------------------------------
    # Helper: expand a range tuple to its explicit element set
    # ------------------------------------------------------------------
    @staticmethod
    def _expand(r):
        start, stop, stride = r
        return set(range(start, stop, stride))

    def _assert_intersection(self, r1, r2, expected_set):
        """Assert that intersect_ranges(r1, r2) produces exactly expected_set."""
        result = intersect_ranges(r1, r2)
        if expected_set:
            self.assertIsNotNone(result, f"Expected non-empty intersection of {r1} and {r2}")
            self.assertEqual(self._expand(result), expected_set,
                             f"intersect_ranges({r1}, {r2}) = {result}, expected elements {expected_set}")
        else:
            self.assertIsNone(result, f"Expected empty intersection of {r1} and {r2}, got {result}")

    def test_intersect_ranges(self):
        # --- Original cases (stride-1 vs stride-2) ---
        self._assert_intersection((0, 2, 2), (1, 3, 1), set())   # {0} ∩ {1,2} = ∅
        self._assert_intersection((0, 3, 2), (1, 3, 1), {2})      # {0,2} ∩ {1,2} = {2}

        # --- Equal strides ---
        self._assert_intersection((0, 8, 4), (0, 12, 4), {0, 4})   # same coset
        self._assert_intersection((0, 8, 4), (2,  8, 4), set())     # different cosets

        # --- One stride divides the other (stride-4 vs stride-2) ---
        # {0,4,8} ∩ {0,2,4,6,8,10} = {0,4,8}
        self._assert_intersection((0, 12, 4), (0, 12, 2), {0, 4, 8})
        # {0,4,8} ∩ {2,4,6,8,10} = {4,8}
        self._assert_intersection((0, 12, 4), (2, 12, 2), {4, 8})
        # {0,4,8} ∩ {1,3,5,7,9,11} = ∅  (different parity)
        self._assert_intersection((0, 12, 4), (1, 12, 2), set())
        # Symmetry: same result when arguments are swapped
        self._assert_intersection((0, 12, 2), (0, 12, 4), {0, 4, 8})

        # --- Stride-4 vs stride-8 (stride-8 is a multiple of stride-4) ---
        # {0,4,8} ∩ {4} = {4}
        self._assert_intersection((0, 12, 4), (4,  8, 8), {4})
        # {0,4,8} ∩ {8} = {8}
        self._assert_intersection((0, 12, 4), (8, 16, 8), {8})
        # {0,4,8} ∩ {2,10} = ∅
        self._assert_intersection((0, 12, 4), (2, 16, 8), set())

        # --- Stride-4 vs stride-6 (lcm=12) ---
        # {0,4,8,12,16,20} ∩ {0,6,12,18} = {0,12}
        self._assert_intersection((0, 24, 4), (0, 24, 6), {0, 12})
        # {4,8,12,16,20} ∩ {6,12,18} = {12}
        self._assert_intersection((4, 24, 4), (6, 24, 6), {12})
        # {4,8,12,16,20} ∩ {3,9,15,21} = ∅  (gcd=2 does not divide 3-4=-1)
        self._assert_intersection((4, 24, 4), (3, 24, 6), set())

        # --- Stop-boundary edge cases ---
        # Intersection element exactly at stop boundary should be excluded
        self._assert_intersection((0, 4, 4), (0, 12, 2), {0})      # only PE 0
        self._assert_intersection((0, 4, 4), (4, 12, 4), set())     # starts just after rect1 ends

        # --- Twophase-like scenario: stride-4 vs stride-2 in mixed phases ---
        # place [0:12:1] split by dataflow [0:12:4] and computes [0:12:2] style blocks
        self._assert_intersection((0, 12, 4), (5, 7,  2), set())    # {0,4,8} ∩ {5} = ∅
        self._assert_intersection((0, 12, 4), (4, 12, 8), {4})        # range(4,12,8) = {4}

    def test_intersect_ranges_symmetry(self):
        """intersect_ranges must return the same element set regardless of argument order."""
        pairs = [
            ((0, 12, 4), (0, 12, 2)),
            ((0, 24, 4), (0, 24, 6)),
            ((4, 24, 4), (6, 24, 6)),
            ((0, 12, 4), (4,  8, 8)),
            ((0, 12, 4), (1, 12, 2)),
        ]
        for r1, r2 in pairs:
            res_fwd = intersect_ranges(r1, r2)
            res_rev = intersect_ranges(r2, r1)
            set_fwd = self._expand(res_fwd) if res_fwd else set()
            set_rev = self._expand(res_rev) if res_rev else set()
            self.assertEqual(set_fwd, set_rev,
                             f"Asymmetric result for intersect_ranges({r1},{r2}): {res_fwd} vs {res_rev}")

    def test_split_rectangles_mixed_strides(self):
        """split_rectangles must correctly handle blocks whose strides are neither equal nor 1."""
        # Simulate the twophase_reduce_1D scenario: stride-1 place, stride-2 and stride-4 computes
        place     = RectWithId(x_range=(0, 12, 1), y_range=(0, 1, 1), metadata=0)
        df_ph2    = RectWithId(x_range=(0, 12, 1), y_range=(0, 1, 1), metadata=1)  # phase-2 dataflow
        df_ph3    = RectWithId(x_range=(0, 12, 4), y_range=(0, 1, 1), metadata=2)  # phase-3 dataflow (stride 4)
        comp_s2_a = RectWithId(x_range=(1,  3, 2), y_range=(0, 1, 1), metadata=3)  # stride-2 compute
        comp_s2_b = RectWithId(x_range=(2,  3, 2), y_range=(0, 1, 1), metadata=4)  # stride-2 compute
        comp_s8   = RectWithId(x_range=(4,  8, 8), y_range=(0, 1, 1), metadata=5)  # stride-8 compute

        rects = [place, df_ph2, df_ph3, comp_s2_a, comp_s2_b, comp_s8]
        split = split_rectangles(rects)
        self.check_rectangle_split_result(rects, split)

    # ------------------------------------------------------------------
    # Tests for Rectangle.largest_contained_x / largest_contained_y
    # ------------------------------------------------------------------

    def test_largest_contained_stride1(self):
        """Stride-1 ranges: last PE equals stop - 1."""
        r = RectWithId(x_range=(0, 7, 1), y_range=(2, 5, 1), metadata=0)
        self.assertEqual(r.largest_contained_x(), 6)
        self.assertEqual(r.largest_contained_y(), 4)

    def test_largest_contained_stride2_even(self):
        """Stride-2 starting at 0: last PE is the largest even index below stop."""
        # [0:6:2] = {0, 2, 4}  → last = 4
        r = RectWithId(x_range=(0, 6, 2), y_range=(0, 6, 2), metadata=0)
        self.assertEqual(r.largest_contained_x(), 4)
        self.assertEqual(r.largest_contained_y(), 4)

    def test_largest_contained_stride2_odd(self):
        """Stride-2 starting at 1: last PE is the largest odd index below stop."""
        # [1:6:2] = {1, 3, 5}  → last = 5
        r = RectWithId(x_range=(1, 6, 2), y_range=(1, 6, 2), metadata=0)
        self.assertEqual(r.largest_contained_x(), 5)
        self.assertEqual(r.largest_contained_y(), 5)

    def test_largest_contained_non_canonical_stop(self):
        """Non-canonical stop (stop not on stride boundary): last PE should equal stop rounded down."""
        # [5:6:2] has canonical stop 7 but only covers {5}; largest_contained_x must return 5.
        r = RectWithId(x_range=(5, 6, 2), y_range=(0, 4, 1), metadata=0)
        self.assertEqual(r.largest_contained_x(), 5)
        # [0:4:1] → last = 3
        self.assertEqual(r.largest_contained_y(), 3)

    def test_largest_contained_single_pe(self):
        """Ranges that cover exactly one PE."""
        r = RectWithId(x_range=(3, 4, 1), y_range=(7, 8, 1), metadata=0)
        self.assertEqual(r.largest_contained_x(), 3)
        self.assertEqual(r.largest_contained_y(), 7)

        r2 = RectWithId(x_range=(6, 7, 2), y_range=(9, 10, 3), metadata=0)
        self.assertEqual(r2.largest_contained_x(), 6)
        self.assertEqual(r2.largest_contained_y(), 9)

    def test_largest_contained_large_stride(self):
        """Stride larger than the covered range: still only one PE."""
        # [3:8:4] = {3, 7}  → last = 7
        r = RectWithId(x_range=(3, 8, 4), y_range=(0, 1, 1), metadata=0)
        self.assertEqual(r.largest_contained_x(), 7)

        # [3:6:4] = {3}  → last = 3  (7 is beyond stop=6)
        r2 = RectWithId(x_range=(3, 6, 4), y_range=(0, 1, 1), metadata=0)
        self.assertEqual(r2.largest_contained_x(), 3)

    def test_largest_contained_rect_size_consistency(self):
        """
        Verify that building rect_size from largest_contained_{x,y}+1 gives the
        correct tight bounding box for a collection of rectangles – the scenario
        that caused the 'expected N PEs, got M' regression in the CSL layout.
        """
        # Simulate the stride-2 blocks of the uvbke benchmark:
        # even-start:  [0:6:2] = {0,2,4}  |  odd-start:  [1:6:2] = {1,3,5}
        # edge blocks: [5:6:2] = {5}  (canonical stop 7, but max PE is still 5)
        rects = [
            RectWithId(x_range=(0, 6, 2), y_range=(0, 6, 2), metadata=0),
            RectWithId(x_range=(1, 6, 2), y_range=(0, 6, 2), metadata=1),
            RectWithId(x_range=(0, 6, 2), y_range=(1, 6, 2), metadata=2),
            RectWithId(x_range=(1, 6, 2), y_range=(1, 6, 2), metadata=3),
            RectWithId(x_range=(5, 6, 2), y_range=(5, 6, 2), metadata=4),  # non-canonical stop
        ]
        x0 = min(r.x_range[0] for r in rects)
        y0 = min(r.y_range[0] for r in rects)
        x1 = max(r.largest_contained_x() + 1 for r in rects)
        y1 = max(r.largest_contained_y() + 1 for r in rects)
        self.assertEqual(x1 - x0, 6, "rect_size width should be 6, not 7")
        self.assertEqual(y1 - y0, 6, "rect_size height should be 6, not 7")


if __name__ == '__main__':
    unittest.main()
