import copy
import unittest

from spatialstencil.syntax.spatial_ir.grid_geometry import Rectangle, intersect_ranges, split_rectangle, split_rectangles, group_rectangles_by_domain

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


    def test_intersect_ranges(self):
        
        range1=(0, 2, 2)
        range2=(1, 3, 1)
        
        isec = intersect_ranges(range1, range2)
        assert isec is None
        
        range1=(0, 3, 2)
        range2=(1, 3, 1)
        
        isec = intersect_ranges(range1, range2)
        assert isec is not None

if __name__ == '__main__':
    unittest.main()
