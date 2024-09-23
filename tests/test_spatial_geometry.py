import copy
import unittest

from spatialstencil.syntax.spatial_ir.grid_geometry import Rectangle, split_rectangles, group_rectangles_by_domain

RectWithId = Rectangle[int]


class TestStencilIR(unittest.TestCase):

    @staticmethod
    def check_rect_is_covered(rect: Rectangle, cover: list[Rectangle]):
        # Note this is a slow check, but it is only used for testing
        for x in range(rect.x_range[0], rect.x_range[1]):
            for y in range(rect.y_range[0], rect.y_range[1]):
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

        rect1 = RectWithId(x_range=(0, 3), y_range=(0, 3), metadata=1)
        rect2 = RectWithId(x_range=(1, 5), y_range=(1, 5), metadata=2)
        rect3 = RectWithId(x_range=(-1, 7), y_range=(-1, 7), metadata=3)
        rect4 = RectWithId(x_range=(2, 5), y_range=(-1, 3), metadata=4)

        rects = [rect1, rect2, rect3, rect4]

        split = split_rectangles(rects)

        self.check_rectangle_split_result(rects, split)

        rect5 = RectWithId(x_range=(0, 3), y_range=(10, 13), metadata=5)
        rect6 = RectWithId(x_range=(0, 3), y_range=(10, 13), metadata=6)
        rect7 = RectWithId(x_range=(0, 3), y_range=(0, 13), metadata=7)

        rects2 = [rect2, rect3, rect4, rect1, rect5, rect6, rect7]
        split2 = split_rectangles(rects2)
        self.check_rectangle_split_result(rects2, split2)





if __name__ == '__main__':
    unittest.main()
