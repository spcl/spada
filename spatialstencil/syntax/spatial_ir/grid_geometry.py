from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar('T')


@dataclass(frozen=True)
class Rectangle(Generic[T]):
    x_range: tuple[int, int]
    y_range: tuple[int, int]
    metadata: T

    def __post_init__(self):
        assert self.x_range[0] <= self.x_range[1]
        assert self.y_range[0] <= self.y_range[1]
        assert isinstance(self.x_range[0], int)
        assert isinstance(self.x_range[1], int)
        assert isinstance(self.y_range[0], int)
        assert isinstance(self.y_range[1], int)

    def contains_point(self, x: int, y: int) -> bool:
        """
        Return True if the point (x, y) is contained in the rectangle.

        :param x:
        :param y:
        :return:
        """
        return self.x_range[0] <= x < self.x_range[1] and self.y_range[0] <= y < self.y_range[1]

    def is_subset_of(self, other: 'Rectangle') -> bool:
        """
        Return True if this rectangle is a subset of another rectangle.

        :param other:
        :return:
        """
        return other.x_range[0] <= self.x_range[0] and other.x_range[1] >= self.x_range[1] and \
               other.y_range[0] <= self.y_range[0] and other.y_range[1] >= self.y_range[1]

    def is_equal(self, other: 'Rectangle') -> bool:
        """
        Return True if the two rectangles are equal, i.e., have the same x and y ranges.

        :param other: The other rectangle
        :return: True if the rectangles are equal, False otherwise
        """
        return self.x_range == other.x_range and self.y_range == other.y_range

    def intersects(self, other: 'Rectangle') -> bool:
        """
        Check if this rectangle intersects with another rectangle.

        :param other: The other rectangle
        :return: True if the rectangles intersect, False otherwise
        """
        return _rectangles_intersect(self, other)

    def __str__(self):
        return f"[{self.x_range[0]}, {self.x_range[1]}) x [{self.y_range[0]}, {self.y_range[1]}) - {self.metadata}"


###
# RECTANGLE SPLITTING
###

def _ranges_overlap(range1: tuple[int, int], range2: tuple[int, int]) -> bool:
    """
    Check if two ranges overlap, considering exclusive upper bound.

    :param range1: The first range
    :param range2: The second range
    :return: True if the ranges overlap, False otherwise
    """
    return not (range1[1] <= range2[0] or range2[1] <= range1[0])


def _rectangles_intersect(rect1: Rectangle, rect2: Rectangle) -> bool:
    """Check if two rectangles intersect.

    :param rect1: The first rectangle
    :param rect2: The second rectangle
    :return: True if the rectangles intersect, False otherwise
    """
    return _ranges_overlap(rect1.x_range, rect2.x_range) and _ranges_overlap(rect1.y_range, rect2.y_range)


def _rectangles_equal(rect1: Rectangle, rect2: Rectangle) -> bool:
    """
    Check if two rectangles are equal.

    :param rect1: The first rectangle
    :param rect2: The second rectangle
    :return: True if the rectangles are equal, False otherwise

    """
    return rect1.x_range == rect2.x_range and rect1.y_range == rect2.y_range


def split_rectangle(rect1: Rectangle, rect2: Rectangle) -> list[Rectangle]:
    """
    Split rect1 by rect2 and return the non-overlapping parts, preserving metadata.

    :param rect1: The rectangle to split
    :param rect2: The rectangle to split by
    :return: A list of non-overlapping rectangles
    """
    new_rectangles = []

    # Get the intersection area
    x_overlap = (max(rect1.x_range[0], rect2.x_range[0]), min(rect1.x_range[1], rect2.x_range[1]))
    y_overlap = (max(rect1.y_range[0], rect2.y_range[0]), min(rect1.y_range[1], rect2.y_range[1]))

    # Only add the overlap rectangles if the ranges are not empty
    if x_overlap[0] < x_overlap[1] and y_overlap[0] < y_overlap[1]:
        # Create two overlap rectangles, one for each original identity
        new_rectangles.append(Rectangle(
            x_range=x_overlap,
            y_range=y_overlap,
            metadata=rect1.metadata
        ))

    # Now create the remaining parts of rect1 that do not overlap
    if rect1.x_range[0] < x_overlap[0]:
        new_rectangles.append(Rectangle(
            x_range=(rect1.x_range[0], x_overlap[0]),
            y_range=rect1.y_range,
            metadata=rect1.metadata
        ))

    if rect1.x_range[1] > x_overlap[1]:
        new_rectangles.append(Rectangle(
            x_range=(x_overlap[1], rect1.x_range[1]),
            y_range=rect1.y_range,
            metadata=rect1.metadata
        ))

    if rect1.y_range[0] < y_overlap[0]:
        new_rectangles.append(Rectangle(
            x_range=x_overlap,
            y_range=(rect1.y_range[0], y_overlap[0]),
            metadata=rect1.metadata
        ))

    if rect1.y_range[1] > y_overlap[1]:
        new_rectangles.append(Rectangle(
            x_range=x_overlap,
            y_range=(y_overlap[1], rect1.y_range[1]),
            metadata=rect1.metadata
        ))

    return [rect for rect in new_rectangles if rect.x_range[0] < rect.x_range[1] and rect.y_range[0] < rect.y_range[1]]


def split_rectangles(rectangles: list[Rectangle]) -> list[Rectangle]:
    """
    Main function to split rectangles until no intersections remain.

    :param rectangles: A list of rectangles to split
    :return: A list of non-overlapping rectangles (preserving metadata)
    """
    result = rectangles.copy()
    i = 0
    while i < len(result):
        has_split = False
        rect1 = result[i]
        for j in range(len(result)):
            if i != j:
                rect2 = result[j]
                if rect1.intersects(rect2) and not rect1.is_equal(rect2):
                    # Split rect1 by rect2
                    split_result = split_rectangle(rect1, rect2)
                    # Replace rect1 with the resulting smaller rectangles
                    result.pop(i)
                    result.extend(split_result)
                    has_split = True
                    break

        if not has_split:
            # Loop invariant: No non-equal intersections between rectangles for indices <= i
            assert all(not result[k].intersects(result[j]) or result[k].is_equal(result[j])
                       for j in range(len(result)) for k in range(i+1))
            i += 1

    # Post-condition:
    # Assert that there are no intersections left (except for equal rectangles)
    assert all(not rect1.intersects(rect2) or rect1.is_equal(rect2)
               for rect1 in result for rect2 in result)

    return result


###
# GROUPING
###

def group_rectangles_by_domain(rects: list[Rectangle]) -> list[list[Rectangle]]:
    """
    Group rectangles that have the same domain.

    :param rects: A list of rectangles to group
    :return: A list containing lists of rectangles with the same domain
    """
    merged = []
    for decl in rects:
        found = False
        for m in merged:
            if m[0].x_range == decl.x_range and m[0].y_range == decl.y_range:
                m.append(decl)
                found = True
                break
        if not found:
            merged.append([decl])

    return merged