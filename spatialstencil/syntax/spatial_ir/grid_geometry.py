from dataclasses import dataclass


@dataclass(frozen=True)
class Rectangle[T]:
    x_range: tuple[int, int]
    y_range: tuple[int, int]
    metadata: T


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
        new_rectangles.append(Rectangle(
            x_range=x_overlap,
            y_range=y_overlap,
            metadata=rect2.metadata
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
    i = 0
    while i < len(rectangles):
        rect1 = rectangles[i]
        has_split = False
        for j in range(len(rectangles)):
            if i != j:
                rect2 = rectangles[j]
                if _rectangles_intersect(rect1, rect2) and not _rectangles_equal(rect1, rect2):
                    # Split rect1 by rect2
                    split_result = split_rectangle(rect1, rect2)
                    # Replace rect1 with the resulting smaller rectangles
                    rectangles.pop(i)
                    rectangles.extend(split_result)
                    has_split = True
                    break
        if not has_split:
            i += 1

    # Postcondition:
    # Assert that there are no intersections left (except for equal rectangles)
    assert all(not _rectangles_intersect(rect1, rect2) or _rectangles_equal(rect1, rect2)
               for rect1 in rectangles for rect2 in rectangles)

    return rectangles


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