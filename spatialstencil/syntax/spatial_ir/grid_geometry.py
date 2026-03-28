from dataclasses import dataclass
from typing import Generic, TypeVar, Union

T = TypeVar('T')


@dataclass(frozen=True)
class Rectangle(Generic[T]):
    x_range: tuple[int, int, int]
    y_range: tuple[int, int, int]
    metadata: T

    def __post_init__(self):
        assert self.x_range[0] <= self.x_range[1]
        assert self.y_range[0] <= self.y_range[1]
        assert isinstance(self.x_range[0], int)
        assert isinstance(self.x_range[1], int)
        assert isinstance(self.x_range[2], int)
        assert isinstance(self.y_range[0], int)
        assert isinstance(self.y_range[1], int)
        assert isinstance(self.y_range[2], int)

    def contains_point(self, x: int, y: int) -> bool:
        """
        Return True if the point (x, y) is contained in the rectangle.

        :param x:
        :param y:
        :return:
        """
        dummy_rectangle = Rectangle((x, x+1, 1), (y, y+1, 1), metadata=self.metadata)
        return dummy_rectangle.intersects(self)
        
    def is_subset_of(self, other: 'Rectangle') -> bool:
        """
        Return True if this rectangle is a subset of another rectangle.

        :param other:
        :return:
        """
        intersection_x = intersect_ranges(self.x_range, other.x_range)
        intersection_y = intersect_ranges(self.y_range, other.y_range)
        
        if not intersection_x or not intersection_y:
            return False
        
        return ranges_equal(intersection_x, self.x_range) and ranges_equal(intersection_y, self.y_range)

    def is_equal(self, other: 'Rectangle') -> bool:
        """
        Return True if the two rectangles are equal, i.e., have the same x and y ranges.

        :param other: The other rectangle
        :return: True if the rectangles are equal, False otherwise
        """
        return ranges_equal(self.x_range, other.x_range) and ranges_equal(self.y_range, other.y_range)


    def intersection(self, other: 'Rectangle') -> Union['Rectangle', None]:
        x = intersect_ranges(self.x_range, other.x_range)
        y = intersect_ranges(self.y_range, other.y_range)
        if x and y:
            return Rectangle(x, y, self.metadata)
        return None 

    def intersects(self, other: 'Rectangle') -> bool:
        """
        Check if this rectangle intersects with another rectangle.

        :param other: The other rectangle
        :return: True if the rectangles intersect, False otherwise
        """
        return _rectangles_intersect(self, other)

    def __str__(self):
        return f"[({self.x_range[0]}, {self.x_range[1]}, {self.x_range[2]}) x ({self.y_range[0]}, {self.y_range[1]}, {self.y_range[2]}) - {self.metadata}]"


###
# RECTANGLE SPLITTING
###


def ranges_equal(r1: tuple[int, int, int], r2: tuple[int, int, int]) -> bool:
    """
    Check if two ranges generate the same sequence of values in O(1) time.
    
    Two ranges are equal if they have:
    - Same start
    - Same stride
    - Same canonicalized stop (rounded up to next multiple of stride)
    """
    start1, stop1, stride1 = r1
    start2, stop2, stride2 = r2
    
    # Quick check for identical tuples
    if r1 == r2:
        return True
    
    # Must have same start and stride
    if start1 != start2 or stride1 != stride2:
        return False
    
    # Canonicalize stops: round up to next multiple of stride
    # This ensures that stop values that generate the same sequence are equivalent
    canonical_stop1 = _canonicalize_stop(start1, stop1, stride1)
    canonical_stop2 = _canonicalize_stop(start2, stop2, stride2)
    
    return canonical_stop1 == canonical_stop2

def intersect_ranges(range1: tuple[int, int, int], range2: tuple[int, int, int]) -> None | tuple[int, int, int]:
    """
    Optimized intersection calculation using modular arithmetic.
    Assumes strides are either the same or at least one equals 1.
    
    Args:
        range1: tuple of (start, stop, stride)
        range2: tuple of (start, stop, stride)
    
    Returns:
        tuple: (intersects: bool, intersection: tuple or None)
               intersection is (start, stop, stride) if ranges intersect, None otherwise
    """
    start1, stop1, stride1 = range1
    start2, stop2, stride2 = range2
    
    # Handle empty ranges
    if start1 >= stop1 or start2 >= stop2:
        return False, None
    
    # Ensure range1 has the larger or equal stride for consistent handling
    if stride2 > stride1:
        start1, stop1, stride1, start2, stop2, stride2 = start2, stop2, stride2, start1, stop1, stride1
    
    # Case 1: Both strides are 1 (continuous integers)
    if stride1 == 1 and stride2 == 1:
        inter_start = max(start1, start2)
        inter_stop = min(stop1, stop2)
        
        if inter_start >= inter_stop:
            return None
        
        return (inter_start, inter_stop, 1)
    
    # Case 2: Both strides are equal (and > 1)
    elif stride1 == stride2:
        # Check if the ranges are aligned (same remainder modulo stride)
        if start1 % stride1 != start2 % stride1:
            return None
        
        inter_start = max(start1, start2)
        inter_stop = min(stop1, stop2)
        
        if inter_start >= inter_stop:
            return None
        
        return (inter_start, inter_stop, stride1)
    
    # Case 3: One stride is 1, the other is > 1
    else:
        if not (stride1 == 1 or stride2 == 1):
            raise NotImplementedError(f"Unsupported Strides: {stride1}, {stride2}")
        
        assert stride1 > stride2
        # stride1 > stride2 == 1 (due to our swap above)
        # Find the overlapping interval
        interval_start = max(start1, start2)
        interval_stop = min(stop1, stop2)
        
        if interval_start >= interval_stop:
            return None
        
        # Find the first element from range1 that falls within the interval
        if start1 >= interval_start:
            first_val = start1
        else:
            # Calculate how many steps needed to reach or exceed interval_start
            k = (interval_start - start1 + stride1 - 1) // stride1  # Ceiling division
            first_val = start1 + k * stride1
        
        # Check if the first value is within the interval
        if first_val >= interval_stop:
            return None
        
        # Calculate the stop of the intersection
        # Find the last element from range1 that is < interval_stop
        num_steps = (interval_stop - first_val - 1) // stride1
        last_val = first_val + num_steps * stride1
        inter_stop = last_val + stride1
        
        return (first_val, inter_stop, stride1)

def _canonicalize_stop(start: int, stop: int, stride: int) -> int:
    """
    Canonicalize a stop value by rounding up relative to the start.
    
    This ensures that stop values that generate the same sequence are equivalent.
    The canonical stop is: start + ceil((stop - start) / stride) * stride
    
    :param start: The start value of the range
    :param stop: The stop value to canonicalize
    :param stride: The stride value
    :return: Canonicalized stop value
    """
    offset = stop - start
    canonical_offset = ((offset + stride - 1) // stride) * stride
    return start + canonical_offset

def _canonicalize_range(range_tuple: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Canonicalize a range by canonicalizing its stop value.
    
    :param range_tuple: (start, stop, stride)
    :return: (start, canonicalized_stop, stride)
    """
    start, stop, stride = range_tuple
    return (start, _canonicalize_stop(start, stop, stride), stride)


def _rectangles_intersect(rect1: Rectangle, rect2: Rectangle) -> bool:
    """Check if two rectangles intersect.

    :param rect1: The first rectangle
    :param rect2: The second rectangle
    :return: True if the rectangles intersect, False otherwise
    """
    return intersect_ranges(rect1.x_range, rect2.x_range) and intersect_ranges(rect1.y_range, rect2.y_range)


def _rectangles_equal(rect1: Rectangle, rect2: Rectangle) -> bool:
    """
    Check if two rectangles are equal.

    :param rect1: The first rectangle
    :param rect2: The second rectangle
    :return: True if the rectangles are equal, False otherwise

    """
    return rect1.x_range == rect2.x_range and rect1.y_range == rect2.y_range

def _checkerboard(rect: Rectangle[T], x_stride: int, y_stride: int) -> list[Rectangle[T]]:
    """
    Decompose a rectangle into sub-rectangles whose strides match (x_stride, y_stride).

    When rect has stride 1 in a dimension and the target stride is N > 1, this produces
    N sub-rectangles (one per offset 0..N-1) that together tile rect exactly.
    If the strides already match in a dimension no extra splitting happens there.

    :param rect: The rectangle to checkerboard.
    :param x_stride: Target X stride (must be a multiple of rect.x_range[2]).
    :param y_stride: Target Y stride (must be a multiple of rect.y_range[2]).
    :return: List of rectangles with the target strides that together cover rect.
    """
    x_start, x_stop, x_s = rect.x_range
    y_start, y_stop, y_s = rect.y_range

    results: list[Rectangle[T]] = []
    for dx in range(x_stride // x_s):
        new_x_start = x_start + dx * x_s
        if new_x_start >= x_stop:
            continue
        for dy in range(y_stride // y_s):
            new_y_start = y_start + dy * y_s
            if new_y_start >= y_stop:
                continue
            results.append(Rectangle(
                (new_x_start, x_stop, x_stride),
                (new_y_start, y_stop, y_stride),
                rect.metadata,
            ))
    return results


def split_rectangle(rect1: Rectangle[T], rect2: Rectangle) -> list[Rectangle[T]]:
    """
    Split rect1 by rect2 and return the non-overlapping parts, preserving metadata.

    When the strides of rect1 and rect2 differ (and one stride is 1), rect1 is first
    checkerboarded into sub-blocks whose strides match rect2, then each sub-block is
    split using the equal-stride path.

    The equal-stride result consists of the intersection (if any) plus up to 4 rectangles:
    - Top: above the intersection (spans full width of rect1)
    - Bottom: below the intersection (spans full width of rect1)
    - Left: to the left of the intersection (spans intersection height)
    - Right: to the right of the intersection (spans intersection height)

    :param rect1: The rectangle to split
    :param rect2: The rectangle to split by
    :return: A list of non-overlapping sub-rectangles of rect1, which together
             fully cover rect1. Each of them either: is fully contained in rect2
             OR is disjoint with rect2
    """
    x1_start, x1_stop, x1_stride = rect1.x_range
    y1_start, y1_stop, y1_stride = rect1.y_range
    x2_start, x2_stop, x2_stride = rect2.x_range
    y2_start, y2_stop, y2_stride = rect2.y_range

    # If strides differ, checkerboard rect1 to match rect2's strides, then split each piece.
    # We only need to checkerboard when rect1 is the fine-grained rectangle (smaller stride);
    # when rect1 already has the larger stride, fall through to the direct intersection split.
    if x1_stride != x2_stride or y1_stride != y2_stride:
        assert x1_stride == 1 or x2_stride == 1 or x1_stride == x2_stride, \
            f"Cannot split: incompatible X strides {x1_stride} vs {x2_stride}"
        assert y1_stride == 1 or y2_stride == 1 or y1_stride == y2_stride, \
            f"Cannot split: incompatible Y strides {y1_stride} vs {y2_stride}"
        if x1_stride < x2_stride or y1_stride < y2_stride:
            target_x = max(x1_stride, x2_stride)
            target_y = max(y1_stride, y2_stride)
            pieces = _checkerboard(rect1, target_x, target_y)
            result: list[Rectangle[T]] = []
            for piece in pieces:
                result.extend(split_rectangle(piece, rect2))
            return result
        # rect1 has larger strides than rect2; fall through to the intersection split below.
    
    intersection = rect1.intersection(rect2)
    
    if not intersection:
        return [rect1]
    
    if rect1.is_equal(rect2):
        return [rect1]
    
    isec_x = _canonicalize_range(intersection.x_range)
    isec_y = _canonicalize_range(intersection.y_range)
    
    result = []
    # Top rectangle: above the intersection (full width of rect1)
    if y1_start < isec_y[0]:
        top = Rectangle(
            (x1_start, x1_stop, x1_stride),
            (y1_start, isec_y[0], y1_stride),
            rect1.metadata
        )
        assert top.is_subset_of(rect1)
        result.append(top)
    
    # Bottom rectangle: below the intersection (full width of rect1)
    if isec_y[1] < y1_stop:
        bottom = Rectangle(
            (x1_start, x1_stop, x1_stride),
            (isec_y[1], y1_stop, y1_stride),
            rect1.metadata
        )
        assert bottom.is_subset_of(rect1)
        result.append(bottom)
    
    # Left rectangle: to the left of intersection (intersection height only)
    if x1_start < isec_x[0]:
        left = Rectangle(
            (x1_start, isec_x[0], x1_stride),
            (isec_y[0], isec_y[1], y1_stride),
            rect1.metadata
        )
        assert left.is_subset_of(rect1)
        result.append(left)
    
    # Right rectangle: to the right of intersection (intersection height only)
    if isec_x[1] < x1_stop:
        right = Rectangle(
            (isec_x[1], x1_stop, x1_stride),
            (isec_y[0], isec_y[1], y1_stride),
            rect1.metadata
        )
        assert right.is_subset_of(rect1)
        result.append(right)
    
    # Intersection: the overlapping part

    assert intersection.is_subset_of(rect1)
    result.append(intersection)
    
    return result


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