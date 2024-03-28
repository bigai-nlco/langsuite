# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import math

from langsuite.shapes import Point2D, Polygon2D


def euclidean_distance(p1: Point2D, p2: Point2D):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def angle_between_vectors(v1: Point2D, v2: Point2D, use_radian=False):
    radian = math.acos((v1.x * v2.x + v1.y * v2.y) / v1.modulus / v2.modulus)
    if not use_radian:
        return math.degrees(radian)
    return radian


def compute_horizonal_aov(focal_length: float, h_dim: float = 36):
    if focal_length < 0:
        raise ValueError(f"Invalid value: {focal_length}")
    return (
        180
        if focal_length == 0
        else math.degrees(2 * math.atan(h_dim / 2.0 / focal_length))
    )


def compute_rectangle_corners(
    center: Point2D, width: float, height: float, angle: float
):
    """Caculate rectangle corners

    Args:
        center: rectangle center position
        width: rectangle width
        height: rectangle height
        angle:

    Returns:
        x:
        y:
    """
    # Convert rotation angle to radians
    angle_rad = math.radians(angle % 360)

    # Compute half width and half height
    half_width = width / 2
    half_height = height / 2

    # Compute rectangle corners
    offset_bl = (-half_width, -half_height)
    offset_br = (half_width, -half_height)
    offset_tl = (-half_width, half_height)
    offset_tr = (half_width, half_height)

    # Compute rotated rectangle corners
    rotated_offset_bl = (
        offset_bl[0] * math.cos(angle_rad) - offset_bl[1] * math.sin(angle_rad),
        offset_bl[0] * math.sin(angle_rad) + offset_bl[1] * math.cos(angle_rad),
    )
    rotated_offset_br = (
        offset_br[0] * math.cos(angle_rad) - offset_br[1] * math.sin(angle_rad),
        offset_br[0] * math.sin(angle_rad) + offset_br[1] * math.cos(angle_rad),
    )
    rotated_offset_tl = (
        offset_tl[0] * math.cos(angle_rad) - offset_tl[1] * math.sin(angle_rad),
        offset_tl[0] * math.sin(angle_rad) + offset_tl[1] * math.cos(angle_rad),
    )
    rotated_offset_tr = (
        offset_tr[0] * math.cos(angle_rad) - offset_tr[1] * math.sin(angle_rad),
        offset_tr[0] * math.sin(angle_rad) + offset_tr[1] * math.cos(angle_rad),
    )

    x = [
        center.x + rotated_offset_bl[0],
        center.x + rotated_offset_br[0],
        center.x + rotated_offset_tr[0],
        center.x + rotated_offset_tl[0],
    ]
    y = [
        center.y + rotated_offset_bl[1],
        center.y + rotated_offset_br[1],
        center.y + rotated_offset_tr[1],
        center.y + rotated_offset_tl[1],
    ]
    return Polygon2D([(px, py) for (px, py) in zip(x, y)])


def rotate_point(point: Point2D, center: Point2D, alpha: float):
    """Rotate point around center by alpha degrees

    Args:
        point: point to rotate
        center: center point
        alpha: rotation angle in degrees

    Returns:
        rotated point
    """
    cos_alpha, sin_alpha = math.cos(math.radians(alpha)), math.sin(math.radians(alpha))
    x_new = (
        (point.x - center.x) * cos_alpha - (point.y - center.y) * sin_alpha + center.x
    )
    y_new = (
        (point.x - center.x) * sin_alpha + (point.y - center.y) * cos_alpha + center.y
    )
    return Point2D(x_new, y_new)


def compute_end_point(x, y, length, alpha):
    """Compute end point of a line segment

    Args:
        x: start point x
        y: start point y
        length: line segment length
        alpha: line segment angle

    Returns:
        end point
    """
    cos_alpha, sin_alpha = math.cos(math.radians(alpha)), math.sin(math.radians(alpha))
    x_new = x + length * cos_alpha
    y_new = y + length * sin_alpha
    return x_new, y_new


def is_point_inside_rotated_rect(x, y, rect_center, rect_size, rect_angle):
    """Check if point is inside rotated rectangle

    Args:
        x: point x
        y: point y
        rect_center: rectangle center
        rect_size: rectangle size
        rect_angle: rectangle rotation angle

    Returns:
        True if point is inside rectangle, False otherwise
    """
    rect_w, rect_h = rect_size
    rect_cx, rect_cy = rect_center

    x_new, y_new = rotate_point(x, y, rect_cx, rect_cy, -rect_angle)
    x1, y1 = rect_cx - rect_w / 2, rect_cy - rect_h / 2
    x2, y2 = rect_cx + rect_w / 2, rect_cy + rect_h / 2
    return x_new >= x1 and x_new <= x2 and y_new >= y1 and y_new <= y2


def intersect_segment_segment(start1, end1, start2, end2):
    """Check if two line segments intersect

    Args:
        start1: start point of first line segment
        end1: end point of first line segment
        start2: start point of second line segment
        end2: end point of second line segment

    Returns:
        True if line segments intersect, False otherwise
    """
    dir1 = (end1[0] - start1[0], end1[1] - start1[1])
    dir2 = (end2[0] - start2[0], end2[1] - start2[1])

    cross1 = dir1[0] * (start2[1] - start1[1]) - dir1[1] * (start2[0] - start1[0])
    cross2 = dir1[0] * (end2[1] - start1[1]) - dir1[1] * (end2[0] - start1[0])

    if cross1 * cross2 < 0:
        cross3 = dir2[0] * (start1[1] - start2[1]) - dir2[1] * (start1[0] - start2[0])
        cross4 = dir2[0] * (end1[1] - start2[1]) - dir2[1] * (end1[0] - start2[0])

        if cross3 * cross4 < 0:
            return True

    return False


def intersect_segment_rectangle(start, end, rect_points):
    """Check if line segment intersects rectangle

    Args:
        start: start point of line segment
        end: end point of line segment
        rect_points: rectangle corners

    Returns:
        True if line segment intersects rectangle, False otherwise
    """
    for i in range(4):
        rect_start = rect_points[i]
        rect_end = rect_points[(i + 1) % 4]
        if intersect_segment_segment(start, end, rect_start, rect_end):
            return True

    return False


def compute_triangle_coordinates(center, side_length, rotation_angle):
    """Compute triangle coordinates

    Args:
        center: triangle center
        side_length: triangle side length
        rotation_angle: triangle rotation angle

    Returns:
        triangle coordinates
    """
    height = math.sqrt(3) * side_length / 2

    x1 = -height / 3
    y1 = -side_length / 2

    x2 = -height / 3
    y2 = side_length / 2

    x3 = 2 * height / 3
    y3 = 0

    point1 = Point2D(center.x + x1, center.y + y1)
    point1.rotate(rotation_angle, center)
    point2 = Point2D(center.x + x2, center.y + y2)
    point2.rotate(rotation_angle, center)
    point3 = Point2D(center.x + x3, center.y + y3)
    point3.rotate(rotation_angle, center)

    # point_1 = rotate_point(point1_x, point1_y, center_x, center_y, rotation_angle)
    # point_2 = rotate_point(point2_x, point2_y, center_x, center_y, rotation_angle)
    # point_3 = rotate_point(point3_x, point3_y, center_x, center_y, rotation_angle)
    return [point1, point2, point3]


def is_point_in_polygen(point, polys):
    """Check if point is inside polygon

    Args:
        point: point to check
        polys: polygon points

    Returns:
        True if point is inside polygon, False otherwise
    """
    x, y = point
    crossings = 0
    for i in range(len(polys)):
        x1, y1 = polys[i]
        x2, y2 = polys[(i + 1) % len(polys)]

        if ((y1 <= y < y2) or (y2 <= y < y1)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1) + x1
        ):
            crossings += 1

    return crossings % 2 == 1


def distance_to_line(x, y, x1, y1, x2, y2):
    """Compute distance from point to line

    Args:
        x: point x
        y: point y
        x1: line start point x
        y1: line start point y
        x2: line end point x
        y2: line end point y

    Returns:
        distance from point to line
    """
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:
        param = dot / len_sq

    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = x - xx
    dy = y - yy
    return math.sqrt(dx * dx + dy * dy)


def is_point_inside_sector(point, center, radius, start_angle, end_angle):
    """Check if point is inside sector

    Args:
        point: point to check
        center: sector center
        radius: sector radius
        start_angle: sector start angle
        end_angle: sector end angle

    Returns:
        True if point is inside sector, False otherwise
    """
    vector_to_point = (point[0] - center[0], point[1] - center[1])
    vector_to_point_length = math.sqrt(
        vector_to_point[0] * vector_to_point[0]
        + vector_to_point[1] * vector_to_point[1]
    )

    if vector_to_point_length <= radius:
        vector_to_point_angle = math.atan2(vector_to_point.y, vector_to_point.x)
        if start_angle <= vector_to_point_angle <= end_angle:
            return True

    return False


def is_line_intersect_sector(line, center, radius, start_angle, end_angle):
    """Check if line intersects sector

    Args:
        line: line to check
        center: sector center
        radius: sector radius
        start_angle: sector start angle
        end_angle: sector end angle

    Returns:
        True if line intersects sector, False otherwise
    """
    if is_point_inside_sector(
        line[0], center, radius, start_angle, end_angle
    ) or is_point_inside_sector(line[1], center, radius, start_angle, end_angle):
        return True

    line_vector = (line[1][0] - line[0][0], line[1][1] - line[0][1])

    start_to_line = (line.start.x - center.x, line.start.y - center.y)
    end_to_line = (line.end.x - center.x, line.end.y - center.y)

    angle_start = angle_between_vectors(start_to_line, line_vector)
    angle_end = angle_between_vectors(end_to_line, line_vector)

    if angle_start + angle_end >= math.fabs(start_angle - end_angle):
        return True

    return False
