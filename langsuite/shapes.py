# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import shapely.affinity
from shapely import LineString, MultiLineString, Point, Polygon


class Geometry:
    def __init__(self) -> None:
        self.shapey_geo = None

    def __repr__(self) -> str:
        return ""


class Point2D(Geometry):
    def __init__(self, *args) -> None:
        if len(args) > 2:
            raise TypeError(f"Point2D takes at most 2 arguements ({len(args)} given)")
        elif len(args) == 2:
            self.x, self.y = float(args[0]), float(args[1])
        elif len(args) == 1:
            if isinstance(args[0], Point2D) or isinstance(args[0], Point):
                self.x, self.y = args[0].x, args[0].y
            elif type(args[0]) in [list, tuple, np.ndarray] and len(args[0]) == 2:
                self.x, self.y = args[0][:2]
            else:
                raise TypeError(
                    f"Unsupport argument type for Point2D ({type(args[0])} given)"
                )
        else:
            raise TypeError("Point2D takes at least 1 argument")
        self.shapely_geo = Point(self.x, self.y)

    @property
    def modulus(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def __add__(self, other):
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float):
        return Point2D(self.x * other, self.y * other)

    def __truediv__(self, other: float):
        if other == 0.0:
            raise RuntimeError("Div Zero in Point2D")
        return Point2D(self.x / other, self.y / other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point2D):
            return False
        return self.x == other.x and self.y == other.y

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def to_wkt(self) -> str:
        return self.shapely_geo.wkt

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def rotate(self, angle, center, use_radians=False):
        """Rotation of Polygon2D geometry
        Refers to https://shapely.readthedocs.io/en/stable/manual.html#shapely.affinity.rotate

        Args:
            angle: degrees or radians by setting `use_radians=True`
            origin: (x0, y0)

        """
        if isinstance(center, Point2D):
            center = (center.x, center.y)
        # TODO
        self.shapely_geo = shapely.affinity.rotate(
            self.shapely_geo, angle, center, use_radians
        )
        self.x = self.shapely_geo.x
        self.y = self.shapely_geo.y


class Line2D(Geometry):
    def __init__(self, coords: List[Union[Point2D, Tuple[float, float]]]):
        self.coords = [Point2D(c) for c in coords]
        self.shapely_geo = LineString([c.shapely_geo for c in self.coords])

    def rotate(self, angle):
        self.shapely_geo = shapely.affinity.rotate(
            self.shapely_geo, angle, origin="center", use_radians=False
        )


class Vector2D(Point2D):
    def __init__(self, *args):
        super().__init__(*args)
        self.shapely_geo = LineString([(0, 0), (self.x, self.y)])

    def rotate(self, angle, use_radians=False):
        # TODO -angle
        new_point = shapely.affinity.rotate(
            Point(self.x, self.y), -angle, (0, 0), use_radians
        )
        self.shapely_geo = LineString([(0, 0), new_point])
        self.x = new_point.x
        self.y = new_point.y


class Polygon2D(Geometry):
    def __init__(
        self,
        coords: List[Union[Point2D, Tuple[float, float]]],
        holes: Optional[List[Union[Point2D, Tuple[float, float]]]] = None,
    ) -> None:
        self.coords = [Point2D(c) for c in coords]
        self.holes = [] if holes is None else [Point2D(c) for c in holes]
        self.shapely_geo = Polygon(
            shell=[c.shapely_geo for c in self.coords],
            holes=[c.shapely_geo for c in self.holes],
        )

    def __repr__(self) -> str:
        return "{" + ", ".join([str(c) for c in self.coords]) + "}"

    @property
    def area(self) -> float:
        return self.shapely_geo.area

    @property
    def is_closed(self) -> bool:
        return len(self.coords) > 1 and self.coords[-1] == self.coords[0]

    @property
    def length(self) -> float:
        return self.shapely_geo.length

    @property
    def centroid(self) -> Point2D:
        return Point2D(self.shapely_geo.centroid)

    @property
    def x_min(self) -> float:
        return np.min([c.x for c in self.coords])

    @property
    def x_max(self) -> float:
        return np.max([c.x for c in self.coords])

    @property
    def y_min(self) -> float:
        return np.min([c.y for c in self.coords])

    @property
    def y_max(self) -> float:
        return np.max([c.y for c in self.coords])

    @property
    def xy(self):
        return self.shapely_geo.exterior.xy

    def intersects(self, other) -> bool:
        return self.shapely_geo.intersects(other.shapely_geo)

    def rotate(self, angle, origin="center", use_radians=False):
        """Rotation of Polygon2D geometry
        Refers to https://shapely.readthedocs.io/en/stable/manual.html#shapely.affinity.rotate

        Args:
            angle: degrees or radians by setting `use_radians=True`
            origin: ['center', 'centroid', (x0, y0)]

        """
        if isinstance(origin, Point2D):
            origin = (origin.x, origin.y)
        self.shapely_geo = shapely.affinity.rotate(
            self.shapely_geo, angle, origin, use_radians
        )
        self.coords = [Point2D(c) for c in self.shapely_geo.exterior.coords]

    def to_wkt(self) -> str:
        """Well-known text representation of geometry
        https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry

        Examples:
            POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))
            POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))

        """
        return self.shapely_geo.wkt

    def to_numpy(self) -> np.array:
        return (
            np.array([p.to_numpy() for p in self.coords[:-1]])
            if self.is_closed
            else np.array([p.to_numpy() for p in self.coords])
        )

    def contains(self, other) -> bool:
        """Returns True if a Point or a Polygon is contained by the current Polygon
        Args:
            other: Point2D or Polygon2D

        Returns:
            a boolean value
        """
        if not isinstance(other, Polygon2D) and not isinstance(other, Point2D):
            raise TypeError(
                f"contains only support Polygon2D or Point2D ({type(other)} given)"
            )
        return self.shapely_geo.contains(other.shapely_geo)


class Box2D(Polygon2D):
    def __init__(self, *args) -> None:
        if len(args) == 2:
            if isinstance(args[0], Point2D) and isinstance(args[1], Point2D):
                self.ul = args[0]
                self.br = args[1]
        elif len(args) == 4:
            self.ul = Point2D(args[0], args[1])
            self.br = Point2D(args[2], args[3])
        else:
            raise TypeError(f"Box2D only takes 2 or 4 arguments ({len(args)} given)")

        self.ur = Point2D(self.ul.x, self.br.y)
        self.bl = Point2D(self.br.x, self.ul.y)
        super().__init__([self.ul, self.ur, self.br, self.bl, self.ul])

    @property
    def centroid(self) -> Point2D:
        return (self.ul + self.br) / 2.0

    def __repr__(self) -> str:
        return f"({self.ul}, {self.br})"


class Cone2D(Polygon2D):
    def __init__(
        self,
        center: Point2D,
        radius: float,
        direction: Vector2D,
        angle: float,
        use_radians: bool = False,
    ):
        super().__init__(coords=[])

        self.center = center
        self.radius = radius
        self.direction = direction
        self.angle = angle
        self.use_radians = use_radians
        (
            self.left_vector,
            self.right_vector,
            self.shapely_geo,
        ) = self._create_shapely_geo()
        self.coords = [Point2D(c) for c in self.shapely_geo.exterior.coords]

    def _create_shapely_geo(self):
        center = Point(self.center.x, self.center.y)
        circle = center.buffer(self.radius)
        # use extra length to get cone
        extra_times = 10
        left_p = self.center + self.direction * self.radius * extra_times
        left_p.rotate(
            -(self.angle / 2), center=self.center, use_radians=self.use_radians
        )
        left_line = Line2D([self.center, left_p])
        right_p = self.center + self.direction * self.radius * extra_times
        right_p.rotate(
            (self.angle / 2), center=self.center, use_radians=self.use_radians
        )
        right_line = Line2D([self.center, right_p])
        multi_line = MultiLineString(
            [left_line.shapely_geo, right_line.shapely_geo]
        ).convex_hull

        shapely_geo = circle.intersection(multi_line)
        return left_line, right_line, shapely_geo

    def rotate(self, angle, use_radians=False):
        return super().rotate(angle, self.center, use_radians)

    @property
    def centroid(self) -> Point2D:
        return self.center
