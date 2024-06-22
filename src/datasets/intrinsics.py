from dataclasses import dataclass
import numpy


@dataclass
class Intrinsics:
    width: int
    height: int
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float

    def scale(self, factor: float):
        nw = round(self.width * factor)
        nh = round(self.height * factor)
        sw = nw / self.width
        sh = nh / self.height
        self.focal_x *= sw
        self.focal_y *= sh
        self.center_x *= sw
        self.center_y *= sh
        self.width = int(nw)
        self.height = int(nh)

    def to_matrix(self):
        return numpy.array([
            [self.focal_x, 0, self.center_x],
            [0, self.focal_y, self.center_y],
            [0, 0, 1]
        ])

    def __repr__(self):
        return (f"Intrinsics(width={self.width}, height={self.height}, "
                f"focal_x={self.focal_x}, focal_y={self.focal_y}, "
                f"center_x={self.center_x}, center_y={self.center_y})")
