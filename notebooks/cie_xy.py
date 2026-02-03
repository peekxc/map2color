import altair as alt
import altair as alt
import numpy as np
import colorsys
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point
from shapely.ops import triangulate

# grid in xy
x = np.linspace(0.0, 0.8, 40)
y = np.linspace(0.0, 0.9, 40)
xx, yy = np.meshgrid(x, y)
eps = 1e-3
mask = (xx + yy <= 1.0) & (yy >= eps)

xm = xx[mask]
ym = yy[mask]
points = np.vstack([xm, ym]).T

## Triangulate
tri = Delaunay(points)

# xyY -> XYZ with Y = 1
Y = np.ones_like(xm)
X = xm / ym
Z = (1.0 - xm - ym) / ym

# XYZ -> linear RGB (sRGB)
M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
RGB = M @ np.vstack([X, Y, Z])
RGB = np.clip(RGB, 0.0, 1.0)


# prepare polygons for Altair
polygons = []
for simplex in tri.simplices:
	verts = [f"rgb({int(255*RGB[0,i])},{int(255*RGB[1,i])},{int(255*RGB[2,i])})" for i in simplex]
	poly_coords = [{"x": float(xm[i]), "y": float(ym[i])} for i in simplex]
	polygons.append(
		{
			"polygon": poly_coords,
			"color": verts[0],  # approximate color by first vertex
		}
	)

alt_chart = (
	alt.Chart(alt.Data(values=polygons))
	.mark_geoshape()
	.encode(color=alt.Color("color:N", scale=None))
	.transform_calculate(shape="datum.polygon")
	.properties(width=450, height=450, title="CIE 1931 xy Chromaticity Diagram (triangulated sRGB)")
)


import altair as alt

triangle = [
	{"x": 0, "y": 0},
	{"x": 1, "y": 0},
	{"x": 0.5, "y": 1},
	{"x": 0, "y": 0},  # close the triangle
]

alt.Chart(alt.Data(values=triangle)).mark_line(filled=True, color="hsl(120.523, 54.2317%, 33.000172%)").encode(
	x="x:Q", y="y:Q"
).properties(width=300, height=300, title="Simple Triangle")


def get_triangle_svg(coords: list[tuple]):
	"""Converts 3 (x,y) tuples into a normalized SVG path string for Altair.
	Expects coords like: [(x1, y1), (x2, y2), (x3, y3)]
	"""
	# 1. Find the centroid (to make it the anchor point)
	cx = sum(p[0] for p in coords) / 3
	cy = sum(p[1] for p in coords) / 3

	# 2. Shift coordinates so centroid is at (0,0)
	shifted = [(x - cx, y - cy) for x, y in coords]

	# 3. Scale to fit within -1 to 1 range
	max_val = max(max(abs(x), abs(y)) for x, y in shifted)
	scaled = [(x / max_val, y / max_val) for x, y in shifted]

	# 4. Format as SVG Path: "M x1 y1 L x2 y2 L x3 y3 Z"
	return f"M {scaled[0][0]} {scaled[0][1]} L {scaled[1][0]} {scaled[1][1]} L {scaled[2][0]} {scaled[2][1]} Z"


# Example usage:
get_triangle_svg([[0, 0], [1, 0], [0, 1]])

data = pd.DataFrame(
	{
		"x": [2, 7],  # The "anchor" positions on the chart
		"y": [2, 7],
		"color": ["red", "blue"],
		"path": [
			get_triangle_svg([(0, 0), (2, 0), (1, 3)]),  # Tall triangle
			get_triangle_svg([(0, 0), (5, 0), (2, 1)]),  # Flat triangle
		],
		"triangle_size": [2_000, 80_000],
	}
)

alt.Chart(data).mark_point(filled=True).encode(
	x=alt.X("x:Q", scale=alt.Scale(domain=[0, 10])),
	y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 10])),
	shape=alt.Shape("path:N", scale=None),  # Uses the string as the literal path
	color=alt.Color("color:N", scale=alt.Scale(domain=["red", "blue"], range=["red", "blue"])),
	size=alt.Size("triangle_size:Q", scale=None),
).properties(width=400, height=400)
# # gamma correction
# rgb = np.where(RGB <= 0.0031308, 12.92 * RGB, 1.055 * RGB ** (1 / 2.4) - 0.055)
# data = [
# 	{"x": float(xm[i]), "y": float(ym[i]), "color": f"rgb({int(255*RGB[0,i])},{int(255*RGB[1,i])},{int(255*RGB[2,i])})"}
# 	for i in range(len(xm))
# ]

# alt.Chart(alt.Data(values=data)).mark_point(size=4).encode(
# 	x=alt.X("x:Q", scale=alt.Scale(domain=[0, 0.8])),
# 	y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 0.9])),
# 	color=alt.Color("color:N", scale=None),
# ).properties(width=450, height=450, title="CIE 1931 xy Chromaticity Diagram (sRGB rendering)")


locus = [
	(0.1741, 0.0050),
	(0.1740, 0.0060),
	(0.1738, 0.0080),
	(0.1736, 0.0110),
	(0.1733, 0.0160),
	(0.1730, 0.0230),
	(0.1726, 0.0330),
	(0.1721, 0.0480),
	(0.1714, 0.0710),
	(0.1703, 0.1100),
	(0.1689, 0.1650),
	(0.1669, 0.2300),
	(0.1644, 0.2900),
	(0.1611, 0.3500),
	(0.1566, 0.4100),
	(0.1510, 0.4800),
	(0.1440, 0.5400),
	(0.1355, 0.5900),
	(0.1241, 0.6300),
	(0.1096, 0.6600),
	(0.0913, 0.6800),
	(0.0687, 0.7000),
	(0.0454, 0.7200),
	(0.0235, 0.7350),
	(0.0082, 0.7450),
	(0.0039, 0.7500),
	(0.0139, 0.7500),
	(0.0389, 0.7200),
	(0.0743, 0.6700),
	(0.1142, 0.6100),
	(0.1547, 0.5500),
	(0.1929, 0.4900),
	(0.2296, 0.4300),
]
