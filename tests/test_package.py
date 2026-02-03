import numpy as np
from numpy.typing import ArrayLike
import map2color
from map2color.color import colors_html_box, hex2rgb, rgb2hex, ColorPalette
from map2color.color import map2rgb
from map2color.distance import pdist_euclidean
from map2color.plotting import violin


def test_color_palette():
	rng = np.random.default_rng(1234)
	pal = ColorPalette("turbo")
	x = np.sort(rng.uniform(size=150, low=0.0, high=1.0))
	cb = pal.bin(x, nbins=10)
	ci = pal.interp(x)
	assert isinstance(pal._repr_html_(), str)
	assert isinstance(cb, np.ndarray) and cb.size == 150 and cb.dtype == "<U7"
	assert isinstance(ci, np.ndarray) and ci.size == 150 and cb.dtype == "<U7"


def test_rgb_hex_conversion():
	from map2color import hex2rgb, rgb2hex

	rgb_col = np.array([[252, 222, 164], [193, 41, 179], [94, 75, 3], [106, 141, 115]], dtype=np.uint8)
	hex_col = np.strings.upper(np.array(["#fcdea4", "#c129b3", "#5e4b03", "#6a8d73"], dtype="<U7"))
	assert np.all(rgb2hex(rgb_col) == hex_col)
	assert np.all(hex2rgb(hex_col) == rgb_col)


## This should map any values in [0,1] to color palettes
def test_mapping():
	from map2color import transform
	# transform.


def test_transform():
	from map2color.transform import Transform
	from map2color.plotting import violin

	rng = np.random.default_rng(1234)
	x = rng.uniform(size=15, low=0, high=5)
	f = Transform().rescale(method="invexp").normalize()
	x2 = f(x)

	from bokeh.plotting import figure, show
	from bokeh.models import ColumnDataSource
	from bokeh.io import output_notebook
	from bokeh.layouts import gridplot

	output_notebook()

	data = dict(x0=x, y=np.zeros(len(x)))
	xc = x.copy()
	for i, (f_str, fp) in enumerate(f.ops):
		xc = fp(xc)
		data[f"x{i+1}"] = xc

	cds = ColumnDataSource(data=data)
	figs = []

	for i in range(3):
		p = violin(cds.data[f"x{i}"], width=400, height=100, tools="xbox_select")
		p.scatter(x=f"x{i}", y="y", color="red", size=5, source=cds)
		# if i > 0:
		# 	p.yaxis.axis_label = f.op_strs[i - 1]
		figs.append([p])

	show(gridplot(figs))
	# show(p2)


def test_dist_flow():
	from map2color.plotting import s_curve, violin
	from bokeh.plotting import figure, show
	from bokeh.io import output_notebook
	from itertools import pairwise

	output_notebook()
	rng = np.random.default_rng()
	d1 = rng.normal(3.0, 1.0, size=100)
	d2 = rng.normal(5.0, 0.75, size=100)
	d3 = rng.exponential(scale=0.25, size=100) + 0.50

	b1 = np.quantile(d1, [0, 0.25, 0.5, 0.75, 1.0])
	b2 = np.quantile(d2, [0, 0.25, 0.5, 0.75, 1.0])
	b3 = np.quantile(d3, [0, 0.25, 0.5, 0.75, 1.0])

	D, B = [d1, d2, d3], [b1, b2, b3]

	p = figure(width=550, height=225)
	for i, (Bs, Be) in enumerate(pairwise(B)):
		for bs, be in zip(Bs, Be):
			s_curve(p, (i, bs), (i + 1, be))
			print(f"{(i,i+1)}")

	violin(p, data=d1, vertical=True)
	violin(p, data=d2, offset=-1.0, vertical=True)
	violin(p, data=d3, offset=-2.0, vertical=True)
	show(p)


def test_colorspaces():
	from map2color.color import hex2rgb

	colors = ColorPalette("turbo")._palette
	colors_rgb = hex2rgb(colors)

	from map2color.colorspaces import rgb2lab

	colors_lab = rgb2lab(colors_rgb / 255.0)

	from coloraide import Color

	Color(color=[203, 43, 3])
	c = Color("srgb", [203 / 255.0, 43 / 255.0, 3 / 255.0])
	c.convert("lab-d65").coords()
	rgb2lab(np.array([[203, 43, 3]]) / 255.0) * np.array([100, 256, 256]) - np.array([0, 128, 128])


def test_landmarks():
	data = np.linspace(0, 100, 100)
	colors = ColorPalette("turbo").interp(data)
	labcolors = rgb2lab(hex2rgb(colors))
	np.diff(pdist_euclidean(labcolors))

	# from landmark import mds


# from IPython.display import display_html
