import re
from difflib import get_close_matches
from typing import Callable, Container, Iterable, Iterator, Optional, Sequence, Union
from functools import singledispatch

import numpy as np
from coloraide import Color, color

# from .color_constants import COLORS
from numpy.typing import ArrayLike

from bokeh.palettes import all_palettes

ALL_PALETTES = {k.lower(): v for k, v in all_palettes.items()}
HEX_DIGITS = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F"], dtype="<U1")


# background: linear-gradient(90deg, #3f87a6, #ebf8e1, #f69d3c);
def colors_html_box(colors: Iterable[str], interpolate: bool = False) -> str:
	"""Generates an HTML box for a sequence of colors using CSS."""

	def _gen_box(width: int, height: int, background: str) -> str:
		return f"""<div style="
				width: {width}px;
				height: {height}px;
				background: {background};
				border: 1px solid #ccc;
				margin: 0;">
		</div>
	"""

	if interpolate:
		background = f"linear-gradient(90deg, {','.join(colors)})"
		return _gen_box(512, 50, background)
	else:
		box_width = 512 // (len(colors) + 5)
		boxes = "".join([_gen_box(box_width, 50, color) for color in colors])
		return f"""<div style="display: flex; flex-direction: row; gap: 5px;">{boxes}</div>"""


class ColorPalette:
	"""Color palette."""

	def __init__(self, name: str, colors: Sequence | None = None, categorical: bool | None = None):
		self.name = name
		self.colors = np.array(self.lookup(name)) if colors is None else np.asarray(colors)
		## TODO: input validation
		self.categorical = len(self.colors) >= 12 if categorical is None else bool(categorical)

	@staticmethod
	def lookup(name: str, exact: bool = False):
		# turbo = bokeh.palettes.all_palettes["Turbo"][256]
		color_pal = name.lower()
		if color_pal in ALL_PALETTES:
			base_palette = ALL_PALETTES[color_pal]
			pal_sz = list(base_palette.keys())[-1]
		else:
			matches = re.search(r"([a-zA-Z]+)(\d*)", color_pal)
			basename, pal_sz = matches.group(1), matches.group(2)
			assert basename in ALL_PALETTES, f"Unable to find color palette '{basename}'"
			base_palette = ALL_PALETTES[basename]
		return base_palette[pal_sz] if pal_sz in base_palette else base_palette[list(base_palette.keys())[-1]]

	def bin(self, x: ArrayLike, low: float | None = None, high: float | None = None, nbins: int | None = None):
		"""Bins data values into colors via a uniform partitioning of the color palette."""
		N = len(self.colors)
		low = np.min(x) if low is None else low
		high = np.max(x) if high is None else high
		nbins = N if nbins is None else int(nbins) + 1
		bins = np.linspace(low, high, int(nbins))
		pal = self.colors if nbins >= N else self.colors[np.arange(0, N, step=N // nbins)]
		return pal[np.minimum(np.digitize(x, bins=bins), nbins - 1)]

	def interp(self, x: ArrayLike, bins: np.ndarray | None = None):
		"""Linearly interpolates given data values in RGB(A) space."""
		rgb_colors = hex2rgb(self.colors).astype(np.float32)  # need float for interp.
		bins = np.linspace(np.min(x), np.max(x), len(self.colors)) if bins is None else np.asarray(bins)
		ind = np.clip(np.digitize(x, bins) - 1, 0, len(bins) - 2)
		r = (x - bins[ind]) / (bins[ind + 1] - bins[ind])
		out_rgb = rgb_colors[ind] + (rgb_colors[ind + 1] - rgb_colors[ind]) * r[:, None]
		return rgb2hex(np.round(out_rgb).astype(np.uint8))

	def __repr__(self) -> str:
		return f"{self.name} color palette"

	def _repr_html_(self) -> str:
		# len(self.colors) > 12
		return colors_html_box(self.colors, interpolate=self.categorical)

	# def __iter__(self) -> Iterator[str]:
	# 	return iter(self.colors)


## RGB(A) here is defined as an n x (3|4) array of type np.uint8
def rgb2hex(colors: ArrayLike, prefix: str = "#") -> np.ndarray[str]:
	"""Converts RGB(A) arrays to hexadecimal strings with a chosen prefix.

	This function efficienly converts any array of RGB or RGBA colors, given as
	an n x 3 or n x 4 array of integers, into a flat array of hexadecimal string,
	encoded as fixed-width Unicode strings.

	Parameters:
		colors: n x (3|4) array of integers representing rgb(a) colors.
		prefix: preferred prefix for the output hexadecimal strings.

	Returns:
		array of hexadecimal strings.

	Examples:
		```python
		rgb2hex(np.array([[255,255,255]]))
		# array(['#FFFFFF'], dtype='<U7')

		rgb2hex(np.array([[255,255,255,100]]))
		# array(['#FFFFFF64'], dtype='<U9')
		```

	See Also:
		- `hex2rgb`, `map2hex`
	"""
	colors = colors if isinstance(colors, np.ndarray) else np.atleast_2d([colors])
	assert colors.shape[1] in {3, 4}, "Color array must be given as rgb or rgba"
	q1, r1 = np.divmod(colors, 16)
	q2, r2 = np.divmod(q1, 16)
	CR = HEX_DIGITS[r2.astype(np.int_)] + HEX_DIGITS[r1.astype(np.int_)]
	out = np.strings.add(prefix, CR[:, 0])
	for j in range(1, CR.shape[1]):
		out = np.strings.add(out, CR[:, j])
	return out


@singledispatch
def hex2rgb(colors: Union[str, np.ndarray, list], prefix: str = "#"):
	"""Converts hexadecimal strings to RGB arrays, i.e. "#FFFFFF" -> [255,255,255].

	Parameters:
		colors: hexadecimal-strings to convert to RGB(A).
		prefix: the prefix used to denote the strings in `colors` are hex. Defaults to '#'.
	"""
	...


@hex2rgb.register
def _(colors: str, prefix: str = "#"):
	return np.frombuffer(bytes.fromhex(colors.replace(prefix, "")), dtype=np.uint8)


@hex2rgb.register
def _(colors: np.ndarray, prefix: str = "#"):
	colors_hex = np.strings.replace(colors, prefix, "")
	colors_int = np.frombuffer(bytes.fromhex("".join(colors_hex)), dtype=np.uint8)
	return colors_int.reshape((len(colors_hex), len(colors_hex[0]) // 2))


@hex2rgb.register
def _(colors: list, prefix: str = "#"):
	colors_hex = np.strings.replace(colors, prefix, "")
	colors_int = np.frombuffer(bytes.fromhex("".join(colors_hex)), dtype=np.uint8)
	return colors_int.reshape((len(colors_hex), len(colors_hex[0]) // 2))


## TODO: implement plots from https://docs.bokeh.org/en/3.0.0/docs/examples/basic/data/color_mappers.html
## and equalization tips from https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html
## From http://www.janeriksolem.net/histogram-equalization-with-python-and.html
def histeq(data: ArrayLike, bins: int = 256):
	x = np.array(data)
	hist, bin_edges = np.histogram(x.flatten(), bins, density=True)
	cdf = hist.cumsum()  # cumulative distribution function
	cdf = bins * cdf / cdf[-1]  # normalize

	# use linear interpolation of cdf to find new pixel values
	data_equalized = np.interp(x.flatten(), bin_edges[:-1], cdf)
	return data_equalized.reshape(x.shape), cdf

	# Components need to be integers for hex to make sense
	# rgb = [int(x) for x in rgb]
	# return "#"+"".join(["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in rgb])


# def hist_equalize(data, low, high, bins: int = 65536):
#   low = np.min(data) if low is None else low
#   high = np.max(data) if high is None else high
#   eq_bin_edges = np.linspace(low, high, bins+1)
#   full_hist = np.bincount(np.digitize(data, eq_bin_edges))

#   ##  np.sum(full_hist != 0)
#   nhist = np.sum(full_hist != 0)

#   ## 2. Remove zeros, leaving extra element at beginning for rescale_discrete_levels
#   hist = full_hist[full_hist != 0]
#   eq_bin_centers = np.zeros(nhist+1)
#   eq_bin_centers = eq_bin_edges + (np.diff(eq_bin_edges)/2)[0]

#   ## 3. CDF scaled from 0 to 1 except for first value
#   cdf = np.cumsum(hist)
#   lo = cdf[1]
#   diff = cdf[-1] - lo
#   cdf = cdf - low / diff
#   cdf[0] = -1.0


# ## TODO: do we make a color mapper? The package was founded because we want to replace them, essentially.
# ## See: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap
# ## Also:https://docs.bokeh.org/en/3.0.0/docs/examples/basic/data/color_mappers.html
# # def color_mapper(color_pal: str = "viridis", lb: Optional[float] = 0.0, ub: Optional[float] = 1.0) -> Callable:


# https://docs.bokeh.org/en/3.0.0/docs/examples/basic/data/color_mappers.html
def map2hex(
	data: Iterable,
	palette: Union[Sequence, str] = "viridis",
	low: Optional[float] = None,
	high: Optional[float] = None,
	interp: str = "bin",
	**kwargs,
):
	"""Maps numeric values to colors in a given color palette at a given color range.

	Given a sequence of numeric values `data` and a chosen color palette, this function returns the image of the map that
	sends values in the range [`low`, `high`] to coordinates in the supplied color palette. In other words, this function
	first creates a map f : [`low`,`high`] -> `palette`, and then returns the values `f(data)`, clipping when necessary.

	Parameters:
		data: numeric values to map to colors.
		palette: color palette name or a sequence of (hex) RGB colors.
		low: lower bound to clip data values below.
		high: upper bound to clip data values above.
		interp: interpolation strategy; either 'bin' or 'lerp' (see details).

	Returns:
		ndarray of colors, given as hexadecimal strings

	The interpolation strategy determines how the RGB(a) space is interpolated.
	"""
	pal = ColorPalette(palette)
	colors = pal.bin(np.asarray(data), nbins=10) if interp == "bin" else pal.interp(x)
	return colors


def map2rgb(
	data: Iterable,
	palette: Union[Sequence, str] = "viridis",
	low: Optional = None,
	high: Optional = None,
	nbins: int = 256,
	interp: str = "bin",
	**kwargs,
):
	"""Maps numeric values to colors from a given color palette or color range.

	Parameters:
		data: numeric values to map to colors.
		palette: color palette name or a sequence of (hex) RGB colors.
		low: lower bound to clip data values below.
		high: upper bound to clip data values above.
		nbins: number of distinct colors to partition the given palette into.
		interp: interpolation strategy; either 'bin' or 'lerp' (see details).

	Returns:
		ndarray of colors, given as rgb(a) values

	See Also:
		- `rgb2hex`
		- `ColorPalette`
	"""
	colors_hex = map2hex(data, palette, low, high, nbins, interp, **kwargs)
	return hex2rgb(colors_hex)
