
import numpy as np
from typing import Union, Iterable, Optional, Callable, Sequence
from difflib import get_close_matches
import re
from .color_constants import COLORS

def ensure(expr_res: bool, message: str):
	if not expr_res: 
		raise ValueError(message)

def is_hex_seq(S) -> bool:
	S = np.array(S)
	all_correct_len = np.all([len(s) in [6,7,8,9] for s in S])
	if not all_correct_len: 
		return False 
	else: 
		hex_str = ''.join(S).replace("#",'').lower()
		all_hex_chars = all((c in 'abcdef0123456789' for c in hex_str))
		return all_hex_chars

def is_rgb_seq(S) -> bool: 
	S = np.atleast_2d(S)
	valid_shape = S.shape[1] in [3,4]
	if not valid_shape: 
		return False 
	else: 
		valid_values = S.dtype == np.uint8 or (np.all(S.flatten() < 256) and np.all(S.flatten() >= 0))
		return valid_values

class BokehColorPalette():
	def __init__(self):
		from bokeh.palettes import all_palettes
		self.palettes = { k.lower() : v for k,v in all_palettes.items() }

	def lookup(self, color_pal: str):
		# turbo = bokeh.palettes.all_palettes["Turbo"][256]
		color_pal = color_pal.lower()
		if color_pal in self.palettes:
			base_palette = self.palettes[color_pal]
			pal_sz = list(base_palette.keys())[-1]
		else:
			matches = re.search(r"([a-zA-Z]+)(\d*)", color_pal)
			basename, pal_sz = matches.group(1), matches.group(2)
			assert basename in self.palettes, f"Unable to find color palette '{basename}'"
			base_palette = self.palettes[basename]
		return base_palette[pal_sz] if pal_sz in base_palette else base_palette[list(base_palette.keys())[-1]]

	def __repr__(self) -> str:
		return f"Palettes: {', '.join(self.palettes.keys())}"

## RGB(A) here is defined as an n x (3|4) array of type np.uint8 
def rgb2hex(c: np.ndarray):
	''' [255,255,255] -> "#FFFFFF" '''
	c = c if isinstance(c, np.ndarray) else np.atleast_2d([c])
	c = c.astype(np.uint8)
	assert c.shape[1] in [3,4], "Color array must be given as rgb or rgba"
	rgb_str = '#%02x%02x%02x' if c.shape[1] == 3 else '#%02x%02x%02x%02x'
	return np.array([rgb_str % (r,g,b) for r,g,b in c])

	# c_str = np.array2string(c, formatter={'int_kind' : lambda x: "%.2x" % x }, separator='')
	# c_hex = np.char.add('#', np.array(c_str.replace('[','').replace(']','').replace('\n','').split(' ')))
	# return np.take(c_hex, 0) if len(c_hex) == 1 else c_hex

## Largely from: https://bsouthga.dev/posts/color-gradients-with-python
def color2hex(colors: Iterable) -> np.ndarray:
	''' Given the name of color or an iterable of color names, returns the closest corresponding hexadecimal color representation '''
	if isinstance(colors, str):
		colors = colors.lower()
		if colors in COLORS:
			return COLORS[colors]
		elif is_hex_seq([colors]):
			return colors
		else: 
			# import editdistance
			keys = list(COLORS.keys())
			closest_color = get_close_matches(colors, keys, n = 1, cutoff = 0.0)
			return COLORS[closest_color]
			# raise ValueError("Invalid input detected")
	else:
		if is_hex_seq(colors): 
			return colors 
		else:
			return np.array([color2hex(c) for c in colors])

def hex2rgb(colors: Union[Iterable, str]):
	''' "#FFFFFF" -> [255,255,255] '''
	if isinstance(colors, str): # int(hex_str[i:i+2], 16) for i in range(1,6,2)
		hex_str = colors[1:] if colors[0] == "#" else colors
		return np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)
	else:
		# return np.array([hex2rgb(h) for h in hex_str], dtype=np.uint8)
		d = (len(colors[0]) - int(colors[0][0] == '#')) // 2
		n = len(colors)
		return np.frombuffer(bytes.fromhex(''.join(colors).replace('#','')), dtype=np.uint8).reshape((n, d))

## TODO: implement plots from https://docs.bokeh.org/en/3.0.0/docs/examples/basic/data/color_mappers.html
## and equalization tips from https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html
## From http://www.janeriksolem.net/histogram-equalization-with-python-and.html
def histeq(data: np.ndarray, bins: int = 256):
	x = np.array(data)
	hist, bin_edges = np.histogram(x.flatten(), bins, density=True)
	cdf = hist.cumsum() # cumulative distribution function
	cdf = bins * cdf / cdf[-1] # normalize

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

def digitize_modulo(x: np.ndarray, bins: np.ndarray):
	"""Bins values in 'x' into bins by calling digitize, but also returns position information about bins""" 
	ind = np.digitize(x, bins=bins, right=False)
	pred_value = bins[np.clip(ind - 1, 0, len(bins)-1)]
	succ_value = bins[np.clip(ind, 0, len(bins)-1)]
	bin_width = (succ_value - pred_value)
	denom = np.reciprocal(np.where(bin_width > 0, bin_width, 1.0))
	return ind, denom*(x - pred_value)

def color_mapper(color_pal: str = 'viridis', lb: Optional[float] = 0.0, ub: Optional[float] = 1.0) -> Callable:
	import bokeh
	ensure(isinstance(color_pal, str), "Must be string input")
	color_pal = color_pal.lower()
	bokeh_palettes = { p.lower() : p for p in dir(bokeh.palettes) if p[0] != "_" }
	ensure(isinstance(color_pal, str), f"Unknown color palette '{color_pal}'")
	
	## Color palette should be end up as a numpy array of hex strings; if function, get 255 colors and convert
	COLOR_PAL = getattr(bokeh.palettes, bokeh_palettes[color_pal.lower()])
	N_COLORS = len(COLOR_PAL)
	VALUE_BINS = np.linspace(lb, ub, N_COLORS)

	## 
	def _color_map(x: Iterable, output: str = ["rgba", "hex"], strategy: str = ["bin", "lerp"]) -> np.ndarray:
		if strategy == ["bin", "lerp"] or strategy == "bin":
			ind = np.digitize(np.clip(x, a_min=lb, a_max=ub), bins=VALUE_BINS)
			ind = np.clip(ind, 0, N_COLORS-1) 	## bound from above and below
			return COLOR_PAL[ind] if output == "hex" else hex2rgb(COLOR_PAL[ind])
		elif strategy == "lerp":
			color_values = np.c_[hex2rgb(COLOR_PAL)/255.0, np.ones(255)]
			ind, rel = digitize_modulo(np.clip(x, a_min=lb, a_max=ub), VALUE_BINS)
			ind = np.clip(ind, 0, N_COLORS-1)
			succ_values = color_values[np.where(ind < (N_COLORS-1), ind+1, ind)]
			x_unit = color_values[ind] + rel[:,np.newaxis] * (succ_values - color_values[ind])
			x_rgba = np.clip(np.round(x_unit * 255).astype(np.uint16), 0, 255)
			return rgb2hex(x_rgba) if output == "hex" else x_rgba
		else: 
			raise ValueError(f"Unknown interpolation strategy '{strategy}' given.")
	return _color_map

## Assumes the data have been clipped
def _transform_lerp(data: np.ndarray, palette: Sequence, bins: Sequence = None):
	"""Linearly interpolates a given numeric data array onto a given palette in RGB(A) space using the supplied bins"""
	bins = np.linspace(np.min(data), np.max(data), len(palette)) if bins is None else bins
	assert len(bins) == len(palette), "The number of bins should match the size of the supplied color palette"
	N_COLORS = len(palette) 
	rgb_palette = hex2rgb(palette)
	ind, rel = digitize_modulo(data, bins)
	ind = np.clip(ind, 0, N_COLORS-1)
	succ_values = rgb_palette[np.where(ind < (N_COLORS-1), ind+1, ind)]
	x_unit = rgb_palette[ind] + rel[:,np.newaxis] * (succ_values - rgb_palette[ind])
	x_rgba = np.clip(np.round(x_unit * 255).astype(np.uint8), 0, 255)
	return rgb2hex(x_rgba)

## Assumes the data have been clipped
def _transform_bin(data: np.ndarray, palette: Sequence, bins: Sequence):
	"""Bins a given numeric data array onto a given palette in RGB(A) space using the supplied bins"""
	bins = np.linspace(np.min(data), np.max(data), len(palette)-1) if bins is None else bins
	assert len(bins) == (len(palette) - 1), f"The number of bins ({len(bins)}) should match the size of the supplied color palette ({len(palette)}) - 1"
	ind = np.digitize(data, bins=bins)
	return np.array(palette)[ind]

def _lerp_palette(palette: Sequence, size: int) -> Sequence:
	"""Given a color palette (hex strings), this function linearly interpolates the palette to a given size via linear interpolation in RGB space"""
	assert is_hex_seq(palette), "Expects the color palette to be an array of hexadecimal strings"
	N_COLORS = len(palette) 
	rgb_palette = hex2rgb(palette) / 255
	key_points = np.linspace(0.0, 1.0, size)
	pal_points = np.linspace(0.0, 1.0, len(palette))
	ind, rel = digitize_modulo(key_points, pal_points)
	ind = np.clip(ind-1, 0, N_COLORS-1)
	succ_values = rgb_palette[np.where(ind < (N_COLORS-1), ind+1, ind)]
	x_unit = rgb_palette[ind] + rel[:,np.newaxis] * (succ_values - rgb_palette[ind])
	x_rgba = np.clip(np.round(x_unit * 255).astype(np.uint8), 0, 255)
	return rgb2hex(x_rgba)

# https://docs.bokeh.org/en/3.0.0/docs/examples/basic/data/color_mappers.html
def map2hex(
	data: Iterable = None, 
	palette: Union[Sequence, str] = 'viridis', 
	low: Optional = None, 
	high: Optional = None,
	nbins: int = 256,
	interp: str = "bin",
	**kwargs
):
	"""Maps numeric values to colors from a given color palette or color range.
	
	Parameters: 
		data = numeric values to map to colors.
		palette = color palette name or a sequence of (hex) RGB colors.
		low = lower bound to clip data values below.
		high = upper bound to clip data values above.
		nbins = number of distinct colors to partition the given palette into. 
		interp = interpolation strategy; either 'bin' or 'lerp' (see details).

	Returns: 
		ndarray of colors, given as hexadecimal strings

	The interpolation strategy determines how the RGB(a) space is interpolated. 
	"""
	## Clip, digitize (bin), and perform the index mapping
	if isinstance(palette, str):
		BP = BokehColorPalette()
		palette = np.array(BP.lookup(palette))
	else: 
		assert isinstance(palette, Sequence) or isinstance(palette, np.ndarray), "Invalid color palette given; must be a sequence or array of colors."
		palette = color2hex(palette)
	
	## Linearly interpolate/extend the palette to the given bin size
	palette = _lerp_palette(palette, size = nbins)

	## Applying clipping bounds, compute bins, and do the mapping
	lb = np.min(data) if low is None else low
	ub = np.max(data) if high is None else high
	data = np.clip(data, a_min=lb, a_max=ub)
	
	## Apply the given interpolation strategy
	if interp == "bin":
		bin_centers = np.linspace(lb,ub,nbins+1,endpoint=True)[1:-1]
		colors = _transform_bin(data, palette=palette, bins=bin_centers)
	elif interp == "lerp":
		bin_centers = np.linspace(lb,ub,nbins+1,endpoint=True)[1:-1]
		colors = _transform_lerp(data, palette=palette, bins=bin_centers)
	else: 
		raise ValueError(f"Invalid interpolation strategy supplied '{str(interp)}'; should be one of 'bin' or 'lerp'.")

	return colors
	
	## TODO: 
	# bins = np.linspace(lb, ub, len(palette))
	
	# ind = np.digitize(data, bins=np.linspace(lb, ub, len(palette))) - 1
	# assert np.min(ind) >= 0 and np.max(ind) < len(palette), "Mapping failed"
	# return palette[ind]



def map2rgb(
	data: Iterable = None, 
	palette: Union[Sequence, str] = 'viridis', 
	low: Optional = None, 
	high: Optional = None,
	nbins: int = 256,
	interp: str = "bin",
	**kwargs
):
	"""Maps numeric values to colors from a given color palette or color range.
	
	Parameters: 
		data = numeric values to map to colors.
		palette = color palette name or a sequence of (hex) RGB colors.
		low = lower bound to clip data values below.
		high = upper bound to clip data values above.
		nbins = number of distinct colors to partition the given palette into. 
		interp = interpolation strategy; either 'bin' or 'lerp' (see details).

	Returns: 
		ndarray of colors, given as rgb(a) values 
	"""
	colors_hex = map2rgb(data, palette, low, high, nbins, interp, **kwargs)
	return hex2rgb(colors_hex)
