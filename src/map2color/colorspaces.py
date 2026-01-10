"""Supports for a few colorspaces.

- sRGB
- CIE XYZ tristimulus (1931)
- CIE Lab (1976)
- OkLab

Where relevant, all colorimetric coordinates use the D65 white point with a standard 2° observer.
"""

from enum import Enum
import numpy as np
from numpy.typing import ArrayLike
from gloe import transformer


class ColorSpace(str, Enum):
	RGB = "rgb"
	XYZ = "xyz"
	LAB = "lab"
	HSV = "hsv"
	OKLAB = "oklab"


D65_WHITE = np.array([0.95047, 1.00000, 1.08883])

# ColorAide constants
EPSILON = 216 / 24389  # 6^3 / 29^3
EPSILON3 = 6 / 29  # Cube root of EPSILON
KAPPA = 24389 / 27
KE = 8  # KAPPA * EPSILON = 8


def _rgb256_to_rgb64(rgb: ArrayLike) -> np.ndarray[np.float64]:
	rgb = np.atleast_2d(rgb)
	if rgb.dtype == np.uint8 or np.min_scalar_type(rgb.max()) == np.uint8:
		return (rgb / 255.0).astype(np.float64)
	else:
		return rgb.astype(np.float64)


def rgb2xyz(rgb: ArrayLike) -> np.ndarray:
	"""Standard sRGB to CIE XYZ conversion using D65 / 2° illuminant."""
	rgb = _rgb256_to_rgb64(rgb)
	rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
	M = np.array(
		[[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
	)
	return rgb @ M.T


# def xyz2rgb(xyz: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
# 	"""Standard CIE XYZ to sRGB conversion using D65 / 2° illuminant."""
# 	xyz = _rgb256_to_rgb64(xyz)
# 	M_srgb = np.array(
# 		[[0.4124564, 0.3575761, 0.1804375],
# 					[0.2126729, 0.7151522, 0.0721750],
#                        [0.0193339, 0.1191920, 0.9503041]])
# 	rgb = xyz @ M_inv.T
# 	rgb = np.where(rgb > 0.0031308, 1.055 * (rgb ** (1 / 2.4)) - 0.055, 12.92 * rgb)
# 	rgb = np.clip(rgb, 0.0, 1.0)
# 	return rgb


def xyz2lab(xyz: ArrayLike) -> np.ndarray:
	xyz = np.atleast_2d(xyz) / D65_WHITE
	f = np.where(xyz > EPSILON, np.cbrt(xyz), (KAPPA * xyz + 16) / 116)
	fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
	return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


# def xyz2lab(xyz: np.ndarray[np.float64]):
# 	"""Standard CIE XYZ to CIE LAB conversion using D65 / 2° illuminant."""
# 	xyz_scaled = xyz / D65_WHITE
# 	fx, fy, fz = np.where(xyz_scaled > EPSILON, np.power(xyz_scaled, 1 / 3), (KAPPA * xyz_scaled + 16) / 116).T
# 	L = 116.0 * fy - 16.0
# 	a = 500.0 * (fx - fy)
# 	b = 200.0 * (fy - fz)
# 	return np.column_stack([L, a, b])


def lab2xyz(lab: ArrayLike) -> np.ndarray:
	lab = np.atleast_2d(lab)
	L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
	fy = (L + 16) / 116
	fx = a / 500 + fy
	fz = fy - b / 200
	xyz_norm = np.column_stack([fx, fy, fz])
	xyz_norm = np.where(xyz_norm**3 > 0.008856, xyz_norm**3, (xyz_norm - 16 / 116) / 7.787)
	Xn, Yn, Zn = 95.047, 100.0, 108.883
	return xyz_norm * np.array([Xn, Yn, Zn])


def rgb2lab(rgb: ArrayLike) -> np.ndarray:
	"""Convert sRGB float coordinates to CIE Lab with D65 white point and 2° observer.

	Parameters:
		rgb: numpy array of shape (N, 3) with RGB values in range [0, 1]

	Returns:
		numpy array of shape (N, 3) with CIE Lab coordinates
	"""
	# Linearize sRGB (gamma correction)
	linear_rgb = np.where(rgb > 0.04045, np.power((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)

	# sRGB to XYZ transformation matrix (D65, 2° observer)
	M = np.array(
		[[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
	)

	# Convert to XYZ
	xyz = linear_rgb @ M.T

	# D65 white point (2° observer)
	Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

	# Normalize by white point
	xyz_norm = xyz / np.array([Xn, Yn, Zn])

	# Apply Lab transformation function
	delta = 6 / 29
	xyz_f = np.where(xyz_norm > delta**3, np.power(xyz_norm, 1 / 3), xyz_norm / (3 * delta**2) + 4 / 29)

	# Calculate Lab coordinates
	L = 116 * xyz_f[:, 1] - 16
	a = 500 * (xyz_f[:, 0] - xyz_f[:, 1])
	b = 200 * (xyz_f[:, 1] - xyz_f[:, 2])

	return np.column_stack([L, a, b])


def rgb2hsv(rgb):
	rgb = rgb.astype(np.float64) / 255.0
	r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

	max_val = np.max(rgb, axis=1)
	min_val = np.min(rgb, axis=1)
	delta = max_val - min_val

	V = max_val
	S = np.where(max_val == 0, 0, delta / max_val)

	H = np.zeros(len(rgb))
	mask = delta != 0

	r_max = (max_val == r) & mask
	g_max = (max_val == g) & mask
	b_max = (max_val == b) & mask

	H[r_max] = (60 * ((g[r_max] - b[r_max]) / delta[r_max]) + 360) % 360
	H[g_max] = 60 * ((b[g_max] - r[g_max]) / delta[g_max]) + 120
	H[b_max] = 60 * ((r[b_max] - g[b_max]) / delta[b_max]) + 240

	return np.column_stack([H, S, V])


def hsv2rgb(hsv: ArrayLike) -> np.ndarray:
	"""Converts HSV colors to RGB."""
	H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]

	C = V * S
	X = C * (1 - np.abs((H / 60) % 2 - 1))
	m = V - C

	rgb = np.zeros_like(hsv)

	mask = (0 <= H) & (H < 60)
	rgb[mask] = np.column_stack([C[mask], X[mask], np.zeros(np.sum(mask))])

	mask = (60 <= H) & (H < 120)
	rgb[mask] = np.column_stack([X[mask], C[mask], np.zeros(np.sum(mask))])

	mask = (120 <= H) & (H < 180)
	rgb[mask] = np.column_stack([np.zeros(np.sum(mask)), C[mask], X[mask]])

	mask = (180 <= H) & (H < 240)
	rgb[mask] = np.column_stack([np.zeros(np.sum(mask)), X[mask], C[mask]])

	mask = (240 <= H) & (H < 300)
	rgb[mask] = np.column_stack([X[mask], np.zeros(np.sum(mask)), C[mask]])

	mask = (300 <= H) & (H < 360)
	rgb[mask] = np.column_stack([C[mask], np.zeros(np.sum(mask)), X[mask]])

	rgb = rgb + m.reshape(-1, 1)
	return (rgb * 255).astype(np.uint8)


from gloe import transformer


# def convert_colorspace(colors: ArrayLike, src: ColorSpace = "rgb", target: ColorSpace = "lab") -> np.ndarray:
# 	s, t = ColorSpace(src.upper()), ColorSpace(target.upper())
# 	colors = np.at_least2d(colors)
# 	if s == t:
# 		return colors

# 	match (s, t):
# 		case (ColorSpace.RGB, ColorSpace.XYZ):
# 			return rgb2xyz(colors)
# 		case (ColorSpace.RGB, ColorSpace.LAB):
# 			return rgb2lab(colors)
# 		case (ColorSpace.RGB, ColorSpace.HSV):
# 			return rgb2hsv(colors)
# 		case (ColorSpace.XYZ, ColorSpace.RGB):
# 			xyz2rgb()
# 		case (ColorSpace.XYZ, ColorSpace.LAB):
# 			xyz2lab()
# 		case (ColorSpace.XYZ, ColorSpace.HSV):
# 			rgb2hsv(xyz2lab())
# 		case (ColorSpace.Xy)
# 	pass
