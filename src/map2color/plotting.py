import numpy as np
from bokeh.models import HoverTool, Span
from bokeh.plotting import figure, show
from numpy.typing import ArrayLike
from scipy.stats import gaussian_kde
from .color import hex2rgb


def violin(
	p: figure,
	data: ArrayLike,
	vertical: bool = True,
	resolution: int = 100,
	extend: float = 0.25,
	offset: float = 0.0,
	fill_color: str = "#1F77B4",
	**kwargs,
):
	"""Function to plot a violin plot."""
	_min, _max = data.min(), data.max()
	pdf = gaussian_kde(data)

	## Extend data range to avoid plotting truncation
	_e_range = (1.0 + extend) * (_max - _min)
	_e_min = (_min + _max - _e_range) / 2
	x = np.linspace(_e_min, _e_min + _e_range, resolution)
	y = pdf.evaluate(x)

	rgb_color = hex2rgb(fill_color)
	fill_alpha = 0.20 if len(rgb_color) == 3 else float(rgb_color[-1] / 250)
	if vertical:
		p.harea(y=x, x1=(y - offset), x2=-(y + offset), fill_color=fill_color, fill_alpha=fill_alpha)
	else:
		p.varea(x=x, y1=y - offset, y2=-(y + offset), fill_color=fill_color, fill_alpha=fill_alpha)
	# p.yaxis.visible = False
	# p.ygrid.grid_line_color = None
	return p


def s_curve(p: figure, p1: tuple, p2: tuple, alpha: float = 0.5, **kwargs):
	"""Plots an S-curve with zero slope its endpoints using two quadratics."""
	xs, ys = p1
	xe, ye = p2

	# Inflection point position
	infl_x = xs + alpha * (xe - xs)
	infl_y = ys + alpha * (ye - ys)

	# Control points: Control distance along x-axis for horizontal tangents
	control_dist = abs(xe - xs) * 0.3  # Adjust to control "flatness"
	cxs, cys = xs + control_dist * alpha, ys
	cxe, cye = xe - control_dist * (1 - alpha), ye

	fig_kwargs = dict(line_width=3, line_color="blue", alpha=0.8) | kwargs
	p.quadratic(x0=[xs], y0=[ys], x1=[infl_x], y1=[infl_y], cx=cxs, cy=cys, **fig_kwargs)
	p.quadratic(x0=[infl_x], y0=[infl_y], x1=[xe], y1=[ye], cx=cxe, cy=cye, **fig_kwargs)
	return p
