import argparse
import shlex
from difflib import get_close_matches
from functools import partial, reduce
from typing import Callable, ClassVar, Optional, Sequence, Literal

import numpy as np
from gloe import transformer, partial_transformer, Transformer
from numpy.typing import ArrayLike


def minmax(x: ArrayLike, feature_range: tuple = (0, 1)):
	x = np.atleast_1d(x)
	_min, _max = feature_range
	_x_min = x.min(axis=0)
	_x_rng = x.max(axis=0) - _x_min
	if np.isclose(_x_rng, 0.0):
		return np.repeat(0.5 * (_max - _min), len(x))
	scale = (_max - _min) / _x_rng
	return scale * x + _min - _x_min * scale


@partial_transformer
def clip(x: ArrayLike, min: float = 0.0, max: float = 1.0) -> np.ndarray:
	return np.clip(np.atleast_1d(x), a_min=min, a_max=max)


@partial_transformer
def affine(x: ArrayLike, scale: float = 1.0, shift: float = 0.0) -> np.ndarray:
	return scale * np.atleast_1d(x) + shift


@partial_transformer
def rescale(x: ArrayLike, method: str = "linear", factor: float = 1.0) -> np.ndarray:
	x = np.atleast_1d(x)
	if method == "linear":
		return x * factor
	elif method == "log":
		return np.log1p(x) * factor
	elif method == "invexp":
		return np.exp(x * factor)
	else:
		raise ValueError("")


@partial_transformer
def normalize(x: ArrayLike, method: Literal["minmax", "stddev", "norm", "robust"] = "minmax") -> np.ndarray:
	x = np.atleast_1d(x)
	method = method.lower()  # type: ignore
	if method == "minmax":
		return minmax(x)
	elif method == "stddev":
		return (x - x.mean()) / x.std()
	elif method == "norm":
		return x / np.sqrt(np.sum(x**2))
	elif method == "robust":
		return (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
	else:
		raise ValueError(f"Invalid method {method}")


# get_close_matches('no', ['normalize', 'affine'], n=1, cutoff=0)


# dsl = "--affine shift=1 scale=1 --normalize 'minmax' 0 1 -c a_min=0 a_max=1 -e 'hist'"
# The main categories are:
# normalize(): Preserve relationships while standardizing
# rescale(): Deliberately change relationships between values
# equalize(): Apply local/adaptive transformations
# clip(): Enforce range constraints
class Transform:
	"""Class for chaining composeable data-transformations."""

	parsers: ClassVar = {
		"normalize": argparse.ArgumentParser(prog="normalize", add_help=False, exit_on_error=False),
		"affine": argparse.ArgumentParser(prog="affine", add_help=False, exit_on_error=False),
		"clip": argparse.ArgumentParser(prog="clip", add_help=False, exit_on_error=False),
		"rescale": argparse.ArgumentParser(prog="rescale", add_help=False, exit_on_error=False),
		"equalize": argparse.ArgumentParser(prog="equalize", add_help=False, exit_on_error=False),
	}
	_initialized = False

	def __new__(cls, *args, **kwargs):
		if not cls._initialized:
			cls.parsers["normalize"].add_argument("method", choices=["minmax", "stddev", "norm", "robust"])
			cls.parsers["affine"].add_argument("--scale", type=float, default=1)
			cls.parsers["affine"].add_argument("--shift", type=float, default=0)
			cls.parsers["clip"].add_argument("a_min", type=float)
			cls.parsers["clip"].add_argument("a_max", type=float)
			cls.parsers["rescale"].add_argument("type", choices=["linear", "log", "invexp"])
			cls.parsers["rescale"].add_argument("-f", "--factor", type=float)
			cls.parsers["equalize"].add_argument("method", choices=["hist", "adapt"])
			cls._initialized = True  # Set the flag to True
		instance = super(Transform, cls).__new__(cls)
		return instance

	def __init__(self, dsl: str = ""):
		self.ops: list[tuple[str, Callable]] = []
		self.op_strs: list[str] = []
		self.parse(dsl)

	def affine(self, scale: float = 1.0, shift: float = 0.0) -> "Transform":
		"""Affine transformation: x |-> scale * x + shift."""
		self.ops.append(("affine", lambda x: scale * x + shift))
		self.op_strs += [f"affine --scale={scale} --shift={shift}"]
		return self

	def normalize(self, method: str = "minmax", **kwargs: dict) -> "Transform":
		"""Rescales data while preserving relative pairwise relationships."""
		normalizers = {
			"minmax": partial(minmax, **kwargs),
			"stddev": lambda x: (x - x.mean()) / x.std(),
			"norm": lambda x: x / np.sqrt(np.sum(x**2)),
			"robust": lambda x: (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25)),
		}
		self.ops.append(("normalize", normalizers[method]))
		self.op_strs += [f"normalize {method}"]
		return self

	def rescale(self, method: str = "linear", **kwargs: dict) -> "Transform":
		"""Globally re-scales the data, changing the relative magnitude of the distances between values."""
		rescalers = {
			"linear": lambda x: x * kwargs.get("factor", 1.0),
			"log": lambda x: np.log1p(x) * kwargs.get("factor", 1.0),
			"invexp": lambda x: np.exp(x * kwargs.get("factor", 1.0)),
		}
		self.ops.append(("rescale", rescalers[method]))
		self.op_strs += [f"rescale {method}"]
		return self

	def clip(self, a_min: float = 0.0, a_max: float = 1.0) -> "Transform":
		"""Constrain values to an interval range."""
		self.ops.append(("clip", lambda x: np.clip(x, a_min, a_max)))
		self.op_strs += [f"clip --a_min={a_min} --a_max={a_max}"]
		return self

	def equalize(self, method: str = "hist", **kwargs: dict) -> "Transform":
		"""Equalizing transformation."""
		from skimage import exposure

		equalizers = {
			# "hist_eq": lambda x: _adaptive_hist_eq(
			# 	x, kernel_size=kwargs.get("kernel_size", max(8, x.shape[0] // 8)), clip_limit=kwargs.get("clip_limit", 3.0)
			# )
			"hist": lambda x: exposure.equalize_hist(x, **kwargs),
			"adapt": lambda x: exposure.equalize_adapthist(x, **kwargs),
		}
		self.ops.append(("equalize", equalizers[method]))
		self.op_strs += [f"equalize {method}"]
		return self

	def apply(self, x: np.ndarray) -> np.ndarray:
		"""Execute all transformations in order."""
		return reduce(lambda x, op: op[1](x), self.ops, x)

	def __call__(self, x: np.ndarray) -> np.ndarray:
		"""Execute all transformations in order."""
		return self.apply(x)

	# "normalize minmax affine --scale=1 --shift=5 clip 0 1 rescale log -f2"
	def parse(self, dsl: str) -> None:
		"""Convert DSL string to Transform operations."""
		args = shlex.split(dsl)

		## Parse the command string
		current_parser = None
		command_args = []
		for arg in args:
			if arg in self.parsers:
				if current_parser:
					method_kwargs = vars(current_parser.parse_args(command_args))
					getattr(self, current_parser.prog)(**method_kwargs)
					# parsed_commands.append(current_parser.parse_args(command_args))
				current_parser = self.parsers[arg]
				command_args.clear()
			else:
				command_args.append(arg)

		## Parse the last command
		if current_parser:
			# parsed_commands.append(current_parser.parse_args(command_args))
			method_kwargs = vars(current_parser.parse_args(command_args))
			getattr(self, current_parser.prog)(**method_kwargs)

	def __repr__(self) -> str:
		if len(self.op_strs) == 0:
			return "Transform ( identity )"
		return "Transform ( " + " ".join(self.op_strs) + " )"

	def plot(self, data: ArrayLike, palette: Optional[str] = None):
		from bokeh.layouts import gridplot
		from bokeh.models import ColumnDataSource

		from .plotting import violin

		data = np.atleast_1d(data)
		data_transformed = dict(x0=data, y=np.zeros(len(data)))
		xc = data.copy()
		for i, (f_str, fp) in enumerate(self.ops):
			xc = fp(xc)
			data_transformed[f"x{i + 1}"] = xc

		cds = ColumnDataSource(data=data_transformed)
		figs = []
		for i in range(len(self.ops) + 1):
			p = violin(cds.data[f"x{i}"], width=400, height=100, tools="xbox_select")
			p.scatter(x=f"x{i}", y="y", color="black", size=5, source=cds)
			# if i > 0:
			# 	p.yaxis.axis_label = f.op_strs[i - 1]
			figs.append([p])

		return gridplot(figs)


# T = Transform()
# T.parse("normalize minmax clip 0 1 affine --scale=2 --shift=5 equalize hist clip 0 0.5")
# "N minmax C 0 1 A 2 5 E hist C 0 0.5"
# T(np.arange(10))

# docopt(help_message, argv=["--verbose", "-o", "hai.txt"], help=False, version=None, options_first=False)

# method_call = list(map(lambda a: a[0] == "-", args))
# method = ""
# method_kwargs = {}
# method_args = []
# method_calls = []
# as_method_call = (
# 	lambda: str(method) + "(" + ",".join(method_args) + ",".join([f"{k}={v}" for k, v in method_kwargs.items()]) + ")"
# )
# for arg, mc in zip(args, method_call):
# 	# method = arg.replace("-", "") if mc else
# 	if arg[0] == "-":
# 		method_calls.append(as_method_call())
# 		true_arg = arg.replace("-", "")
# 		method = self.short_map.get(true_arg, get_close_matches(true_arg, self.short_map.values(), n=1, cutoff=0)[0])
# 		method_args.clear()
# 		method_kwargs.clear()
# 	else:
# 		if "=" in arg:
# 			kv = arg.split("=")
# 			method_kwargs.update(dict([kv]))
# 		else:
# 			method_args.append(arg)
# method_calls.append(as_method_call())
# return method_calls


# def _adaptive_hist_eq(x, kernel_size, clip_limit):
# 	local_hist = uniform_filter(x, size=kernel_size)
# 	local_mean = uniform_filter(np.ones_like(x), size=kernel_size)
# 	pdfs = np.clip(local_hist / local_mean, 0, clip_limit)
# 	cdf = np.cumsum(pdfs)
# 	return (cdf - cdf.min()) / (cdf.max() - cdf.min())
