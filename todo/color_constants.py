import numpy as np
from bokeh.palettes import all_palettes

from map2color.color import hex2rgb


# np.savez_compressed("all_palettes.npz", palettes=all_palettes)
## The Hexadecimal storage is much smaller actually! 21 KB!
# np.savez_compressed("all_palettes_dict.npz", palettes=all_palettes)

ALL_PALETTES = np.load("all_palettes_dict.npz", allow_pickle=True)["palettes"].item()

palettes = {}
for k, pal in ALL_PALETTES.items():
	for i in pal.keys():
		palettes[f"{k.lower()}_{i}"] = pal[i]
np.savez_compressed("all_palettes_dict.npz", palettes=palettes)


from coloraide import Color, ease_out, ease_in


LI_viridis = Color.interpolate(ALL_PALETTES["Viridis"][256])
from map2color import map2hex

x = np.random.uniform(size=(150), low=0, high=1)
map2hex(x, palette=ALL_PALETTES["Viridis"][256])

# wut = [LI_viridis(xi).convert("srgb").coords() for xi in x]
[LI_viridis(xi).convert("hex") for xi in x]

viridis256 = ALL_PALETTES["Viridis"][256]
LI = Color.interpolate(viridis256, space="srgb")
LI(1.0).to_string(hex=True)

viridis256_rgb = hex2rgb(viridis256)
xp = np.linspace(np.min(x), np.max(x), 256)
RI = np.interp(x=x, xp=xp, fp=viridis256_rgb[:, 0] / 255.0)
GI = np.interp(x=x, xp=xp, fp=viridis256_rgb[:, 1] / 255.0)
BI = np.interp(x=x, xp=xp, fp=viridis256_rgb[:, 2] / 255.0)

np.round(np.c_[RI, GI, BI] * 255.0).clip(0, 255).astype(np.uint8)

from map2color import hex2rgb

hex2rgb(viridis256[-2])

## 119 KB
palettes = {}
for k, pal in ALL_PALETTES.items():
	for i in pal.keys():
		rgb_coords = hex2rgb(pal[i])
		palettes[f"{k.tolower()}_{i}"] = rgb_coords
np.savez_compressed("all_palettes.npz", **palettes)

rgb_coords = np.clip(np.round(np.array(wut) * 255), 0, 255).astype(np.uint8)

## TODO: rewrite begin() and __call__()
## TODO: https://colorcet.com/download/index.html
## TODO: https://www.fabiocrameri.ch/colourmaps/
# def interpolate_linear():
# if self._domain:
# 	point = self.scale(point)

# if self._padding:
# 	slope = self._padding[1] - self._padding[0]
# 	point = self._padding[0] + slope * point
# 	if not self.extrapolate:
# 		point = min(max(point, self._padding[0]), self._padding[1])

# # See if point extends past either the first or last stop
# if point < self.start:
# 	first, last = self.start, self.stops[1]
# 	return self.begin(point, first, last, 1)
# elif point > self.end:
# 	first, last = self.stops[self.length - 2], self.end
# 	return self.begin(point, first, last, self.length - 1)
# else:
# 	# Iterate stops to find where our point falls between
# 	first = self.start
# 	for i in range(1, self.length):
# 		last = self.stops[i]
# 		if point <= last:
# 			return self.begin(point, first, last, i)
# 		first = last
# Color.steps(["lch(75% 50 0)", "lch(75% 50 300)"], steps=8, space="lch", hue="longer")


# set1_pal = Color.interpolate(Set1[9])

# bs = Color.interpolate(["red", "green", "blue", "orange"], method="bspline")
# interp = Color.interpolate(["red", ease_in, "green", ease_out, "blue"])
# LI = interp.discretize(steps=150)

# interp.coordinates[0 : 0 + 2]

# p0 + (p1 - p0) * t
