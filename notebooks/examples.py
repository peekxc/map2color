import numpy as np
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.models import LinearColorMapper
from bokeh.transform import linear_cmap
from bokeh.palettes import Turbo256

output_notebook()


from map2color.color import *
from map2color.color import _transform_lerp

x = np.random.uniform(size=1250, low=-1, high=1)
y = np.random.normal(size=1250)
X = np.c_[x, y]

## Default behavior
p = figure(width=450, height=300)
p.scatter(*X.T, color=map2hex(x, palette="viridis", interp="bin"), size=4)
show(p)

## Interpolation strategy
p = figure(width=450, height=300)
p.scatter(*X.T, color=map2hex(x, palette="viridis", interp="lerp"), size=4)
show(p)


x


## Low number of bins
p = figure(width=450, height=300)
p.scatter(*X.T, color=map2hex(x, palette="viridis", nbins=5), size=4)
show(p)

p = figure(width=450, height=300)
p.scatter(*X.T, color=map2hex(x, palette="viridis", nbins=2, interp="lerp"), size=4)
show(p)


from scipy.stats import norm
from map2color.color import histeq

colors = map2color(histeq(norm.cdf(y), 256)[0], palette="viridis")
p = figure(width=450, height=300)
p.scatter(*X.T, color=colors, size=4)
show(p)


prob, bin_edges = np.histogram(y, density=True)
bin_width = np.diff(bin_edges)[0]
bin_centers = (bin_edges + (bin_width / 2))[:-1]

p = figure(width=450, height=300)
p.vbar(x=bin_centers, top=prob, width=np.diff(bin_edges)[0] * 0.90)
p.line(x=bin_centers, y=(prob.cumsum() / np.sum(prob)) * np.max(prob), line_width=2.5, line_color="red")
show(p)

from map2color.color import BokehColorPalette

palette = BokehColorPalette().lookup("viridis")
n = 1500
p = figure(width=200, height=200)
p.scatter(np.arange(n), np.arange(n), color=_expand_palette(palette, n), size=10)
show(p)

from map2color.color import histeq, hex2rgb

x = np.random.normal(size=250)
colors = map2color(histeq(x, bins=50)[0], color_pal="turbo")
p = figure(width=300, height=300)
p.scatter(x=x, y=x, color=hex2rgb(colors), size=5)
show(p)


colors = map2color(x, palette=["red", "blue"])


# from bokeh.plotting import

# s = p.line(x=np.linspace(0,1,10), y=np.linspace(0,1,10), line_color=colors)
from timeit import timeit

timeit(lambda: hex2rgb(colors), number=10)  # .36
# mapper = LinearColorMapper(low=0,high=1, palette=["red", 'blue'])
# mapper = linear_cmap(field_name="x", palette=Turbo256, low=0, high=1)

# p = figure()
# s = p.scatter(x=np.linspace(0,1,10), y=np.linspace(0,1,10), color=mapper, size=10)
# show(p)

# dir(s)

_transform_lerp([-2, -1, 0, 1, 2], "turbo")

# mapper(np.arange(10))

# c_arr = np.array(turbo)


#  # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

#   # get image histogram
#   image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
#   cdf = image_histogram.cumsum() # cumulative distribution function
#   cdf = (number_bins-1) * cdf / cdf[-1] # normalize

#   # use linear interpolation of cdf to find new pixel values
#   image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
