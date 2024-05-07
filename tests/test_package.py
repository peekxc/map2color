import numpy as np 
import map2color

def test_package():
  from skpackage import _prefixsum # native extension module
  true_sum = np.cumsum(range(10))
  test_sum = _prefixsum.prefixsum(np.arange(10))
  assert np.all(test_sum == true_sum)

def test_digitize():
  breaks = np.array([0.2, 0.4, 0.6, 0.8])
  data = np.append(breaks - 0.1, [0.9])
  assert np.all(np.digitize(data, breaks) == np.array([0,1,2,3,4]))

def test_lerp():
  from map2color.color import *
  palette = BokehColorPalette().lookup('viridis')
  palette = _lerp_palette(palette, 6)
  assert len(palette) == 6, "Palette interpolation size incorrect"
  assert np.all(_lerp_palette(palette, 6) == palette), "Linear palette interpolation is not idempotent"
