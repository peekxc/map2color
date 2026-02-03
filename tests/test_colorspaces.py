import numpy as np
from map2color.colorspaces import ColorSpace, rgb2xyz, rgb2hsv, rgb2lab, _rgb256_to_rgb64, xyz2lab
from map2color.color import ColorPalette
from map2color.color import hex2rgb

## Based on: https://www.nixsensor.com/free-color-converter/


## Valid to with 0.1% of coloraide's definition
def test_colorspaces():
	from coloraide import Color

	palette = ColorPalette("turbo_256")
	for col_hex, col_rgb in zip(palette.colors, hex2rgb(palette.colors)):
		xyz_truth = Color(col_hex).convert(space="xyz-d65").coords()
		xyz_test = rgb2xyz(col_rgb)
		assert np.allclose(xyz_truth, xyz_test.ravel(), atol=0.001)

		lab_truth = Color(col_hex).convert(space="lab-d65").coords()
		lab_test = xyz2lab(xyz_truth).ravel()
		lab_error = np.abs(lab_truth - lab_test.ravel()) / np.array([100, 260, 260])
		assert sum(lab_error) <= 0.001

		## https://facelessuser.github.io/coloraide/colors/oklab/
		oklab = Color(col_hex).convert(space="oklab")
		# / np.array([1.0, 0.8, 0.8])
