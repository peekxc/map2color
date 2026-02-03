import numpy as np


def test_basic():
	from coloraide import Color

	red2blue_dist = Color("red").distance("blue", space="srgb")

	from map2color.colorspaces import *


# Unit tests against ColorAide
def test_conversions():
	"""Test colorspace conversions against ColorAide"""
	try:
		from coloraide import Color

		# Test RGB to XYZ conversion
		test_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]])

		for rgb in test_colors:
			our_xyz = rgb2xyz(rgb.reshape(1, -1))[0]
			ca_xyz = Color("srgb", rgb / 255.0).convert("xyz-d65").coords()
			ca_xyz = np.array(ca_xyz) * 100  # ColorAide uses 0-1 range

			assert np.allclose(our_xyz, ca_xyz, atol=1e-3), f"XYZ mismatch for RGB {rgb}"

		# Test XYZ to LAB conversion
		test_xyz = np.array([[95.047, 100.0, 108.883], [41.24, 21.26, 1.93]])

		for xyz in test_xyz:
			our_lab = xyz2lab(xyz.reshape(1, -1))[0]
			ca_lab = Color("xyz-d65", xyz / 100.0).convert("lab-d65").coords()

			assert np.allclose(our_lab, ca_lab, atol=1e-2), f"LAB mismatch for XYZ {xyz}"

		print("Conversion tests passed!")

	except ImportError:
		print("ColorAide not available, skipping conversion tests")


def test_hsv_conversion():
	"""Test HSV conversion against ColorAide"""
	try:
		from coloraide import Color

		test_colors = np.array([[255, 0, 0], [0, 255, 0], [255, 255, 0], [128, 64, 192]])

		for rgb in test_colors:
			our_hsv = rgb2hsv(rgb.reshape(1, -1))[0]
			ca_hsv = Color("srgb", rgb / 255.0).convert("hsv").coords()
			ca_hsv[1:] *= 100  # ColorAide S,V in 0-1, we use 0-1 too but let's normalize

			# Handle hue wraparound
			if abs(our_hsv[0] - ca_hsv[0]) > 180:
				if our_hsv[0] > ca_hsv[0]:
					ca_hsv[0] += 360
				else:
					our_hsv[0] += 360

			assert np.allclose(our_hsv, ca_hsv, atol=1e-2), f"HSV mismatch for RGB {rgb}"

		print("HSV conversion tests passed!")

	except ImportError:
		print("ColorAide not available, skipping HSV tests")


def test_lab_distances():
	"""Test LAB distance calculations against ColorAide"""
	try:
		from coloraide import Color

		# Test Delta E 76
		lab1 = np.array([[50, 0, 0], [75, 25, -25]])
		lab2 = np.array([[60, 10, -5], [80, 20, -30]])

		our_de76 = cdist_lab(lab1, lab2)

		for i in range(len(lab1)):
			for j in range(len(lab2)):
				c1 = Color("lab-d65", lab1[i])
				c2 = Color("lab-d65", lab2[j])
				ca_de76 = c1.delta_e(c2, method="76")

				assert np.allclose(our_de76[i, j], ca_de76, atol=1e-3), f"Delta E 76 mismatch: {our_de76[i, j]} vs {ca_de76}"

		# Test Delta E 94
		our_de94 = cdist_lab_deltaE94(lab1, lab2)

		for i in range(len(lab1)):
			for j in range(len(lab2)):
				c1 = Color("lab-d65", lab1[i])
				c2 = Color("lab-d65", lab2[j])
				ca_de94 = c1.delta_e(c2, method="94")

				assert np.allclose(our_de94[i, j], ca_de94, atol=1e-2), f"Delta E 94 mismatch: {our_de94[i, j]} vs {ca_de94}"

		print("LAB distance tests passed!")

	except ImportError:
		print("ColorAide not available, skipping LAB distance tests")
