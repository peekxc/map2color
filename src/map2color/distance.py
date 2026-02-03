import numpy as np
from numpy.typing import ArrayLike


## From: https://stackoverflow.com/questions/42138681/faster-numpy-solution-instead-of-itertools-combinations
def _combs(n: int, k: int) -> np.ndarray:
	if n < k:
		return np.empty(shape=(0,), dtype=int)
	a = np.ones((k, n - k + 1), dtype=int)
	a[0] = np.arange(n - k + 1)
	for j in range(1, k):
		reps = (n - k + j) - a[j - 1]
		a = np.repeat(a, reps, axis=1)
		ind = np.add.accumulate(reps)
		a[j, ind[:-1]] = 1 - reps[1:]
		a[j, 0] = j
		a[j] = np.add.accumulate(a[j])
	return a


def pdist_euclidean(X: ArrayLike):
	"""Pairwise euclidean distances for points in `X`."""
	X = np.atleast_2d(X).astype(np.float64)
	n = len(X)
	D = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1))
	return D[np.triu_indices(n, k=1)]


def cdist_euclidean(XA: ArrayLike, XB: ArrayLike):
	"""Cross euclidean distances between colors in XA and XB."""
	XA = np.atleast_2d(XA).astype(np.float64)
	XB = np.atleast_2d(XB).astype(np.float64)
	diff = XA[:, np.newaxis, :] - XB[np.newaxis, :, :]
	return np.sqrt(np.sum(diff**2, axis=2))


def pdist_rgb(rgb):
	"""Pairwise RGB euclidean distances"""
	return pdist_euclidean(rgb)


def cdist_rgb(rgb1, rgb2):
	"""Cross RGB euclidean distances"""
	return cdist_euclidean(rgb1, rgb2)


def pdist_xyz(xyz):
	"""Pairwise XYZ euclidean distances"""
	return pdist_euclidean(xyz)


def cdist_xyz(xyz1, xyz2):
	"""Cross XYZ euclidean distances"""
	return cdist_euclidean(xyz1, xyz2)


def pdist_lab(lab):
	"""Pairwise CIELAB euclidean distances (Delta E 76)"""
	return pdist_euclidean(lab)


def cdist_lab(lab1, lab2):
	"""Cross CIELAB euclidean distances (Delta E 76)"""
	return cdist_euclidean(lab1, lab2)


def pdist_lab_deltaE94(lab):
	"""Pairwise CIELAB Delta E 94 distances"""
	n = len(lab)
	distances = []
	for i in range(n):
		for j in range(i + 1, n):
			L1, a1, b1 = lab[i]
			L2, a2, b2 = lab[j]

			dL = L1 - L2
			da = a1 - a2
			db = b1 - b2

			C1 = np.sqrt(a1**2 + b1**2)
			C2 = np.sqrt(a2**2 + b2**2)
			dC = C1 - C2
			dH = np.sqrt(da**2 + db**2 - dC**2)

			SL = 1
			SC = 1 + 0.045 * C1
			SH = 1 + 0.015 * C1

			deltaE = np.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2)
			distances.append(deltaE)
	return np.array(distances)


def cdist_lab_deltaE94(lab1: ArrayLike, lab2: ArrayLike) -> np.ndarray:
	"""Cross CIELAB Delta E 94 distances"""
	L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
	L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

	dL = L1[:, np.newaxis] - L2[np.newaxis, :]
	da = a1[:, np.newaxis] - a2[np.newaxis, :]
	db = b1[:, np.newaxis] - b2[np.newaxis, :]

	C1 = np.sqrt(a1**2 + b1**2)
	C2 = np.sqrt(a2**2 + b2**2)
	dC = C1[:, np.newaxis] - C2[np.newaxis, :]
	dH = np.sqrt(da**2 + db**2 - dC**2)

	SL = 1
	SC = 1 + 0.045 * C1[:, np.newaxis]
	SH = 1 + 0.015 * C1[:, np.newaxis]

	return np.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2)


def pdist_lab_deltaE2000(lab):
	"""Pairwise CIELAB Delta E 2000 distances"""
	n = len(lab)
	distances = []
	for i in range(n):
		for j in range(i + 1, n):
			L1, a1, b1 = lab[i]
			L2, a2, b2 = lab[j]

			C1 = np.sqrt(a1**2 + b1**2)
			C2 = np.sqrt(a2**2 + b2**2)
			Cavg = (C1 + C2) / 2

			G = 0.5 * (1 - np.sqrt(Cavg**7 / (Cavg**7 + 25**7)))
			ap1 = (1 + G) * a1
			ap2 = (1 + G) * a2

			Cp1 = np.sqrt(ap1**2 + b1**2)
			Cp2 = np.sqrt(ap2**2 + b2**2)

			hp1 = np.arctan2(b1, ap1) * 180 / np.pi % 360
			hp2 = np.arctan2(b2, ap2) * 180 / np.pi % 360

			dL = L2 - L1
			dC = Cp2 - Cp1
			dhp = hp2 - hp1
			if dhp > 180:
				dhp -= 360
			elif dhp < -180:
				dhp += 360
			dH = 2 * np.sqrt(Cp1 * Cp2) * np.sin(np.radians(dhp) / 2)

			Lavg = (L1 + L2) / 2
			Cavg = (Cp1 + Cp2) / 2
			Havg = (hp1 + hp2) / 2
			if abs(hp1 - hp2) > 180:
				Havg = (Havg + 180) % 360

			T = (
				1
				- 0.17 * np.cos(np.radians(Havg - 30))
				+ 0.24 * np.cos(np.radians(2 * Havg))
				+ 0.32 * np.cos(np.radians(3 * Havg + 6))
				- 0.20 * np.cos(np.radians(4 * Havg - 63))
			)

			dtheta = 30 * np.exp(-(((Havg - 275) / 25) ** 2))
			RC = 2 * np.sqrt(Cavg**7 / (Cavg**7 + 25**7))
			SL = 1 + (0.015 * (Lavg - 50) ** 2) / np.sqrt(20 + (Lavg - 50) ** 2)
			SC = 1 + 0.045 * Cavg
			SH = 1 + 0.015 * Cavg * T
			RT = -np.sin(2 * np.radians(dtheta)) * RC

			deltaE = np.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2 + RT * (dC / SC) * (dH / SH))
			distances.append(deltaE)
	return np.array(distances)


def cdist_lab_deltaE2000(lab1, lab2):
	"""Cross CIELAB Delta E 2000 distances"""
	m, n = len(lab1), len(lab2)
	distances = np.zeros((m, n))

	for i in range(m):
		for j in range(n):
			L1, a1, b1 = lab1[i]
			L2, a2, b2 = lab2[j]

			C1 = np.sqrt(a1**2 + b1**2)
			C2 = np.sqrt(a2**2 + b2**2)
			Cavg = (C1 + C2) / 2

			G = 0.5 * (1 - np.sqrt(Cavg**7 / (Cavg**7 + 25**7)))
			ap1 = (1 + G) * a1
			ap2 = (1 + G) * a2

			Cp1 = np.sqrt(ap1**2 + b1**2)
			Cp2 = np.sqrt(ap2**2 + b2**2)

			hp1 = np.arctan2(b1, ap1) * 180 / np.pi % 360
			hp2 = np.arctan2(b2, ap2) * 180 / np.pi % 360

			dL = L2 - L1
			dC = Cp2 - Cp1
			dhp = hp2 - hp1
			if dhp > 180:
				dhp -= 360
			elif dhp < -180:
				dhp += 360
			dH = 2 * np.sqrt(Cp1 * Cp2) * np.sin(np.radians(dhp) / 2)

			Lavg = (L1 + L2) / 2
			Cavg = (Cp1 + Cp2) / 2
			Havg = (hp1 + hp2) / 2
			if abs(hp1 - hp2) > 180:
				Havg = (Havg + 180) % 360

			T = (
				1
				- 0.17 * np.cos(np.radians(Havg - 30))
				+ 0.24 * np.cos(np.radians(2 * Havg))
				+ 0.32 * np.cos(np.radians(3 * Havg + 6))
				- 0.20 * np.cos(np.radians(4 * Havg - 63))
			)

			dtheta = 30 * np.exp(-(((Havg - 275) / 25) ** 2))
			RC = 2 * np.sqrt(Cavg**7 / (Cavg**7 + 25**7))
			SL = 1 + (0.015 * (Lavg - 50) ** 2) / np.sqrt(20 + (Lavg - 50) ** 2)
			SC = 1 + 0.045 * Cavg
			SH = 1 + 0.015 * Cavg * T
			RT = -np.sin(2 * np.radians(dtheta)) * RC

			distances[i, j] = np.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2 + RT * (dC / SC) * (dH / SH))

	return distances


def pdist_hsv(hsv: ArrayLike):
	"""Pairwise HSV euclidean distances (with circular hue handling)"""
	hsv = np.atleast_2d(hsv)
	n = len(hsv)
	distances = []
	for i in range(n):
		for j in range(i + 1, n):
			h1, s1, v1 = hsv[i]
			h2, s2, v2 = hsv[j]

			dh = min(abs(h1 - h2), 360 - abs(h1 - h2))
			ds = s1 - s2
			dv = v1 - v2

			dist = np.sqrt(dh**2 + ds**2 + dv**2)
			distances.append(dist)
	return np.array(distances)


def cdist_hsv(hsv1: ArrayLike, hsv2: ArrayLike):
	"""Cross HSV euclidean distances (with circular hue handling)."""
	hsv1, hsv2 = np.atleast_2d(hsv1) % 360, np.atleast_2d(hsv2) % 360
	h1, s1, v1 = hsv1[:, 0], hsv1[:, 1], hsv1[:, 2]
	h2, s2, v2 = hsv2[:, 0], hsv2[:, 1], hsv2[:, 2]

	dh_raw = np.abs(h1[:, np.newaxis] - h2[np.newaxis, :])
	dh = np.minimum(dh_raw, 360 - dh_raw)
	ds = s1[:, np.newaxis] - s2[np.newaxis, :]
	dv = v1[:, np.newaxis] - v2[np.newaxis, :]

	return np.sqrt(dh**2 + ds**2 + dv**2)
