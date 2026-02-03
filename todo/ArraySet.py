import numpy as np
from numpy.typing import ArrayLike, NDArray
from hirola import HashTable
from collections.abc import MutableSet
from typing import Optional, Iterator, Union
from functools import singledispatchmethod

IntArray = NDArray[np.int64]
IntLike = Union[int, ArrayLike]


# %%
def make_hash_table(arr: ArrayLike | None, size: int = 0):
	arr = [] if arr is None else np.asarray(arr, dtype=np.int64).ravel()
	capacity = max(int(len(arr) * 1.35), 8, size)
	ht = HashTable(capacity, dtype=np.int64)
	ht.add(arr)
	return ht


class ArraySet(MutableSet[int]):
	def __init__(self, init: Optional[ArrayLike] = None, size: int = 0) -> None:
		self.S = make_hash_table(init)

	def contains(self, other: ArrayLike) -> np.ndarray:
		return self.S.contains(np.asarray(other, dtype=np.int64))

	def __contains__(self, x: int) -> bool:
		arr = np.asarray([x], dtype=np.int64)
		return self.S.contains(arr)[0]

	def __len__(self) -> int:
		return self.S.length

	def __iter__(self) -> Iterator[int]:
		return iter(self.S.keys)

	def add(self, x: int) -> None:
		arr = np.asarray([x], dtype=np.int64)
		self.S.add(arr)

	def discard(self, x: int) -> None:
		# arr = np.asarray([x], dtype=np.int64)
		keys_keep = self.S.keys[self.S.keys != x].copy()
		self.S = HashTable(int(len(keys_keep) * 1.35), dtype=np.int64)
		self.S.add(keys_keep)

	def __ior__(self, other: ArrayLike) -> "ArraySet":
		arr = np.asarray(other, dtype=np.int64).ravel()
		arr_disjoint = arr[~self.S.contains(arr)]
		self.S.resize(new_size=int(len(self.S) + len(arr_disjoint) * 1.35), in_place=True)
		self.S.add(arr_disjoint)
		return self

	def __iand__(self, other: ArrayLike) -> "ArraySet":
		arr = np.asarray(other, dtype=np.int64).ravel()
		keys_keep = arr[self.S.contains(arr)]
		self.S = make_hash_table(keys_keep, size=int(len(keys_keep) * 1.35))
		return self

	# @singledispatchmethod
	# def __isub__(self, other) -> "ArraySet":
	# 	pass

	# @__isub__.register
	# def __isub__(self, other: ArrayLike) -> "ArraySet":
	# 	arr = np.asarray(other, dtype=np.int64).ravel()
	# 	R = make_hash_table(arr)
	# 	self.S = make_hash_table(self.keys[~R.contains(self.keys)])
	# 	return self

	# @__isub__.register
	def __isub__(self, other: "ArraySet") -> "ArraySet":
		self.S = make_hash_table(self.S.keys[~other.contains(self.S.keys)])
		return self

	def __or__(self, other: ArrayLike) -> "ArraySet":
		result = ArraySet(self.S.keys, size=len(self.S) + len(other))
		result |= other
		return result

	def __and__(self, other: ArrayLike) -> "ArraySet":
		result = ArraySet(self.S.keys, size=max(len(self.S), len(other)))
		result &= other
		return result

	def __sub__(self, other: ArrayLike) -> "ArraySet":
		result = ArraySet(self.S.keys)
		result -= other
		return result

	def __array__(self, dtype=np.int64) -> NDArray[np.int64]:
		return np.asarray(self.S.keys, dtype=dtype)

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}({self.S.keys})"


# %%
A = HashTable(8, dtype=np.int64)
A.add([1, 2, 3, 4, 5])
# K = A.keys.copy()
R = HashTable(3, dtype=np.int64)


A - ArraySet([0, 1, 2])

R.add(np.array([2, 3]))

B = HashTable()
make_hash_table(A.keys[~R.contains(A.keys)])


HashTable
A.contains(R)


a = ArraySet([1, 2, 3])
b = ArraySet([3, 4, 5, 6, 7])
a | b
a - b
a -= b
a
a |= b
a
a & b
a &= b
a


assert sorted(a | b) == [1, 2, 3, 4, 5]
assert sorted(a & b) == [3]
assert sorted(a - b) == [1, 2]
assert sorted(b - a) == [4, 5]

# %% Benchmarks
import timeit

rng = np.random.default_rng(1234)

A = rng.choice(range(350), size=500)
B = rng.choice(range(50), size=500)

timeit.timeit(lambda: ArraySet(A) | ArraySet(B), number=1500)
timeit.timeit(lambda: set(A) | set(B), number=1500)

timeit.timeit(lambda: ArraySet(A) & ArraySet(B), number=1500)
timeit.timeit(lambda: set(A) & set(B), number=1500)

timeit.timeit(lambda: ArraySet(A) - ArraySet(B), number=1500)
timeit.timeit(lambda: set(A) & set(B), number=1500)

AS = ArraySet(A)
AB = set(A)
timeit.timeit(lambda: AS.__ior__(ArraySet(B)), number=1500)
timeit.timeit(lambda: AB.__ior__(set(B)), number=1500)

AS = ArraySet(A)
AB = set(A)
timeit.timeit(lambda: AS.__iand__(ArraySet(B)), number=1500)
timeit.timeit(lambda: AB.__iand__(set(B)), number=1500)

timeit.timeit(lambda: ArraySet(A) - ArraySet(B), number=1500)
timeit.timeit(lambda: set(A) & set(B), number=1500)

U1 = ArraySet(A) | ArraySet(B)
U2 = set(A) | set(B)

len(U1), len(U2)
