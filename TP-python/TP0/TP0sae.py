# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:57:18 2025

@author: dougl
"""

import numpy as np
# %% Question 1
print("\nQuestion 1 :\n")

A = np.array([[1, 2, 3], [4, 5, 6]])
print("A =\n", A)
B = np.zeros((2, 3))
print("B =\n", B)
C = np.ones((3, 2))
print("C =\n", C)
D = np.eye(4)
print("D =\n ", D)

# %% Question 2
print("\nQuestion 2 :\n")

M1 = np.array([5, 6, 7, 8, 9])
print("M1 =\n", M1)
M2 = np.ones(5)
print("M2 =\n", M2)
M3 = np.array([[1], [2], [3]])
print("M3 =\n", M3)
M4 = np.array([
    [1, 2, 4],
    [-1, 2, 3],
    [1, 8, 9]
])
print("M4 =\n", M4)

# %% Question 3
print("\nQuestion 3 :\n")
L3 = M3.tolist()

print("L3 = ", L3)

print("Longueur de L3 = ", len(L3))
print("Longueur de M3 = ", len(M3))
print("Méthode shape de L3 = ", np.shape(L3))
print("Méthode shape de M3= ", np.shape(M3))

# %% Question 4

print("\nQuestion 4 :\n")
print(np.concatenate((A, B), axis=0))
# concatène verticalement

print(np.concatenate((A, B), axis=1))
# concatène horizontalement

# np.concatenate((B, C), axis=0)
# erreur formes incompatibles

print("M4[1, 0] = ", M4[1, 0], "\n")

print("M4[0, :] = ", M4[0, :], "\n")

print("M4[0:1, :] = ", M4[0:1, :], "\n")

print("M4[:, 1] = ", M4[:, 1], "\n")

print("M4[1:3, 0:2] =", M4[1:3, 0:2], "\n")


# %% Question 5


np.shape(M4[0, :])      # (3,)
np.shape(M4[0:1, :])    # (1, 3)
np.shape(A[:, 1])       # (2,)
np.shape(A[:, 1:2])     # (2, 1)

# M4[0, :] aplatit la ligne, M4[0:1, :] la garde en 2D

# %% Question 6

print("\nQuestion 6 :\n")

M5 = np.concatenate((M2, M1))
print("M5 =\n", M5, "\n")

M6 = np.vstack((M1, M2))
print("M6 =\n", M6, "\n")

M7 = M4[1:3, 0:2]
print("M7 =\n", M7, "\n")

M8 = M4[[2, 0, 1], :]
print("M8 =\n", M8, "\n")

# %% Question 7

print("\nQuestion 7 :\n")

M9 = M1.copy()
M9[[0, 2]] = 8
print("M9 =\n", M9, "\n")

M10 = np.insert(M2, 0, 2)
print("M10 =\n", M10, "\n")

M11 = np.vstack(([9], M3, [8]))
print("M11 =\n", M11, "\n")

M12 = np.delete(M4, 1, axis=0)
print("M12 =\n", M12, "\n")

M13 = np.delete(M4, 1, axis=1)
print("M13 =\n", M13, "\n")

# %% Question 9

A = np.array([[-1, 2], [1, 8]])
B = np.array([[1, 2], [3, 4]])
print("produit de A par b =\n", np.dot(A, B), "\n")


C = np.array([1, 2, 3])
D = np.array([
    [1, 8],
    [1, 2],
    [-1, 2]
])
print("produit de C par D =\n", np.dot(C, D), "\n")

# %% Question 10

v = np.arange(1, 11)
w = np.arange(1, 11)
v.shape = (10, 1)
w.shape = (1, 10)
M14 = np.dot(v, w)
print("M14 =\n", M14, "\n")

# %% Question 11
