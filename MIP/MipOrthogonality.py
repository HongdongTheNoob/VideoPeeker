# This code explores the orthorgonality between different patterns indicated by each reference sample within the same set of matrix data.

import numpy as np
import MipData

matrices = np.array(MipData.data8x8)

np.set_printoptions(precision=3)
interaction_matrix = np.ones([matrices.shape[2], matrices.shape[2]])
for matrixIdx in range(matrices.shape[0]):
    matrix = matrices[matrixIdx, :, :]
    for i in range(matrices.shape[2]):
        mi = matrix[:, i]
        mi = (mi - mi.mean()) / mi.std()
        for j in range(i + 1, matrices.shape[2]):
            mj = matrix[:, j]
            mj = (mj - mj.mean()) / mj.std()
            interaction_matrix[i][j] = mi.dot(mj) / mi.size
            interaction_matrix[j][i] = interaction_matrix[i][j]
    print(interaction_matrix)
    print("\n")
