import numpy as np
import GetBlock
from scipy.signal import convolve2d

luma_filter_4x4 = np.array([
  [  0, 0,   0, 64 * 4,  0,   0,  0],
  [  0, 1 * 4,  -3 * 4, 63 * 4,  4 * 4,  -2 * 4,  1 * 4],
  [  0, 1 * 4,  -5 * 4, 62 * 4,  8 * 4,  -3 * 4,  1 * 4],
  [  0, 2 * 4,  -8 * 4, 60 * 4, 13 * 4,  -4 * 4,  1 * 4],
  [  0, 3 * 4, -10 * 4, 58 * 4, 17 * 4,  -5 * 4,  1 * 4], #1/4
  [  0, 3 * 4, -11 * 4, 52 * 4, 26 * 4,  -8 * 4,  2 * 4],
  [  0, 2 * 4,  -9 * 4, 47 * 4, 31 * 4, -10 * 4,  3 * 4],
  [  0, 3 * 4, -11 * 4, 45 * 4, 34 * 4, -10 * 4,  3 * 4],
  [  0, 3 * 4, -11 * 4, 40 * 4, 40 * 4, -11 * 4,  3 * 4], #1/2
  [  0, 3 * 4, -10 * 4, 34 * 4, 45 * 4, -11 * 4,  3 * 4],
  [  0, 3 * 4, -10 * 4, 31 * 4, 47 * 4,  -9 * 4,  2 * 4],
  [  0, 2 * 4,  -8 * 4, 26 * 4, 52 * 4, -11 * 4,  3 * 4],
  [  0, 1 * 4,  -5 * 4, 17 * 4, 58 * 4, -10 * 4,  3 * 4], #3/4}
  [  0, 1 * 4,  -4 * 4, 13 * 4, 60 * 4,  -8 * 4,  2 * 4],
  [  0, 1 * 4,  -3 * 4,  8 * 4, 62 * 4,  -5 * 4,  1 * 4],
  [  0, 1 * 4,  -2 * 4,  4 * 4, 63 * 4,  -3 * 4,  1 * 4]
])

luma_filter_12 = np.array([
    [ 0,     0,     0,     0,     0,   256,     0,     0,     0,     0,     0,     0, ], # 0
    [-1,     2,    -3,     6,   -14,   254,    16,    -7,     4,    -2,     1,     0, ],
    [-1,     3,    -7,    12,   -26,   249,    35,   -15,     8,    -4,     2,     0, ],
    [-2,     5,    -9,    17,   -36,   241,    54,   -22,    12,    -6,     3,    -1, ],
    [-2,     5,   -11,    21,   -43,   230,    75,   -29,    15,    -8,     4,    -1, ], # 4
    [-2,     6,   -13,    24,   -48,   216,    97,   -36,    19,   -10,     4,    -1, ],
    [-2,     7,   -14,    25,   -51,   200,   119,   -42,    22,   -12,     5,    -1, ],
    [-2,     7,   -14,    26,   -51,   181,   140,   -46,    24,   -13,     6,    -2, ],
    [-2,     6,   -13,    25,   -50,   162,   162,   -50,    25,   -13,     6,    -2, ], # 8
    [-2,     6,   -13,    24,   -46,   140,   181,   -51,    26,   -14,     7,    -2, ],
    [-1,     5,   -12,    22,   -42,   119,   200,   -51,    25,   -14,     7,    -2, ],
    [-1,     4,   -10,    19,   -36,    97,   216,   -48,    24,   -13,     6,    -2, ],
    [-1,     4,    -8,    15,   -29,    75,   230,   -43,    21,   -11,     5,    -2, ], # 12
    [-1,     3,    -6,    12,   -22,    54,   241,   -36,    17,    -9,     5,    -2, ],
    [ 0,     2,    -4,     8,   -15,    35,   249,   -26,    12,    -7,     3,    -1, ],
    [ 0,     1,    -2,     4,    -7,    16,   254,   -14,     6,    -3,     2,    -1, ],
])


chroma_filter_6 = [
    [0, 0, 256, 0, 0, 0],
    [1, -6, 256, 7, -2, 0],
    [2, -11, 253, 15, -4, 1],
    [3, -16, 251, 23, -6, 1],
    [4, -21, 248, 33, -10, 2],
    [5, -25, 244, 42, -12, 2],
    [7, -30, 239, 53, -17, 4],
    [7, -32, 234, 62, -19, 4],
    [8, -35, 227, 73, -22, 5],
    [9, -38, 220, 84, -26, 7],
    [10, -40, 213, 95, -29, 7],
    [10, -41, 204, 106, -31, 8],
    [10, -42, 196, 117, -34, 9],
    [10, -41, 187, 127, -35, 8],
    [11, -42, 177, 138, -38, 10],
    [10, -41, 168, 148, -39, 10],
    [10, -40, 158, 158, -40, 10],
    [10, -39, 148, 168, -41, 10],
    [10, -38, 138, 177, -42, 11],
    [8, -35, 127, 187, -41, 10],
    [9, -34, 117, 196, -42, 10],
    [8, -31, 106, 204, -41, 10],
    [7, -29, 95, 213, -40, 10],
    [7, -26, 84, 220, -38, 9],
    [5, -22, 73, 227, -35, 8],
    [4, -19, 62, 234, -32, 7],
    [4, -17, 53, 239, -30, 7],
    [2, -12, 42, 244, -25, 5],
    [2, -10, 33, 248, -21, 4],
    [1, -6, 23, 251, -16, 3],
    [1, -4, 15, 253, -11, 2],
    [0, -2, 7, 256, -6, 1],
]

chroma_filter_4 = [
  [  0 * 4, 64 * 4,  0 * 4,  0 * 4 ],
  [ -1 * 4, 63 * 4,  2 * 4,  0 * 4 ],
  [ -2 * 4, 62 * 4,  4 * 4,  0 * 4 ],
  [ -2 * 4, 60 * 4,  7 * 4, -1 * 4 ],
  [ -2 * 4, 58 * 4, 10 * 4, -2 * 4 ],
  [ -3 * 4, 57 * 4, 12 * 4, -2 * 4 ],
  [ -4 * 4, 56 * 4, 14 * 4, -2 * 4 ],
  [ -4 * 4, 55 * 4, 15 * 4, -2 * 4 ],
  [ -4 * 4, 54 * 4, 16 * 4, -2 * 4 ],
  [ -5 * 4, 53 * 4, 18 * 4, -2 * 4 ],
  [ -6 * 4, 52 * 4, 20 * 4, -2 * 4 ],
  [ -6 * 4, 49 * 4, 24 * 4, -3 * 4 ],
  [ -6 * 4, 46 * 4, 28 * 4, -4 * 4 ],
  [ -5 * 4, 44 * 4, 29 * 4, -4 * 4 ],
  [ -4 * 4, 42 * 4, 30 * 4, -4 * 4 ],
  [ -4 * 4, 39 * 4, 33 * 4, -4 * 4 ],
  [ -4 * 4, 36 * 4, 36 * 4, -4 * 4 ],
  [ -4 * 4, 33 * 4, 39 * 4, -4 * 4 ],
  [ -4 * 4, 30 * 4, 42 * 4, -4 * 4 ],
  [ -4 * 4, 29 * 4, 44 * 4, -5 * 4 ],
  [ -4 * 4, 28 * 4, 46 * 4, -6 * 4 ],
  [ -3 * 4, 24 * 4, 49 * 4, -6 * 4 ],
  [ -2 * 4, 20 * 4, 52 * 4, -6 * 4 ],
  [ -2 * 4, 18 * 4, 53 * 4, -5 * 4 ],
  [ -2 * 4, 16 * 4, 54 * 4, -4 * 4 ],
  [ -2 * 4, 15 * 4, 55 * 4, -4 * 4 ],
  [ -2 * 4, 14 * 4, 56 * 4, -4 * 4 ],
  [ -2 * 4, 12 * 4, 57 * 4, -3 * 4 ],
  [ -2 * 4, 10 * 4, 58 * 4, -2 * 4 ],
  [ -1 * 4,  7 * 4, 60 * 4, -2 * 4 ],
  [  0 * 4,  4 * 4, 62 * 4, -2 * 4 ],
  [  0 * 4,  2 * 4, 63 * 4, -1 * 4 ],
]

# directly input dimensions (x, y, w, h) with decimal numbers
# decimal position will be rounded down to 1/16 pixel accuracy
def get_block_subpixel(video, frame_number, dimensions, colour_component, kernel):
  expanded_dimensions = [int(x) for x in list(dimensions)]
    

  expanded_lines = (kernel.shape[1] - 1) // 2
  expanded_dimensions[0] -= expanded_lines if colour_component == 'y' else expanded_lines * 2
  expanded_dimensions[1] -= expanded_lines if colour_component == 'y' else expanded_lines * 2
  expanded_dimensions[2] += (kernel.shape[1] - 1) if colour_component == 'y' else (kernel.shape[1] - 1) * 2
  expanded_dimensions[3] += (kernel.shape[1] - 1) if colour_component == 'y' else (kernel.shape[1] - 1) * 2
  expanded_dimensions = tuple(expanded_dimensions)

  expanded_block, _ = GetBlock.get_block(video, frame_number, expanded_dimensions, colour_component, 0)

  # boundary check and padding
  if int(dimensions[0]) < expanded_lines: # left
    left_most_column = expanded_block[:, 0].reshape((-1, 1))
    expanded_block = np.column_stack((np.tile(left_most_column, (1, expanded_lines - int(dimensions[0]))), expanded_block))
    print(expanded_block.shape)
  if expanded_block.shape[1] < dimensions[2] + (kernel.shape[1] - 1): #right
    right_most_column = expanded_block[:, -1].reshape((-1, 1))
    expanded_block = np.column_stack((expanded_block, np.tile(right_most_column, (1, dimensions[2] + (kernel.shape[1] - 1) - expanded_block.shape[1]))))
    print(expanded_block.shape)
  if int(dimensions[1]) < expanded_lines: #top
    top_most_row = expanded_block[0, :].reshape((1, -1))
    expanded_block = np.row_stack((np.tile(top_most_row, (expanded_lines - int(dimensions[1]), 1)), expanded_block))
    print(expanded_block.shape)
  if expanded_block.shape[0] < dimensions[3] + (kernel.shape[1] - 1): #bottom
    bottom_most_row = expanded_block[-1, :].reshape((1, -1))
    expanded_block = np.row_stack((expanded_block, np.tile(bottom_most_row, (dimensions[3] + (kernel.shape[1] - 1) - expanded_block.shape[0], 1))))
    print(expanded_block.shape)

  if colour_component == 'y':
    y_fractional = int(dimensions[1] * 16) % 16
    x_fractional = int(dimensions[0] * 16) % 16
  else:
    y_fractional = int(dimensions[1] * 16) % 32
    x_fractional = int(dimensions[0] * 16) % 32

  horizontal_filtered_block = (convolve2d(expanded_block, np.array(kernel[x_fractional])[::-1].reshape((1, -1)), mode = 'valid') / 256).astype(expanded_block.dtype)
  filtered_block = (convolve2d(horizontal_filtered_block, np.array(kernel[y_fractional])[::-1].reshape((-1, 1)), mode = 'valid') / 256).astype(expanded_block.dtype)

  return filtered_block

