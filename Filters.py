import os
import numpy as np
from scipy.signal import convolve2d
import GetBlock

# apply filter using video as context
def apply_filter(video, frame_number, dimensions, colour_component, block, kernel, use_bottom_right = False):
  template_lines = max(kernel.shape[0] // 2, kernel.shape[1] // 2)

  if colour_component != 'y':
    template_lines *= 2
  
  get_block_dimensions = list(dimensions)

  # extend top and left areas
  get_block_dimensions[0] = max(dimensions[0] - template_lines, 0)
  get_block_dimensions[1] = max(dimensions[1] - template_lines, 0)
  get_block_dimensions[2] += dimensions[0] - get_block_dimensions[0]
  get_block_dimensions[3] += dimensions[1] - get_block_dimensions[1]

  # extend right and bottom areas
  if use_bottom_right:
    get_block_dimensions[2] += template_lines
    get_block_dimensions[3] += template_lines

  context_area, _ = GetBlock.get_block(video, frame_number, tuple(get_block_dimensions), colour_component, 0)

  # fill block into context
  if colour_component != 'y':
    template_lines = template_lines // 2
  x_start = min(template_lines, dimensions[0] if colour_component == 'y' else dimensions[0] // 2)
  y_start = min(template_lines, dimensions[1] if colour_component == 'y' else dimensions[1] // 2)
  for row in range(y_start, y_start + block.shape[0]):
    context_area[row][x_start:x_start + block.shape[1]] = block[row - y_start][:]

  # fill up bottom/right
  if context_area.shape[1] < block.shape[1] + 2 * template_lines:
    right_most_column = context_area[:, -1].reshape((-1, 1))
    context_area = np.column_stack((context_area, np.tile(right_most_column, (1, block.shape[1] + 2 * template_lines - context_area.shape[1]))))
  if context_area.shape[0] < block.shape[0] + 2 * template_lines:
    bottom_most_row = context_area[-1, :].reshape((1, -1))
    context_area = np.row_stack((context_area, np.tile(bottom_most_row, (block.shape[0] + 2 * template_lines - context_area.shape[0], 1))))

  # filter
  filtered_block = convolve2d(context_area, kernel, mode = 'valid')

  return filtered_block