import sys
sys.path.append('..')
import numpy as np
import cv2

import GetBlock

def pad_left(image):
  new_column = image[:, 0].reshape(-1, 1)
  return np.hstack((new_column, image))

def pad_top(image):
  new_row = image[0, :]
  return np.vstack((new_row, image))

# Input: video struct, frame number, dimensions as (x, y, w, h), number of lines in template (usually 6)
# Output: predicted blocks, coefficients, SADs
def simulate_eip(video, frame_number, dimensions, template_lines):
  x, y, w, h = dimensions
  luma_block, luma_template = GetBlock.get_block(video, frame_number, dimensions, 'y', template_lines)
  
  template_left_width = min(template_lines, x)
  template_top_height = min(template_lines, y)

  block_mask = np.ones_like(luma_template, dtype = bool)
  block_mask[0:template_top_height, :] = False
  block_mask[:, 0:template_left_width] = False

  template_mask = np.ones_like(luma_template, dtype = bool)
  template_mask[block_mask] = False
  template_mask[0:3, :] = False
  template_mask[:, 0:3] = False

  # offsets
  offset_y = [0,-1,0,-1,-2,0,-1,-2,-3,-1,-2,-3,-2,-3,-3]
  offset_x = [-1,0,-2,-1,0,-3,-2,-1,0,-3,-2,-1,-3,-2,-3]

  # sampling
  template_all_y, template_all_x = np.where(template_mask)
  samples = np.zeros((len(template_all_y), 0)).astype(video.data_type)
  for s in range(len(offset_y)):
    samples = np.hstack((samples, luma_template[template_all_y + offset_y[s], template_all_x + offset_x[s]].reshape(-1, 1)))
  samples_y = luma_template[template_all_y, template_all_x]

  # regression
  coeffs, _, _, _ = np.linalg.lstsq(samples, samples_y, rcond = None)

  # prediction
  for r in range(template_top_height, template_top_height + h):
    for c in range(template_left_width, template_left_width + w):
      neighbours = []
      for s in range(len(offset_y)):
        neighbour_y = max(r + offset_y[s], 0)
        neighbour_x = max(c + offset_x[s], 0)
        neighbours.append(luma_template[neighbour_y, neighbour_x])
      luma_template[r, c] = np.dot(neighbours, coeffs).astype(luma_template.dtype)

  predicted_block = luma_template[template_top_height:template_top_height+h, template_left_width:template_left_width+w]

  sad = np.sum(np.abs(predicted_block - luma_block))

  return predicted_block, coeffs, sad