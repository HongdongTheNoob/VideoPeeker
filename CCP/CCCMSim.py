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
def simulate_cccm(video, frame_number, dimensions, template_lines_in_chroma):
  x, y, w, h = dimensions
  sample_dimensions = (x, y, w + 2, h + 2)
  template_lines_in_luma = (template_lines_in_chroma + 1) * 2
  luma_block, luma_template = GetBlock.get_downsampled_block(video, frame_number, sample_dimensions, template_lines_in_luma)
  cb_block, cb_template = GetBlock.get_block(video, frame_number, sample_dimensions, 'cb', template_lines_in_luma)
  cr_block, cr_template = GetBlock.get_block(video, frame_number, sample_dimensions, 'cr', template_lines_in_luma)
  luma_block = luma_block[:-1, :-1]
  cb_block = cb_block[:-1, :-1]
  cr_block = cr_block[:-1, :-1]
                            
  template_left_width = min(template_lines_in_chroma + 1, x // 2)
  template_top_height = min(template_lines_in_chroma + 1, y // 2)

  if template_left_width == 0:
    luma_template = pad_left(luma_template)
    cb_template = pad_left(cb_template)
    cr_template = pad_left(cr_template)
    template_left_width = 1
  if template_top_height == 0:
    luma_template = pad_top(luma_template)
    cb_template = pad_top(cb_template)
    cr_template = pad_top(cr_template)
    template_top_height = 1

  block_mask = np.ones_like(luma_template, dtype = bool)
  block_mask[0:template_top_height, :] = False
  block_mask[:, 0:template_left_width] = False
  block_mask[-1, :] = False
  block_mask[:, -1] = False

  template_mask = np.ones_like(luma_template, dtype = bool)
  template_mask[[0, -1], :] = False
  template_mask[:, [0, -1]] = False
  template_mask[block_mask] = False

  # sampling
  template_all_y, template_all_x = np.where(template_mask)
  C = luma_template[template_all_y, template_all_x].reshape(-1, 1)
  N = luma_template[template_all_y-1, template_all_x].reshape(-1, 1)
  S = luma_template[template_all_y+1, template_all_x].reshape(-1, 1)
  W = luma_template[template_all_y, template_all_x-1].reshape(-1, 1)
  E = luma_template[template_all_y, template_all_x+1].reshape(-1, 1)
  CC = np.square(C)
  B = np.ones_like(C, dtype = C.dtype) * (2 ** (video.bit_depth - 1))

  samples = np.hstack((C, N, S, W, E, CC, B))
  samples_cb = cb_template[template_all_y, template_all_x].reshape(-1, 1)
  samples_cr = cr_template[template_all_y, template_all_x].reshape(-1, 1)

  # regression
  x_cb, _, _, _ = np.linalg.lstsq(samples, samples_cb, rcond = None)
  x_cr, _, _, _ = np.linalg.lstsq(samples, samples_cr, rcond = None)

  # prediction
  luma_block_with_template = luma_template.copy()
  luma_block_with_template[template_top_height:-1, template_left_width:-1] = luma_block
  block_all_y, block_all_x = np.where(block_mask)
  C = luma_block_with_template[block_all_y, block_all_x].reshape(-1, 1)
  N = luma_block_with_template[block_all_y-1, block_all_x].reshape(-1, 1)
  S = luma_block_with_template[block_all_y+1, block_all_x].reshape(-1, 1)
  W = luma_block_with_template[block_all_y, block_all_x-1].reshape(-1, 1)
  E = luma_block_with_template[block_all_y, block_all_x+1].reshape(-1, 1)
  CC = np.square(C)
  B = np.ones_like(C, dtype = C.dtype) * (2 ** (video.bit_depth - 1))

  samples = np.hstack((C, N, S, W, E, CC, B))
  predicted_cb = np.dot(samples, x_cb).astype(video.data_type)
  predicted_cr = np.dot(samples, x_cr).astype(video.data_type)
  cb_template[block_mask] = predicted_cb.flatten()
  cr_template[block_mask] = predicted_cr.flatten()

  predicted_cb_block = cb_template[template_top_height:-1, template_left_width:-1]
  predicted_cr_block = cr_template[template_top_height:-1, template_left_width:-1]

  sad_cb = np.sum(np.abs(predicted_cb_block - cb_block))
  sad_cr = np.sum(np.abs(predicted_cr_block - cr_block))

  return predicted_cb_block, predicted_cr_block, x_cb, x_cr, sad_cb, sad_cr