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

def pad_right(image):
  new_column = image[:, -1].reshape(-1, 1)
  return np.hstack((image, new_column))

def pad_bottom(image):
  new_row = image[-1, :]
  return np.vstack((image, new_row))

def get_cccm_blocks_and_templates(video, frame_number, dimensions, template_lines_in_chroma):
  x, y, w, h = dimensions
  # extend 1 line to the  right and the bottom because of CCCM sampling
  sample_dimensions = (x, y, w + 2, h + 2)
  template_lines_in_luma = (template_lines_in_chroma + 1) * 2
  luma_block, luma_template = GetBlock.get_downsampled_block(video, frame_number, sample_dimensions, template_lines_in_luma)
  cb_block, cb_template = GetBlock.get_block(video, frame_number, sample_dimensions, 'cb', template_lines_in_luma)
  cr_block, cr_template = GetBlock.get_block(video, frame_number, sample_dimensions, 'cr', template_lines_in_luma)
  # check if the block hits the right or the bottom of the image
  if x + w >= video.width:
    # extend template
    luma_template = pad_right(luma_template)
    cb_template = pad_right(cb_template)
    cr_template = pad_right(cr_template)
  else:
    # shrink block
    luma_block = luma_block[:, :-1]
    cb_block = cb_block[:, :-1]
    cr_block = cr_block[:, :-1]
  if y + h >= video.height:
    # extend template
    luma_template = pad_bottom(luma_template)
    cb_template = pad_bottom(cb_template)
    cr_template = pad_bottom(cr_template)
  else:
    # shrink block
    luma_block = luma_block[:-1, :]
    cb_block = cb_block[:-1, :]
    cr_block = cr_block[:-1, :]

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
  
  return (luma_block, luma_template, cb_block, cb_template, cr_block, cr_template, template_left_width, template_top_height, block_mask, template_mask)

# Input: video struct, frame number, dimensions as (x, y, w, h), number of lines in template (usually 6)
# Output: predicted blocks, SADs, coefficients
def simulate_cccm(input_video, frame_number, dimensions, template_lines_in_chroma, reconstructed_video = None, glcccm = 0, l2_regularisation = 0):
  x, y, w, h = dimensions

  blocks_and_templates = get_cccm_blocks_and_templates(input_video, frame_number, dimensions, template_lines_in_chroma)
  luma_block, luma_template, cb_block, cb_template, cr_block, cr_template, template_left_width, template_top_height, block_mask, template_mask = blocks_and_templates

  if reconstructed_video is not None:
    reconstructed_templates = get_cccm_blocks_and_templates(reconstructed_video, frame_number, dimensions, template_lines_in_chroma)
    luma_block, luma_template, _, cb_template, _, cr_template, template_left_width, template_top_height, block_mask, template_mask = reconstructed_templates
    if input_video.bit_depth == 8:
      cb_block = cb_block.astype(reconstructed_video.data_type) * 4
      cr_block = cr_block.astype(reconstructed_video.data_type) * 4
  else:
    reconstructed_video = input_video
    # use 10-bit as operation bit-depth
    if input_video.bit_depth == 8:
      reconstructed_video.bit_depth = 10
      reconstructed_video.data_type = np.uint16
      luma_block = luma_block.astype("uint16") * 4
      luma_template = luma_template.astype("uint16") * 4
      cb_block = cb_block.astype("uint16") * 4
      cb_template = cb_template.astype("uint16") * 4
      cr_block = cr_block.astype("uint16") * 4
      cr_template = cr_template.astype("uint16") * 4

  # sampling
  template_all_y, template_all_x = np.where(template_mask)
  if len(template_all_y) == 0:
    return None, None, 0, 0, [], []
  
  C = luma_template[template_all_y, template_all_x].reshape(-1, 1).astype("float64")
  N = luma_template[template_all_y-1, template_all_x].reshape(-1, 1).astype("float64")
  S = luma_template[template_all_y+1, template_all_x].reshape(-1, 1).astype("float64")
  W = luma_template[template_all_y, template_all_x-1].reshape(-1, 1).astype("float64")
  E = luma_template[template_all_y, template_all_x+1].reshape(-1, 1).astype("float64")
  CC = np.square(C)
  B = np.ones_like(C, dtype = C.dtype) * (2 ** (reconstructed_video.bit_depth - 1))
  if not glcccm:
    samples = np.hstack((C, N, S, W, E, CC, B))
  else:
    NW = luma_template[template_all_y-1, template_all_x-1].reshape(-1, 1).astype("float64")
    NE = luma_template[template_all_y-1, template_all_x+1].reshape(-1, 1).astype("float64")
    SW = luma_template[template_all_y+1, template_all_x-1].reshape(-1, 1).astype("float64")
    SE = luma_template[template_all_y+1, template_all_x+1].reshape(-1, 1).astype("float64")
    GY = (N * 2 + NW + NE - S * 2 - SW - SE)
    GX = (W * 2 + NW + SW - E * 2 - NE - SE)
    X, Y = np.meshgrid(range(luma_template.shape[1]), range(luma_template.shape[0]))
    X = np.reshape(X, luma_template.shape)[template_all_y, template_all_x].reshape(-1, 1).astype("float64")
    Y = np.reshape(Y, luma_template.shape)[template_all_y, template_all_x].reshape(-1, 1).astype("float64")
    samples = np.hstack((C, GY, GX, Y, X, CC, B))

  samples_cb = cb_template[template_all_y, template_all_x].reshape(-1, 1).astype("float64")
  samples_cr = cr_template[template_all_y, template_all_x].reshape(-1, 1).astype("float64")

  # regression
  if l2_regularisation == 0:
    coeffs_cb, _, _, _ = np.linalg.lstsq(samples, samples_cb, rcond = None)
    coeffs_cr, _, _, _ = np.linalg.lstsq(samples, samples_cr, rcond = None)
  else:
    coeffs_cb = np.linalg.inv(samples.T @ samples + l2_regularisation * samples.shape[1] * np.eye(samples.shape[1])) @ samples.T @ samples_cb
    coeffs_cr = np.linalg.inv(samples.T @ samples + l2_regularisation * samples.shape[1] * np.eye(samples.shape[1])) @ samples.T @ samples_cr

  # prediction
  luma_block_with_template = luma_template.copy()
  luma_block_with_template[template_top_height:-1, template_left_width:-1] = luma_block
  block_all_y, block_all_x = np.where(block_mask)
  C = luma_block_with_template[block_all_y, block_all_x].reshape(-1, 1).astype("float64")
  N = luma_block_with_template[block_all_y-1, block_all_x].reshape(-1, 1).astype("float64")
  S = luma_block_with_template[block_all_y+1, block_all_x].reshape(-1, 1).astype("float64")
  W = luma_block_with_template[block_all_y, block_all_x-1].reshape(-1, 1).astype("float64")
  E = luma_block_with_template[block_all_y, block_all_x+1].reshape(-1, 1).astype("float64")
  CC = np.square(C)
  B = np.ones_like(C, dtype = C.dtype) * (2 ** (reconstructed_video.bit_depth - 1))
  if not glcccm:
    samples = np.hstack((C, N, S, W, E, CC, B))
  else:
    NW = luma_template[block_all_y-1, block_all_x-1].reshape(-1, 1).astype("float64")
    NE = luma_template[block_all_y-1, block_all_x+1].reshape(-1, 1).astype("float64")
    SW = luma_template[block_all_y+1, block_all_x-1].reshape(-1, 1).astype("float64")
    SE = luma_template[block_all_y+1, block_all_x+1].reshape(-1, 1).astype("float64")
    GY = (N * 2 + NW + NE - S * 2 - SW - SE)
    GX = (W * 2 + NW + SW - E * 2 - NE - SE)
    X, Y = np.meshgrid(range(luma_template.shape[1]), range(luma_template.shape[0]))
    X = np.reshape(X, luma_template.shape)[block_all_y, block_all_x].reshape(-1, 1).astype("float64")
    Y = np.reshape(Y, luma_template.shape)[block_all_y, block_all_x].reshape(-1, 1).astype("float64")
    samples = np.hstack((C, GY, GX, Y, X, CC, B))

  predicted_cb = np.dot(samples, coeffs_cb).astype(reconstructed_video.data_type)
  predicted_cr = np.dot(samples, coeffs_cr).astype(reconstructed_video.data_type)
  cb_template[block_mask] = predicted_cb.flatten()
  cr_template[block_mask] = predicted_cr.flatten()

  predicted_cb_block = cb_template[template_top_height:-1, template_left_width:-1]
  predicted_cr_block = cr_template[template_top_height:-1, template_left_width:-1]

  sad_cb = np.sum(np.abs(predicted_cb_block.astype("int32") - cb_block))
  sad_cr = np.sum(np.abs(predicted_cr_block.astype("int32") - cr_block))

  if input_video.bit_depth == 8:
    sad_cb = np.sum(np.abs(((predicted_cb_block + 2) // 4).astype("int32") - ((cb_block + 2) // 4)))
    sad_cr = np.sum(np.abs(((predicted_cr_block + 2) // 4).astype("int32") - ((cr_block + 2) // 4)))
    predicted_cb_block = ((predicted_cb_block + 2) // 4).astype("uint8")
    predicted_cr_block = ((predicted_cr_block + 2) // 4).astype("uint8")
  else:
    sad_cb = np.sum(np.abs(predicted_cb_block.astype("int32") - cb_block))
    sad_cr = np.sum(np.abs(predicted_cr_block.astype("int32") - cr_block))

  return predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, (coeffs_cb, coeffs_cr)

# Input: video struct, frame number, dimensions as (x, y, w, h), number of lines in template (usually 6)
# Output: predicted blocks, SADs, coefficients, MADs
# evaluate_on_template: also calculates MAD on template. Basically trying to compare training and validation/test residues.
def simulate_mm_cccm(input_video, frame_number, dimensions, template_lines_in_chroma, reconstructed_video = None, glcccm = 0, l2_regularisation = 0, evaluate_on_template = False):
  x, y, w, h = dimensions

  blocks_and_templates = get_cccm_blocks_and_templates(input_video, frame_number, dimensions, template_lines_in_chroma)
  luma_block, luma_template, cb_block, cb_template, cr_block, cr_template, template_left_width, template_top_height, block_mask, template_mask = blocks_and_templates

  if reconstructed_video is not None:
    reconstructed_templates = get_cccm_blocks_and_templates(reconstructed_video, frame_number, dimensions, template_lines_in_chroma)
    luma_block, luma_template, _, cb_template, _, cr_template, template_left_width, template_top_height, block_mask, template_mask = reconstructed_templates
    if input_video.bit_depth == 8:
      cb_block = cb_block.astype(reconstructed_video.data_type) * 4
      cr_block = cr_block.astype(reconstructed_video.data_type) * 4
  else:
    reconstructed_video = input_video
    # use 10-bit as operation bit-depth
    if input_video.bit_depth == 8:
      reconstructed_video.bit_depth = 10
      reconstructed_video.data_type = np.uint16
      luma_block = luma_block.astype(reconstructed_video.data_type) * 4
      luma_template = luma_template.astype(reconstructed_video.data_type) * 4
      cb_block = cb_block.astype(reconstructed_video.data_type) * 4
      cb_template = cb_template.astype(reconstructed_video.data_type) * 4
      cr_block = cr_block.astype(reconstructed_video.data_type) * 4
      cr_template = cr_template.astype(reconstructed_video.data_type) * 4

  # sampling
  template_all_y, template_all_x = np.where(template_mask)
  if len(template_all_y) == 0:
    return None, None, 0, 0, (), ()

  C = luma_template[template_all_y, template_all_x].reshape(-1, 1).astype("float64")
  N = luma_template[template_all_y-1, template_all_x].reshape(-1, 1).astype("float64")
  S = luma_template[template_all_y+1, template_all_x].reshape(-1, 1).astype("float64")
  W = luma_template[template_all_y, template_all_x-1].reshape(-1, 1).astype("float64")
  E = luma_template[template_all_y, template_all_x+1].reshape(-1, 1).astype("float64")
  CC = np.square(C)
  B = np.ones_like(C, dtype = C.dtype) * (2 ** (reconstructed_video.bit_depth - 1))
  template_luma_average = np.mean(C).astype(reconstructed_video.data_type)
  index_model0 = np.column_stack(np.where(C < template_luma_average))[:, 0]
  index_model1 = np.column_stack(np.where(C >= template_luma_average))[:, 0]

  if not glcccm:
    C0, N0, S0, W0, E0, CC0, B0 = C[index_model0], N[index_model0], S[index_model0], W[index_model0], E[index_model0], CC[index_model0], B[index_model0]
    C1, N1, S1, W1, E1, CC1, B1 = C[index_model1], N[index_model1], S[index_model1], W[index_model1], E[index_model1], CC[index_model1], B[index_model1]
    samples0 = np.hstack((C0, N0, S0, W0, E0, CC0, B0))
    samples1 = np.hstack((C1, N1, S1, W1, E1, CC1, B1))
  else:
    NW = luma_template[template_all_y-1, template_all_x-1].reshape(-1, 1).astype("float64")
    NE = luma_template[template_all_y-1, template_all_x+1].reshape(-1, 1).astype("float64")
    SW = luma_template[template_all_y+1, template_all_x-1].reshape(-1, 1).astype("float64")
    SE = luma_template[template_all_y+1, template_all_x+1].reshape(-1, 1).astype("float64")
    GY = (N * 2 + NW + NE - S * 2 - SW - SE)
    GX = (W * 2 + NW + SW - E * 2 - NE - SE)
    X, Y = np.meshgrid(range(luma_template.shape[1]), range(luma_template.shape[0]))
    X = np.reshape(X, luma_template.shape)[template_all_y, template_all_x].reshape(-1, 1).astype("float64")
    Y = np.reshape(Y, luma_template.shape)[template_all_y, template_all_x].reshape(-1, 1).astype("float64")
    C0, GY0, GX0, Y0, X0, CC0, B0 = C[index_model0], GY[index_model0], GX[index_model0], Y[index_model0], X[index_model0], CC[index_model0], B[index_model0]
    C1, GY1, GX1, Y1, X1, CC1, B1 = C[index_model1], GY[index_model1], GX[index_model1], Y[index_model1], X[index_model1], CC[index_model1], B[index_model1]
    samples0 = np.hstack((C0, GY0, GX0, Y0, X0, CC0, B0))
    samples1 = np.hstack((C1, GY1, GX1, Y1, X1, CC1, B1))

  samples_cb = cb_template[template_all_y, template_all_x].reshape(-1, 1).astype("float64")
  samples_cr = cr_template[template_all_y, template_all_x].reshape(-1, 1).astype("float64")
  samples_cb0 = samples_cb[index_model0]
  samples_cb1 = samples_cb[index_model1]
  samples_cr0 = samples_cr[index_model0]
  samples_cr1 = samples_cr[index_model1]

  # regression
  if l2_regularisation == 0:
    coeffs_cb0, _, _, _ = np.linalg.lstsq(samples0, samples_cb0, rcond = None)
    coeffs_cb1, _, _, _ = np.linalg.lstsq(samples1, samples_cb1, rcond = None)
    coeffs_cr0, _, _, _ = np.linalg.lstsq(samples0, samples_cr0, rcond = None)
    coeffs_cr1, _, _, _ = np.linalg.lstsq(samples1, samples_cr1, rcond = None)
  else:
    coeffs_cb0 = np.linalg.inv(samples0.T @ samples0 + l2_regularisation * samples0.shape[1] * np.eye(samples0.shape[1])) @ samples0.T @ samples_cb0
    coeffs_cb1 = np.linalg.inv(samples1.T @ samples1 + l2_regularisation * samples1.shape[1] * np.eye(samples1.shape[1])) @ samples1.T @ samples_cb1
    coeffs_cr0 = np.linalg.inv(samples0.T @ samples0 + l2_regularisation * samples0.shape[1] * np.eye(samples0.shape[1])) @ samples0.T @ samples_cr0
    coeffs_cr1 = np.linalg.inv(samples1.T @ samples1 + l2_regularisation * samples1.shape[1] * np.eye(samples1.shape[1])) @ samples1.T @ samples_cr1

  mad_cb_template = 0.0
  mad_cr_template = 0.0
  if evaluate_on_template:
    predicted_cb0_template = np.dot(samples0, coeffs_cb0).astype(reconstructed_video.data_type)
    predicted_cr0_template = np.dot(samples0, coeffs_cr0).astype(reconstructed_video.data_type)
    predicted_cb1_template = np.dot(samples1, coeffs_cb1).astype(reconstructed_video.data_type)
    predicted_cr1_template = np.dot(samples1, coeffs_cr1).astype(reconstructed_video.data_type)
    sad_cb_template = np.sum(np.abs(predicted_cb0_template.astype("int32") - samples_cb0)) + np.sum(np.abs(predicted_cb1_template.astype("int32") - samples_cb1)) 
    sad_cr_template = np.sum(np.abs(predicted_cr0_template.astype("int32") - samples_cr0)) + np.sum(np.abs(predicted_cr1_template.astype("int32") - samples_cr1))
    mad_cb_template = sad_cb_template / (samples_cb0.shape[0] + samples_cb1.shape[0])
    mad_cr_template = sad_cr_template / (samples_cr0.shape[0] + samples_cr1.shape[0])

  # prediction
  luma_block_with_template = luma_template.copy()
  luma_block_with_template[template_top_height:-1, template_left_width:-1] = luma_block
  block_all_y, block_all_x = np.where(block_mask)
  C = luma_block_with_template[block_all_y, block_all_x].reshape(-1, 1).astype("float64")
  N = luma_block_with_template[block_all_y-1, block_all_x].reshape(-1, 1).astype("float64")
  S = luma_block_with_template[block_all_y+1, block_all_x].reshape(-1, 1).astype("float64")
  W = luma_block_with_template[block_all_y, block_all_x-1].reshape(-1, 1).astype("float64")
  E = luma_block_with_template[block_all_y, block_all_x+1].reshape(-1, 1).astype("float64")
  CC = np.square(C)
  B = np.ones_like(C, dtype = C.dtype) * (2 ** (reconstructed_video.bit_depth - 1))
  index_model0 = np.column_stack(np.where(C < template_luma_average))[:, 0]
  index_model1 = np.column_stack(np.where(C >= template_luma_average))[:, 0]
  if not glcccm:
    C0, N0, S0, W0, E0, CC0, B0 = C[index_model0], N[index_model0], S[index_model0], W[index_model0], E[index_model0], CC[index_model0], B[index_model0]
    C1, N1, S1, W1, E1, CC1, B1 = C[index_model1], N[index_model1], S[index_model1], W[index_model1], E[index_model1], CC[index_model1], B[index_model1]
    samples0 = np.hstack((C0, N0, S0, W0, E0, CC0, B0))
    samples1 = np.hstack((C1, N1, S1, W1, E1, CC1, B1))
  else:
    NW = luma_template[block_all_y-1, block_all_x-1].reshape(-1, 1).astype("float64")
    NE = luma_template[block_all_y-1, block_all_x+1].reshape(-1, 1).astype("float64")
    SW = luma_template[block_all_y+1, block_all_x-1].reshape(-1, 1).astype("float64")
    SE = luma_template[block_all_y+1, block_all_x+1].reshape(-1, 1).astype("float64")
    GY = (N * 2 + NW + NE - S * 2 - SW - SE)
    GX = (W * 2 + NW + SW - E * 2 - NE - SE)
    X, Y = np.meshgrid(range(luma_template.shape[1]), range(luma_template.shape[0]))
    X = np.reshape(X, luma_template.shape)[block_all_y, block_all_x].reshape(-1, 1).astype("float64")
    Y = np.reshape(Y, luma_template.shape)[block_all_y, block_all_x].reshape(-1, 1).astype("float64")
    C0, GY0, GX0, Y0, X0, CC0, B0 = C[index_model0], GY[index_model0], GX[index_model0], Y[index_model0], X[index_model0], CC[index_model0], B[index_model0]
    C1, GY1, GX1, Y1, X1, CC1, B1 = C[index_model1], GY[index_model1], GX[index_model1], Y[index_model1], X[index_model1], CC[index_model1], B[index_model1]
    samples0 = np.hstack((C0, GY0, GX0, Y0, X0, CC0, B0))
    samples1 = np.hstack((C1, GY1, GX1, Y1, X1, CC1, B1))

  predicted_cb0 = np.dot(samples0, coeffs_cb0).astype(reconstructed_video.data_type)
  predicted_cr0 = np.dot(samples0, coeffs_cr0).astype(reconstructed_video.data_type)
  predicted_cb1 = np.dot(samples1, coeffs_cb1).astype(reconstructed_video.data_type)
  predicted_cr1 = np.dot(samples1, coeffs_cr1).astype(reconstructed_video.data_type)

  predicted_cb = np.zeros_like(C, dtype = C.dtype)
  predicted_cr = np.zeros_like(C, dtype = C.dtype)
  predicted_cb[index_model0] = predicted_cb0
  predicted_cr[index_model0] = predicted_cr0
  predicted_cb[index_model1] = predicted_cb1
  predicted_cr[index_model1] = predicted_cr1
  cb_template[block_mask] = predicted_cb.flatten()
  cr_template[block_mask] = predicted_cr.flatten()

  predicted_cb_block = cb_template[template_top_height:-1, template_left_width:-1]
  predicted_cr_block = cr_template[template_top_height:-1, template_left_width:-1]

  if input_video.bit_depth == 8:
    sad_cb = np.sum(np.abs(((predicted_cb_block + 2) // 4).astype("int32") - ((cb_block + 2) // 4)))
    sad_cr = np.sum(np.abs(((predicted_cr_block + 2) // 4).astype("int32") - ((cr_block + 2) // 4)))
    predicted_cb_block = ((predicted_cb_block + 2) // 4).astype("uint8")
    predicted_cr_block = ((predicted_cr_block + 2) // 4).astype("uint8")
  else:
    sad_cb = np.sum(np.abs(predicted_cb_block.astype("int32") - cb_block))
    sad_cr = np.sum(np.abs(predicted_cr_block.astype("int32") - cr_block))

  mad_cb = sad_cb / (cb_block.size)
  mad_cr = sad_cr / (cr_block.size)

  return predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, (coeffs_cb0, coeffs_cb1, coeffs_cr0, coeffs_cr1), (mad_cb, mad_cr, mad_cb_template, mad_cr_template)

# Input: video struct, frame number, dimensions as (x, y, w, h), number of lines in template (usually 6)
# Output: predicted blocks, SADs, coefficients
def simulate_soft_classified_mm_cccm(video, frame_number, dimensions, template_lines_in_chroma, soft_classification_when_apply = False):
  x, y, w, h = dimensions

  blocks_and_templates = get_cccm_blocks_and_templates(video, frame_number, dimensions, template_lines_in_chroma)
  luma_block, luma_template, cb_block, cb_template, cr_block, cr_template, template_left_width, template_top_height, block_mask, template_mask = blocks_and_templates

  # sampling
  template_all_y, template_all_x = np.where(template_mask)
  if len(template_all_y) == 0:
    return None, None, 0, 0, [], [], [], []
  
  C = luma_template[template_all_y, template_all_x].reshape(-1, 1).astype("float64")
  N = luma_template[template_all_y-1, template_all_x].reshape(-1, 1).astype("float64")
  S = luma_template[template_all_y+1, template_all_x].reshape(-1, 1).astype("float64")
  W = luma_template[template_all_y, template_all_x-1].reshape(-1, 1).astype("float64")
  E = luma_template[template_all_y, template_all_x+1].reshape(-1, 1).astype("float64")
  CC = np.square(C)
  B = np.ones_like(C, dtype = C.dtype) * (2 ** (video.bit_depth - 1))

  template_luma_average = np.mean(C).astype(video.data_type)
  template_luma_max = np.max(C).astype(video.data_type)
  template_luma_min = np.min(C).astype(video.data_type)
  template_luma_model0_upperbound = ((template_luma_average.astype("int32") + template_luma_max) / 2).astype(video.data_type)
  template_luma_model1_lowerbound = ((template_luma_average.astype("int32") + template_luma_min) / 2).astype(video.data_type)
  index_model0 = np.column_stack(np.where(C <= template_luma_model0_upperbound))[:, 0]
  index_model1 = np.column_stack(np.where(C >= template_luma_model1_lowerbound))[:, 0]

  C0, N0, S0, W0, E0, CC0, B0 = C[index_model0], N[index_model0], S[index_model0], W[index_model0], E[index_model0], CC[index_model0], B[index_model0]
  C1, N1, S1, W1, E1, CC1, B1 = C[index_model1], N[index_model1], S[index_model1], W[index_model1], E[index_model1], CC[index_model1], B[index_model1]
  samples0 = np.hstack((C0, N0, S0, W0, E0, CC0, B0))
  samples1 = np.hstack((C1, N1, S1, W1, E1, CC1, B1))
  samples_cb = cb_template[template_all_y, template_all_x].reshape(-1, 1).astype("float64")
  samples_cr = cr_template[template_all_y, template_all_x].reshape(-1, 1).astype("float64")
  samples_cb0 = samples_cb[index_model0]
  samples_cb1 = samples_cb[index_model1]
  samples_cr0 = samples_cr[index_model0]
  samples_cr1 = samples_cr[index_model1]

  # regression
  coeffs_cb0, _, _, _ = np.linalg.lstsq(samples0, samples_cb0, rcond = None)
  coeffs_cb1, _, _, _ = np.linalg.lstsq(samples1, samples_cb1, rcond = None)
  coeffs_cr0, _, _, _ = np.linalg.lstsq(samples0, samples_cr0, rcond = None)
  coeffs_cr1, _, _, _ = np.linalg.lstsq(samples1, samples_cr1, rcond = None)

  # prediction
  luma_block_with_template = luma_template.copy()
  luma_block_with_template[template_top_height:-1, template_left_width:-1] = luma_block
  block_all_y, block_all_x = np.where(block_mask)
  C = luma_block_with_template[block_all_y, block_all_x].reshape(-1, 1).astype("float64")
  N = luma_block_with_template[block_all_y-1, block_all_x].reshape(-1, 1).astype("float64")
  S = luma_block_with_template[block_all_y+1, block_all_x].reshape(-1, 1).astype("float64")
  W = luma_block_with_template[block_all_y, block_all_x-1].reshape(-1, 1).astype("float64")
  E = luma_block_with_template[block_all_y, block_all_x+1].reshape(-1, 1).astype("float64")
  CC = np.square(C)
  B = np.ones_like(C, dtype = C.dtype) * (2 ** (video.bit_depth - 1))

  if soft_classification_when_apply:
    weight1 = (C - template_luma_model1_lowerbound) / (template_luma_model0_upperbound - template_luma_model1_lowerbound)
  else:
    weight1 = (C > template_luma_average).astype(int)

  weight1[weight1 < 0] = 0
  weight1[weight1 > 1] = 1
  weight0 = 1 - weight1

  samples = np.hstack((C, N, S, W, E, CC, B))

  predicted_cb0 = np.dot(samples, coeffs_cb0)
  predicted_cb1 = np.dot(samples, coeffs_cb1)
  predicted_cr0 = np.dot(samples, coeffs_cr0)
  predicted_cr1 = np.dot(samples, coeffs_cr1)

  predicted_cb = predicted_cb0 * weight0 + predicted_cb1 * weight1
  predicted_cr = predicted_cr0 * weight0 + predicted_cr1 * weight1

  cb_template[block_mask] = predicted_cb.astype(video.data_type).flatten()
  cr_template[block_mask] = predicted_cr.astype(video.data_type).flatten()

  predicted_cb_block = cb_template[template_top_height:-1, template_left_width:-1]
  predicted_cr_block = cr_template[template_top_height:-1, template_left_width:-1]

  sad_cb = np.sum(np.abs(predicted_cb_block.astype("int32") - cb_block))
  sad_cr = np.sum(np.abs(predicted_cr_block.astype("int32") - cr_block))

  return predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, (coeffs_cb0, coeffs_cb1, coeffs_cr0, coeffs_cr1)
