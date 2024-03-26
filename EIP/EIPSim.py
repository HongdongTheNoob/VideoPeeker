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
def simulate_eip(input_video, frame_number, dimensions, template_lines, l2_regularisation = 0, reconstructed_video = None, evaluate_on_template = False):
  x, y, w, h = dimensions
  luma_block, luma_template = GetBlock.get_block(input_video, frame_number, dimensions, 'y', template_lines)
  h = luma_block.shape[0]
  w = luma_block.shape[1]

  if reconstructed_video is not None:
    _, luma_template = GetBlock.get_block(reconstructed_video, frame_number, dimensions, 'y', template_lines)
    if input_video.bit_depth == 8:
      luma_block = luma_block.astype(reconstructed_video.data_type) * 4
  else:
    reconstructed_video = input_video
    # use 10-bit as operation bit-depth
    if input_video.bit_depth == 8:
      reconstructed_video.bit_depth = 10
      reconstructed_video.data_type = np.uint16
      luma_block = luma_block.astype("uint16") * 4
      luma_template = luma_template.astype("uint16") * 4
  
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
  invalid_samples = np.where((template_all_y < 3) & (template_all_x < 3))
  template_all_y = np.delete(template_all_y, invalid_samples)
  template_all_x = np.delete(template_all_x, invalid_samples)
  samples = np.zeros((len(template_all_y), 0)).astype("float64")
  for s in range(len(offset_y)):
    samples = np.hstack((samples, luma_template[template_all_y + offset_y[s], template_all_x + offset_x[s]].reshape(-1, 1).astype("float64")))
  samples_y = luma_template[template_all_y, template_all_x].astype("float64")

  # regression
  if l2_regularisation == 0:
    coeffs, _, _, _ = np.linalg.lstsq(samples, samples_y, rcond = None)
  else:
    coeffs = np.linalg.inv(samples.T @ samples + l2_regularisation * samples.shape[1] * np.eye(samples.shape[1])) @ samples.T @ samples_y

  mad_template = 0.0
  cod = 0.0
  skewness = 0.0
  kurtosis = 0.0
  outlier_ratio = 0.0
  if evaluate_on_template:
    predicted_template = np.minimum(np.maximum(0, np.dot(samples, coeffs)), 1023).astype(reconstructed_video.data_type)
    if input_video.bit_depth == 8:
      predicted_template = (predicted_template + 2) // 4
      samples_compare = (samples_y + 2) // 4
    else:
      samples_compare = samples_y
    sad_template = np.sum(np.abs(predicted_template.astype("int32") - samples_compare))
    mad_template = sad_template / samples.shape[0]
    mean = np.average(samples_compare)
    TSS = np.sum((samples_compare.astype("int32") - mean) ** 2)
    RSS = np.sum((predicted_template.astype("int32") - samples_compare) ** 2)
    cod = 1 - RSS / TSS

    error_template = predicted_template.astype("int32") - samples_compare
    mean_error = np.average(error_template)
    error_mean_removed = error_template - mean_error
    mean_square_error = np.average(error_mean_removed ** 2)
    mean_cubic_error = np.average(error_mean_removed ** 3)
    mean_quadratic_error = np.average(error_mean_removed ** 4)
    skewness = mean_cubic_error / (mean_square_error ** 1.5) if mean_square_error > 0 else 0.0
    kurtosis = mean_quadratic_error / (mean_square_error ** 2) if mean_square_error > 0 else 0.0
    outlier_count = np.sum(error_mean_removed > 2.0 * mean_error)
    outlier_ratio = outlier_count / error_mean_removed.size

  # prediction
  for r in range(template_top_height, template_top_height + h):
    for c in range(template_left_width, template_left_width + w):
      neighbours = []
      for s in range(len(offset_y)):
        neighbour_y = max(r + offset_y[s], 0)
        neighbour_x = max(c + offset_x[s], 0)
        neighbours.append(luma_template[neighbour_y, neighbour_x].astype("float64"))
      luma_template[r, c] = np.minimum(np.maximum(0, np.dot(neighbours, coeffs)), 1023).astype(luma_template.dtype)

  predicted_block = luma_template[template_top_height:template_top_height+h, template_left_width:template_left_width+w].astype(reconstructed_video.data_type)

  if input_video.bit_depth == 8:
    sad = np.sum(np.abs(((predicted_block + 2) // 4).astype("int32") - ((luma_block + 2) // 4)))
    predicted_block = ((predicted_block + 2) // 4).astype("uint8")
  else:
    sad = np.sum(np.abs(predicted_block.astype("int32") - luma_block))

  mad = sad / (predicted_block.size)

  return predicted_block, sad, coeffs, (mad, mad_template, cod, skewness, kurtosis, outlier_ratio)


# Input: video struct, frame number, dimensions as (x, y, w, h), number of lines in template (usually 6)
# Output: predicted blocks, coefficients, SADs
def simulate_alternative_eip(video, frame_number, dimensions, template_lines):
  x, y, w, h = dimensions
  luma_block, luma_template = GetBlock.get_block(video, frame_number, dimensions, 'y', template_lines)
  h = luma_block.shape[0]
  w = luma_block.shape[1]
  
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
  offset_y = [0,-1,0,-1,-2,0,-1,-2,-3]
  offset_x = [-1,0,-2,-1,0,-3,-2,-1,0]

  # sampling
  template_all_y, template_all_x = np.where(template_mask)
  invalid_samples = np.where((template_all_y < 3) & (template_all_x < 3))
  template_all_y = np.delete(template_all_y, invalid_samples)
  template_all_x = np.delete(template_all_x, invalid_samples)
  samples = np.zeros((len(template_all_y), 0)).astype("float64")
  for s in range(len(offset_y)):
    samples = np.hstack((samples, luma_template[template_all_y + offset_y[s], template_all_x + offset_x[s]].reshape(-1, 1).astype("float64")))
  for s in range(3):
    samples = np.hstack((samples, np.square(luma_template[template_all_y + offset_y[s], template_all_x + offset_x[s]].reshape(-1, 1).astype("float64"))))

  X, Y = np.meshgrid(range(luma_template.shape[1]), range(luma_template.shape[0]))
  X = np.reshape(X, luma_template.shape)[template_all_y, template_all_x].reshape(-1, 1).astype("float64")
  Y = np.reshape(Y, luma_template.shape)[template_all_y, template_all_x].reshape(-1, 1).astype("float64")
  B = np.ones((len(template_all_y), 1)).astype("float64") * (2 ** (video.bit_depth - 1))
  samples = np.hstack((samples, X, Y, B))

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
        neighbours.append(luma_template[neighbour_y, neighbour_x].astype("float64"))
      for s in range(3):
        neighbour_y = max(r + offset_y[s], 0)
        neighbour_x = max(c + offset_x[s], 0)
        neighbours.append(luma_template[neighbour_y, neighbour_x].astype("float64") ** 2)
      neighbours.append(c)
      neighbours.append(r)
      neighbours.append((2.0 ** (video.bit_depth - 1)))
      luma_template[r, c] = np.minimum(np.maximum(0, np.dot(neighbours, coeffs)), 1023).astype(luma_template.dtype)

  predicted_block = luma_template[template_top_height:template_top_height+h, template_left_width:template_left_width+w]

  sad = np.sum(np.abs(predicted_block.astype("int32") - luma_block))

  return predicted_block, sad, coeffs