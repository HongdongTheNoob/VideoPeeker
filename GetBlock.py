import numpy as np
from ClassVideoInfo import VideoInformation

def apply_121_filter_and_select_odd(row):
    filtered_row = np.convolve(row, np.array([1, 2, 1]), mode='same')
    return filtered_row[1::2]

def get_block(video, frame_number, dimensions, colour_component, template_lines_in_luma):
  x, y, w, h = dimensions
  frame_start = frame_number * video.frame_size_in_bytes()
  component_start = frame_start
  component_stride = video.luma_stride_in_bytes()
  template_lines = template_lines_in_luma

  if x < 0:
    w += x
    x = 0
  if y < 0:
    h += y
    y = 0
  w = min(w, video.width - x)
  h = min(h, video.height - y)

  if colour_component == 'cb':
    component_start = frame_start + video.luma_size_in_bytes()
    component_stride = component_stride // 2
    template_lines = template_lines // 2
    x, y, w, h = x // 2, y // 2, w // 2, h // 2
  elif colour_component == 'cr':
    component_start = frame_start + video.luma_size_in_bytes() + video.chroma_size_in_bytes()
    component_stride = component_stride // 2
    template_lines = template_lines // 2
    x, y, w, h = x // 2, y // 2, w // 2, h // 2
  elif colour_component != 'y':
    return

  block = np.zeros((h, w)).astype(video.data_type)
  template = np.zeros((h + template_lines, w + template_lines)).astype(video.data_type)

  with open(video.file_path, "rb") as video_file:
    block_start = int(component_start + y * component_stride + x * video.pixel_size)

    # load block
    video_file.seek(block_start)
    for r in range(h):
      line = np.fromfile(video_file, count = w, dtype = video.data_type)
      block[r][:] = line
      video_file.seek(component_stride - w * video.pixel_size, 1)

    # load template
    if template_lines > 0:
      start_read_x = max(x - template_lines, 0)
      start_read_y = max(y - template_lines, 0)
      start_write_x = template_lines + start_read_x - x
      start_write_y = template_lines + start_read_y - y

      block_start = int(component_start + start_read_y * component_stride + start_read_x * video.pixel_size)
      video_file.seek(block_start)
      for r in range(start_write_y, template_lines):
        line = np.fromfile(video_file, count = x + w - start_read_x, dtype = video.data_type)
        template[r][start_write_x:] = line
        video_file.seek(component_stride - (x + w - start_read_x) * video.pixel_size, 1)
      for r in range(template_lines, template_lines + h):
        line = np.fromfile(video_file, count = x - start_read_x, dtype = video.data_type)
        template[r][start_write_x:start_write_x+len(line)] = line
        video_file.seek(component_stride - (x - start_read_x) * video.pixel_size, 1)
      template = template[start_write_y:, start_write_x:]

  return block, template

def get_downsampled_block(video, frame_number, dimensions, template_lines_in_luma):
  x, y, w, h = dimensions
  w = min(w, video.width - x)
  h = min(h, video.height - y)
  start_get_x = max(x - template_lines_in_luma, 0)
  start_get_y = max(y - template_lines_in_luma, 0)
  get_w = w + x - start_get_x
  get_h = h + y - start_get_y

  if start_get_x > 0:
    original_block, _ = get_block(video, frame_number, (start_get_x - 1, start_get_y, get_w + 1, get_h), 'y', 0)
  else:
    original_block, _ = get_block(video, frame_number, (start_get_x, start_get_y, get_w, get_h), 'y', 0)
    new_column = original_block[:, 0].reshape(-1, 1)
    original_block = np.hstack((new_column, original_block))
  
  reshaped_block = original_block.reshape(-1, 2, original_block.shape[1])
  new_block_with_template = np.sum(reshaped_block, axis=1)
  new_block_with_template = np.apply_along_axis(apply_121_filter_and_select_odd, axis=1, arr=new_block_with_template)
  new_block_with_template = np.round(new_block_with_template / 8).astype(video.data_type)

  template_lines = template_lines_in_luma // 2
  block = new_block_with_template[(y - start_get_y) // 2:, (x - start_get_x) // 2:].copy()
  new_block_with_template[(y - start_get_y) // 2:, (x - start_get_x) // 2:] = 0

  return block, new_block_with_template
