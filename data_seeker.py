import numpy as np
from ClassVideoInfo import VideoInformation

def seek_block(video, frame_number, x, y, w, h, luma_template_width, get_luma, get_chroma)
  luma_block = np.zeros((h, w)).astype(video.data_type)
  chroma_block = np.zeros((h//2, w//2)).astype(video.data_type)
  luma_template = np.zeros((h + luma_template_width, w + luma_template_width)).astype(video.data_type)
  chroma_template = np.zeros((h + luma_template_width, w + luma_template_width)).astype(video.data_type)

  with open(video.file_path, "rb") as video_file:
    frame_start = frame_number * video.frame_size_in_bytes()

    if get_luma:
      luma_block_start = frame_start + y * video.luma_stride_in_bytes() + x * video.pixel_size
      video_file.seek(luma_block_start)
      for r in range(h):
        line = video_file.read(w, dtype = video.data_type)
        luma_block[r][:] = line

      if luma_template_width > 0:
        start_x = max(x - luma_template_width, 0)
        start_y = max(y - luma_template_width, 0)
        

    if get_chroma:


  return luma_block, luma_template, chroma_block, chroma_template