import math  
import numpy as np

class VideoInformation:
  def __init__(self, file_path, width, height, bit_depth):
    self.file_path = file_path
    self.width = width
    self.height = height
    self.pixel_size = math.ceil(bit_depth / 8)
    self.data_type = np.uint8 if self.pixel_size ==  1 else np.uint16

  def frame_size_in_bytes(self):
    return self.width * self.height * self.pixel_size * 3 / 2

  def luma_stride_in_bytes(self):
    return self.width * self.pixel_size

  def chroma_stride_in_bytes(self):
    return self.width * self.pixel_size / 2

