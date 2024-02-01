import sys
sys.path.append('..')
import numpy as np
import cv2

import GetBlock
import MipData

def mip_simulation(video, frame_number, dimensions, colour_component, mip_index):
  x, y, w, h = dimensions
  block, template = GetBlock.get_block(video, frame_number, dimensions, colour_component, 1)

  template_left_width = min(1, x)
  template_top_height = min(1, y)
  