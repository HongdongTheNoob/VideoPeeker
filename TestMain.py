import os
import cv2
import numpy as np
import CCP.CCCMSim as CCCMSim

import GetBlock
from ClassVideoInfo import VideoInformation

def replace_extension(file_path, new_extension):
    base_path, old_extension = os.path.splitext(file_path)
    new_file_path = base_path + new_extension
    return new_file_path

if __name__ == "__main__":
  file_path = "D:/Data/xcy_test/ClassC/BasketballDrill_832x480_50.yuv"
  my_video = VideoInformation(file_path, 832, 480, 8)
  dimensions = (64, 64, 64, 64)

  block, template = GetBlock.get_block(my_video, 0, dimensions, 'y', 12)
  cv2.imwrite(replace_extension(file_path, '_y_(64,64)_64x64.png'), block)
  cv2.imwrite(replace_extension(file_path, '_y_template_(64,64)_64x64_12.png'), template)

  block, template = GetBlock.get_block(my_video, 0, dimensions, 'cb', 12)
  cv2.imwrite(replace_extension(file_path, '_cb_(64,64)_64x64.png'), block)
  cv2.imwrite(replace_extension(file_path, '_cb_template_(64,64)_64x64_12.png'), template)

  block, template = GetBlock.get_block(my_video, 0, dimensions, 'cr', 12)
  cv2.imwrite(replace_extension(file_path, '_cr_(64,64)_64x64.png'), block)
  cv2.imwrite(replace_extension(file_path, '_cr_template_(64,64)_64x64_12.png'), template)

  predicted_cb_block, predicted_cr_block, x_cb, x_cr, sad_cb, sad_cr = CCCMSim.simulate_cccm(my_video, 0, dimensions, 6)
  cv2.imwrite(replace_extension(file_path, '_cb_predicted_(64,64)_64x64.png'), predicted_cb_block)
  cv2.imwrite(replace_extension(file_path, '_cr_predicted_(64,64)_64x64.png'), predicted_cr_block)
  print(sad_cb, sad_cr)