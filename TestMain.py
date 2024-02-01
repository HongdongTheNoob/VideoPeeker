import os
import cv2
import numpy as np
import CCP.CCCMSim as CCCMSim
import EIP.EIPSim as EIPSim
# import MIP.MIPSim as MIPSim
import BmsStatsScanner

import GetBlock
from ClassVideoInfo import VideoInformation

def replace_extension(file_path, new_extension):
    base_path, old_extension = os.path.splitext(file_path)
    new_file_path = base_path + new_extension
    return new_file_path

def simulation_use_case():
  file_path = "D:/Data/xcy_test/ClassC/BasketballDrill_832x480_50.yuv"
  my_video = VideoInformation(file_path, 832, 480, 8)
  dimensions = (320, 64, 64, 64)
  position_string = '_(' + str(dimensions[0]) + ',' + str(dimensions[1]) + ')_' + str(dimensions[2]) + 'x' + str(dimensions[3])

  block, template = GetBlock.get_block(my_video, 0, dimensions, 'y', 12)
  cv2.imwrite(replace_extension(file_path, '_y' + position_string + '.png'), block)
  cv2.imwrite(replace_extension(file_path, '_y_template' + position_string + '_12.png'), template)

  block, template = GetBlock.get_block(my_video, 0, dimensions, 'cb', 12)
  cv2.imwrite(replace_extension(file_path, '_cb' + position_string + '.png'), block)
  cv2.imwrite(replace_extension(file_path, '_cb_template' + position_string + '_12.png'), template)

  block, template = GetBlock.get_block(my_video, 0, dimensions, 'cr', 12)
  cv2.imwrite(replace_extension(file_path, '_cr' + position_string + '.png'), block)
  cv2.imwrite(replace_extension(file_path, '_cr_template' + position_string + '_12.png'), template)

  predicted_cb_block, predicted_cr_block, x_cb, x_cr, sad_cb, sad_cr = CCCMSim.simulate_cccm(my_video, 0, dimensions, 6)
  cv2.imwrite(replace_extension(file_path, '_cb_predicted' + position_string + '.png'), predicted_cb_block)
  cv2.imwrite(replace_extension(file_path, '_cr_predicted' + position_string + '.png'), predicted_cr_block)
  print(sad_cb, sad_cr)

  predicted_block, coeffs, sad = EIPSim.simulate_eip(my_video, 0, dimensions, 6)
  cv2.imwrite(replace_extension(file_path, '_y_EIP' + position_string + '.png'), predicted_block)
  print(coeffs)
  print(sad)


if __name__ == "__main__":
  # simulation_use_case()

  file_path = "D:\Data\Bitstream\ECM11_LF0\ClassC\BasketballDrill\str-BasketballDrill-AI-22.vtmbmsstats"

  BmsStatsScanner.bms_stats_scan(file_path)
  