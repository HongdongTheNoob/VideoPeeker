import os
import cv2
import numpy as np
import CCP.CCCMSim as CCCMSim
import EIP.EIPSim as EIPSim
# import MIP.MIPSim as MIPSim
import BmsStatsScanner

import VideoDataset
from ClassVideoInfo import VideoInformation

import GetBlock
import Interpolations
import Filters

def replace_extension(file_path, new_extension):
    base_path, old_extension = os.path.splitext(file_path)
    new_file_path = base_path + new_extension
    return new_file_path

def simulation_use_case():
  file_path = "D:/Data/xcy_test/ClassC/BasketballDrill_832x480_50.yuv"
  my_video = VideoInformation(file_path, 832, 480, 8)
  dimensions = (64, 64, 64, 64)
  position_string = '_(' + str(dimensions[0]) + ',' + str(dimensions[1]) + ')_' + str(dimensions[2]) + 'x' + str(dimensions[3])

  block, template = GetBlock.get_block(my_video, 0, dimensions, 'y', 12)
  cv2.imwrite(replace_extension(file_path, '_y' + position_string + '.png'), np.kron(block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_y_template' + position_string + '_12.png'), np.kron(template, np.ones((8, 8))))

  block, template = GetBlock.get_block(my_video, 0, dimensions, 'cb', 12)
  cv2.imwrite(replace_extension(file_path, '_cb' + position_string + '.png'), np.kron(block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_cb_template' + position_string + '_12.png'), np.kron(template, np.ones((8, 8))))

  block, template = GetBlock.get_block(my_video, 0, dimensions, 'cr', 12)
  cv2.imwrite(replace_extension(file_path, '_cr' + position_string + '.png'), np.kron(block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_cr_template' + position_string + '_12.png'), np.kron(template, np.ones((8, 8))))

  predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs = CCCMSim.simulate_cccm(my_video, 0, dimensions, 6)
  cv2.imwrite(replace_extension(file_path, '_cb_predicted_cccm' + position_string + '.png'), np.kron(predicted_cb_block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_cr_predicted_cccm' + position_string + '.png'), np.kron(predicted_cr_block, np.ones((8, 8))))

  predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs = CCCMSim.simulate_cccm(my_video, 0, dimensions, 6, glcccm=1)
  cv2.imwrite(replace_extension(file_path, '_cb_predicted_glcccm' + position_string + '.png'), np.kron(predicted_cb_block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_cr_predicted_glcccm' + position_string + '.png'), np.kron(predicted_cr_block, np.ones((8, 8))))

  predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs, mads = CCCMSim.simulate_mm_cccm(my_video, 0, dimensions, 6)
  cv2.imwrite(replace_extension(file_path, '_cb_predicted_mmlm' + position_string + '.png'), np.kron(predicted_cb_block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_cr_predicted_mmlm' + position_string + '.png'), np.kron(predicted_cr_block, np.ones((8, 8))))

  predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs, mads = CCCMSim.simulate_mm_cccm(my_video, 0, dimensions, 6, glcccm=1)
  cv2.imwrite(replace_extension(file_path, '_cb_predicted_mm_glcccm' + position_string + '.png'), np.kron(predicted_cb_block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_cr_predicted_mm_glcccm' + position_string + '.png'), np.kron(predicted_cr_block, np.ones((8, 8))))

  predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs = CCCMSim.simulate_soft_classified_mm_cccm(my_video, 0, dimensions, 6)
  cv2.imwrite(replace_extension(file_path, '_cb_predicted_new_mmlm' + position_string + '.png'), np.kron(predicted_cb_block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_cr_predicted_new_mmlm' + position_string + '.png'), np.kron(predicted_cr_block, np.ones((8, 8))))

  # predicted_block, coeffs, sad = EIPSim.simulate_eip(my_video, 0, dimensions, 6)
  # cv2.imwrite(replace_extension(file_path, '_y_EIP' + position_string + '.png'), predicted_block)

  for step in range(16):
    subpel_dimensions = (64 + step/16.0, 64 + step/16.0, 64, 64)
    position_string = '_(' + str(int(subpel_dimensions[0])) + '+' + str(step) + '%16, ' + str(int(subpel_dimensions[1])) + '+' + str(step) + '%16)_'
    print(position_string)
    dimension_string = position_string + str(subpel_dimensions[2]) + 'x' + str(subpel_dimensions[3])
    subpel_block_y = Interpolations.get_block_subpixel(my_video, 0, subpel_dimensions, 'y', Interpolations.luma_filter_12)
    subpel_block_cb = Interpolations.get_block_subpixel(my_video, 0, subpel_dimensions, 'cb', Interpolations.chroma_filter_6)
    subpel_block_cr = Interpolations.get_block_subpixel(my_video, 0, subpel_dimensions, 'cr', Interpolations.chroma_filter_6)
    subpel_block = np.row_stack((subpel_block_y, np.column_stack((subpel_block_cb, subpel_block_cr))))
    cv2.imwrite(replace_extension(file_path, '_ycbcr' + dimension_string + '.png'), np.kron(subpel_block, np.ones((8, 8))))

def get_block_examples():
  video_class = "C"
  sequence_name = "BasketballDrill"
  file_path, video_dimensions = VideoDataset.find_video_properties(video_class, sequence_name)
  if file_path == None or len(file_path) == 0:
    print("Video not found")
    return
  
  my_video = VideoInformation(file_path, video_dimensions[0], video_dimensions[1], video_dimensions[2])

  dimensions = (128, 128, 32, 32)
  position_string = '_(' + str(dimensions[0]) + ',' + str(dimensions[1]) + ')_' + str(dimensions[2]) + 'x' + str(dimensions[3])
  block, template = GetBlock.get_block(my_video, 0, dimensions, 'y', 0)
  # cv2.imwrite(replace_extension(file_path, '_y' + position_string + '.png'), np.kron(block, np.ones((8, 8))))

  for t in range(16):
    dimensions = (128 + t/4.0, 128 - t/4.0, 32, 32)
    position_string = str(t)
    sub_pixel_block_y = Interpolations.get_block_subpixel(my_video, 0, dimensions, 'y', Interpolations.luma_filter_12)
    cv2.imwrite(replace_extension(file_path, '_y_filter12_' + position_string + '.png'), np.kron(sub_pixel_block_y, np.ones((8, 8))))
    
    sub_pixel_block_cb = Interpolations.get_block_subpixel(my_video, 0, dimensions, 'cb', Interpolations.chroma_filter_6)
    cv2.imwrite(replace_extension(file_path, '_cb_filter_' + position_string + '.png'), np.kron(sub_pixel_block_cb, np.ones((8, 8))))
    
    sub_pixel_block_cr = Interpolations.get_block_subpixel(my_video, 0, dimensions, 'cr', Interpolations.chroma_filter_6)
    cv2.imwrite(replace_extension(file_path, '_cr_filter_' + position_string + '.png'), np.kron(sub_pixel_block_cr, np.ones((8, 8))))

    sub_pixel_block_chroma = np.column_stack((sub_pixel_block_cb, sub_pixel_block_cr))
    sub_pixel_block_yuv = np.row_stack((sub_pixel_block_y, sub_pixel_block_chroma))
    cv2.imwrite(replace_extension(file_path, '_yuv_filter_' + position_string + '.png'), np.kron(sub_pixel_block_yuv, np.ones((8, 8))))


if __name__ == "__main__":
  get_block_examples()
  # simulation_use_case()

  # file_path = "D:\Data\Bitstream\ECM11_LF0\ClassC\BasketballDrill\str-BasketballDrill-AI-22.vtmbmsstats"
  # BmsStatsScanner.bms_stats_scan(file_path)
  