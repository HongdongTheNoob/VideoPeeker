import os
import cv2
import numpy as np
import sys
import EIP.EIPSim as EIPSim
import BmsStatsScanner

import GetBlock
from ClassVideoInfo import VideoInformation

def replace_extension(file_path, new_extension):
    base_path, old_extension = os.path.splitext(file_path)
    new_file_path = base_path + new_extension
    return new_file_path

def simulation_loop():
  file_path = "/Data/xcy_test/ClassC/BasketballDrill_832x480_50.yuv"
  bms_file_path = "/Data/Bitstream/ECM11_LF0/ClassC/BasketballDrill/str-BasketballDrill-AI-22.vtmbmsstats"
  video_width = 832
  video_height = 480
  my_video = VideoInformation(file_path, video_width, video_height, 8)

  video_folder = os.path.dirname(file_path)
  output_folder = os.path.join(video_folder, "EIP_Sim")
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  # Collect blocks
  # With regular grid
  # all_blocks = []
  # block_width = 64
  # block_height = 64
  # for frame in range(1):
  #   for y in list(range(0, video_height, block_height)):
  #     for x in list(range(0, video_width, block_width)):
  #       dimensions = (x, y, block_width, block_height)
  #       all_blocks.append((frame, dimensions))
  
  # By grabbing from bms stats file
  all_blocks = BmsStatsScanner.collect_blocks(bms_file_path, frame_range = range(1), target_string = "")

  # Loop through
  for block in all_blocks:
    frame, dimensions = block
    print(frame, dimensions)

    position_string = '_frame_' + str(frame) + '_(' + str(dimensions[0]) + ',' + str(dimensions[1]) + ')_' + str(dimensions[2]) + 'x' + str(dimensions[3])
    predicted_block, coeffs, sad = EIPSim.simulate_eip(my_video, frame, dimensions, 6)
    
    output_file = os.path.join(output_folder, os.path.basename(my_video.file_path))
    cv2.imwrite(replace_extension(output_file, '_y_EIP' + position_string + '.png'), predicted_block)
    print("EIP coeffs:", coeffs.reshape((1, -1)))
    print("Sad: ", sad)

if __name__ == "__main__":
  simulation_loop()
