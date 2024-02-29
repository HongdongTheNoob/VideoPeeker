import os
import cv2
import numpy as np
import json
import EIP.EIPSim as EIPSim
import VideoDataset
import BmsStatsScanner

import GetBlock
import Filters
from ClassVideoInfo import VideoInformation

def replace_extension(file_path, new_extension):
    base_path, old_extension = os.path.splitext(file_path)
    new_file_path = base_path + new_extension
    return new_file_path

def simulation_use_case():
  file_path = "D:/Data/xcy_test/ClassC/BasketballDrill_832x480_50.yuv"
  my_video = VideoInformation(file_path, 832, 480, 8)
  dimensions = (128, 128, 64, 64)
  position_string = '_(' + str(dimensions[0]) + ',' + str(dimensions[1]) + ')_' + str(dimensions[2]) + 'x' + str(dimensions[3])

  predicted_block, sad, coeffs = EIPSim.simulate_eip(my_video, 0, dimensions, 6)
  cv2.imwrite(replace_extension(file_path, '_y_EIP' + position_string + '.png'), predicted_block)
  print("EIP coeffs:", coeffs.reshape((1, -1)))
  print("Sad: ", sad)

  predicted_block, sad, coeffs = EIPSim.simulate_eip(my_video, 0, dimensions, 6, l2_regularisation = 200.0)
  cv2.imwrite(replace_extension(file_path, '_y_EIP_l2' + position_string + '.png'), predicted_block)
  print("EIP-L2 coeffs:", coeffs.reshape((1, -1)))
  print("Sad: ", sad)

  predicted_block, sad, coeffs = EIPSim.simulate_alternative_eip(my_video, 0, dimensions, 6)
  cv2.imwrite(replace_extension(file_path, '_y_Custom_EIP' + position_string + '.png'), predicted_block)
  print("Custom EIP coeffs:", coeffs.reshape((1, -1)))
  print("Sad: ", sad)

def simulate_eip_looped():
  video_class = "D"
  qps = ["22"]

  test_l2_regularisation = True
  l2_lambdas = [32, 64, 128, 256, 512]
  test_alternative_eip = True

  load_block_stats = False

  for i in range(len(VideoDataset.video_sequences[video_class])):
    sequence = VideoDataset.video_sequences[video_class][i]
    file_path = VideoDataset.video_file_names[video_class][i]
    video_width, video_height, video_bitdepth = VideoDataset.video_width_height_bitdepth[video_class][i]
    my_video = VideoInformation(file_path, video_width, video_height, video_bitdepth)

    video_folder = os.path.dirname(file_path)

    for qp in qps:
      bms_files = VideoDataset.find_stats_files(os.path.join(VideoDataset.bms_file_base_path, 'Class' + video_class), sequence, qp)

      if not bms_files:
        continue

      if not load_block_stats:
        all_blocks = BmsStatsScanner.collect_blocks(bms_files[0], frame_range = range(1), target_string = "EIPFlag=1")
        with open("./Tests/EIP_Sim/" + sequence + "-" + qp + ".json", "w") as file:
          json.dump(all_blocks, file)   
      else: 
        with open("./Tests/EIP_Sim/" + sequence + "-" + qp + ".json", "r") as file:
          all_blocks = json.load(file)    

      # Loop
      pixel_count = 0
      total_sad_eip = 0
      overall_sad_change_l2 = [0 for _ in l2_lambdas]
      overall_sad_gain_l2 = [0 for _ in l2_lambdas]
      overall_sad_change_alternative = 0
      overall_sad_gain_alternative = 0

      eip_test_log_file = open("./Tests/EIP_Sim/" + sequence + "-" + qp + ".txt", "w")
      for block in all_blocks:
        frame, dimensions = block
        if dimensions[0] == 0 and dimensions[1] == 0:
          continue

        position_string = '_frame_' + str(frame) + '_(' + str(dimensions[0]) + ',' + str(dimensions[1]) + ')_' + str(dimensions[2]) + 'x' + str(dimensions[3])
        eip_test_log_file.write("frame " + str(frame) + " " + str(dimensions) + "\n")

        block, _ = GetBlock.get_block(my_video, frame, dimensions, 'y', 0)

        predicted_block, sad_eip, _ = EIPSim.simulate_eip(my_video, 0, dimensions, 6)
        eip_test_log_file.write(" ".join(["SAD EIP: ", str(sad_eip), "\n"]))
        total_sad_eip += sad_eip

        if test_l2_regularisation:
          for i in range(len(l2_lambdas)):
            predicted_block, sad_eip_l2, _ = EIPSim.simulate_eip(my_video, 0, dimensions, 6, l2_regularisation = l2_lambdas[i])
            eip_test_log_file.write(" ".join(["SAD EIP-L2-" + str(l2_lambdas[i]) + ": ", str(sad_eip_l2), " change ", str(sad_eip_l2 - sad_eip), "\n"]))
            overall_sad_change_l2[i] += sad_eip_l2 - sad_eip
            overall_sad_gain_l2[i] += max(0, sad_eip - sad_eip_l2)

        if test_alternative_eip:
          predicted_block, sad_eip_alternative, _ = EIPSim.simulate_alternative_eip(my_video, 0, dimensions, 6)
          eip_test_log_file.write(" ".join(["SAD EIP-Alternative: ", str(sad_eip_l2), " change ", str(sad_eip_alternative - sad_eip), "\n"]))
          overall_sad_change_alternative += sad_eip_alternative - sad_eip
          overall_sad_gain_alternative += max(0, sad_eip - sad_eip_alternative)

        pixel_count += dimensions[2] * dimensions[3]
      
      print(sequence, qp)
      print("Pixel count: ", pixel_count)
      print("EIP total SAD: ", total_sad_eip)
      if test_alternative_eip:
        print("Overall SAD change alternative EIP: ", overall_sad_change_alternative)
      if test_l2_regularisation:
        for i in range(len(l2_lambdas)):
          print("Overall SAD change L2 lambda =", str(l2_lambdas[i]), ": ", overall_sad_change_l2[i])

      if test_alternative_eip:
        print("Overall SAD gain alternative EIP: ", overall_sad_gain_alternative)
      if test_l2_regularisation:
        for i in range(len(l2_lambdas)):
          print("Overall SAD gain L2 lambda =", str(l2_lambdas[i]), ": ", overall_sad_gain_l2[i])

if __name__ == "__main__":
  # simulation_use_case()
  simulate_eip_looped()
