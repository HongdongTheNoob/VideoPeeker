import os
import sys
import cv2
import numpy as np
import json
import csv
import EIP.EIPSim as EIPSim
import VideoDataset
import BmsStatsScanner
import ResultVisualisation
from PIL import Image, ImageDraw, ImageFont

import GetBlock
import Filters
from ClassVideoInfo import VideoInformation

def replace_extension(file_path, new_extension):
    base_path, old_extension = os.path.splitext(file_path)
    new_file_path = base_path + new_extension
    return new_file_path

def simulation_use_case():
  video_path = "D:/Data/xcy_test/ClassC/BasketballDrill_832x480_50.yuv"
  my_video = VideoInformation(video_path, 832, 480, 8)
  dimensions = (192, 100, 8, 4)
  position_string = '_(' + str(dimensions[0]) + ',' + str(dimensions[1]) + ')_' + str(dimensions[2]) + 'x' + str(dimensions[3])

  rec_video_path = "/Data/Bitstream/ECM12/ClassC/BasketballDrill/str-BasketballDrill-AI-22_1280x720_10bits.yuv"
  rec_video = VideoInformation(rec_video_path, 832, 480, 10)

  y_block, _ = GetBlock.get_block(my_video, 0, dimensions, 'y', 6)
  _, y_template = GetBlock.get_block(rec_video, 0, dimensions, 'y', 6)
  GetBlock.print_block(replace_extension(video_path, '_y' + position_string + '.png'), y_block, bit_depth = my_video.bit_depth, pixel_zoom = 8)
  GetBlock.print_block(replace_extension(video_path, '_y_template' + position_string + '_6.png'), y_template, bit_depth = rec_video.bit_depth, pixel_zoom = 8)

  predicted_block, sad, coeffs = EIPSim.simulate_eip(my_video, 0, dimensions, 6, reconstructed_video = rec_video)
  GetBlock.print_block(replace_extension(video_path, '_y_EIP' + position_string + '.png'), predicted_block, my_video.bit_depth, 8)
  print("EIP coeffs:", coeffs.reshape((1, -1)))
  print("Sad: ", sad)

  predicted_block, sad, coeffs = EIPSim.simulate_eip(my_video, 0, dimensions, 6, reconstructed_video = rec_video, l2_regularisation = 128.0)
  GetBlock.print_block(replace_extension(video_path, '_y_EIP_l2' + position_string + '.png'), predicted_block, my_video.bit_depth, 8)
  print("EIP-L2 coeffs:", coeffs.reshape((1, -1)))
  print("Sad: ", sad)

  # predicted_block, sad, coeffs = EIPSim.simulate_alternative_eip(my_video, 0, dimensions, 6)
  # cv2.imwrite(replace_extension(file_path, '_y_Custom_EIP' + position_string + '.png'), predicted_block)
  # print("Custom EIP coeffs:", coeffs.reshape((1, -1)))
  # print("Sad: ", sad)

def simulate_eip_looped():
  load_block_stats = True
  print_blocks = False

  video_class = "C"
  qps = ["22", "27", "32", "37"] if not print_blocks else ["22"]

  test_l2_regularisation = True
  l2_lambdas = [16, 32, 64, 128, 256, 512, 1024] if not print_blocks else [32, 128, 512]
  test_alternative_eip = False

  if print_blocks:
    if not os.path.exists("./Tests/EIP_Sim/Visualisation"):
      os.mkdir("./Tests/EIP_Sim/Visualisation")
    if not os.path.exists("./Tests/EIP_Sim/Visualisation/Class" + video_class):
      os.mkdir("./Tests/EIP_Sim/Visualisation/Class" + video_class)

  eip_test_log_csv = open("./Tests/EIP_Sim/EIP_stats_class" + video_class + ".csv", mode = 'w', newline = '')
  eip_test_log_csv_writer = csv.writer(eip_test_log_csv)

  for i in range(len(VideoDataset.video_sequences[video_class])):
    sequence = VideoDataset.video_sequences[video_class][i]
    file_path = VideoDataset.video_file_names[video_class][i]
    video_width, video_height, video_bitdepth = VideoDataset.video_width_height_bitdepth[video_class][i]
    test_video = VideoInformation(file_path, video_width, video_height, video_bitdepth)

    video_folder = os.path.dirname(file_path)

    for qp in qps:
      reconstructed_video_files = VideoDataset.find_reconstructed_video_file(os.path.join(VideoDataset.bms_file_base_path, 'Class' + video_class), sequence, "AI", qp)
      if not reconstructed_video_files:
        continue
      rec_video = VideoInformation(reconstructed_video_files[0], video_width, video_height, 10) # reconstructed video is always 10 bit

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
      sad_eip_l2 = [0 for _ in l2_lambdas]

      # Stats
      cod = []
      skewness = []
      kurtosis = []
      outlier_ratio = []
      mad_change_l2 = [[] for _ in l2_lambdas]

      eip_test_log_file = open("./Tests/EIP_Sim/" + sequence + "-" + qp + ".txt", "w")
      for block in all_blocks:
        frame, dimensions = block
        if dimensions[0] == 0 and dimensions[1] == 0:
          continue

        if dimensions[2] * dimensions[3] <= 16:
          continue

        position_string = '_frame_' + str(frame) + '_(' + str(dimensions[0]) + ',' + str(dimensions[1]) + ')_' + str(dimensions[2]) + 'x' + str(dimensions[3])
        # eip_test_log_file.write("frame " + str(frame) + " " + str(dimensions) + "\n")

        if print_blocks:
          rec_block, _ = GetBlock.get_block(rec_video, frame, dimensions, 'y', 0)
          if test_video.bit_depth == 8:
            rec_block = (rec_block + 2) // 4
          print_block_prefix = os.path.join("./Tests/EIP_Sim/Visualisation/Class" + video_class, sequence + "_" + qp + position_string + "_")

        if print_blocks:
          GetBlock.print_block(print_block_prefix + "input.png", rec_block, test_video.bit_depth, 8)

        predicted_block, sad_eip, _, template_stats = EIPSim.simulate_eip(test_video, 0, dimensions, 6, reconstructed_video = rec_video, evaluate_on_template = True)
        # eip_test_log_file.write(" ".join(["SAD EIP: ", str(sad_eip), "\n"]))
        total_sad_eip += sad_eip
        if print_blocks:
          GetBlock.print_block(print_block_prefix + "rec_eip.png", predicted_block, test_video.bit_depth, 8)
          best_mode = "eip"
          best_mode_sad = sad_eip

        if test_l2_regularisation:
          for i in range(len(l2_lambdas)):
            predicted_block, sad_eip_l2[i], _, _ = EIPSim.simulate_eip(test_video, 0, dimensions, 6, reconstructed_video = rec_video, l2_regularisation = l2_lambdas[i])
            # eip_test_log_file.write(" ".join(["SAD EIP-L2-" + str(l2_lambdas[i]) + ": ", str(sad_eip_l2), " change ", str(sad_eip_l2 - sad_eip), "\n"]))
            overall_sad_change_l2[i] += sad_eip_l2[i] - sad_eip
            overall_sad_gain_l2[i] += max(0, sad_eip - sad_eip_l2[i])
            if print_blocks:
              GetBlock.print_block(print_block_prefix + f"rec_eip-l2-{l2_lambdas[i]}.png", predicted_block, test_video.bit_depth, 8)
              if sad_eip_l2[i] < best_mode_sad:
                best_mode = f"l2-{l2_lambdas[i]}"
                best_mode_sad = sad_eip_l2[i]

        if test_alternative_eip:
          predicted_block, sad_eip_alternative, _ = EIPSim.simulate_alternative_eip(test_video, 0, dimensions, 6)
          # eip_test_log_file.write(" ".join(["SAD EIP-Alternative: ", str(sad_eip_l2), " change ", str(sad_eip_alternative - sad_eip), "\n"]))
          overall_sad_change_alternative += sad_eip_alternative - sad_eip
          overall_sad_gain_alternative += max(0, sad_eip - sad_eip_alternative)
          if print_blocks:
            GetBlock.print_block(print_block_prefix + "rec_eip-alt.png", predicted_block, test_video.bit_depth, 8)
            if sad_eip_alternative < best_mode_sad:
              best_mode = "alternative"
              best_mode_sad = sad_eip_alternative

        if print_blocks:
          best_mode_image = ResultVisualisation.text_to_image(best_mode)
          best_mode_image.save(print_block_prefix + "best.png")

        pixel_count += dimensions[2] * dimensions[3]
        
        # write log per block
        info_list = []
        block_pixel_count = dimensions[2] * dimensions[3]
        info_list.append(str(block_pixel_count))

        info_list.append(str(template_stats[2])) # cod
        cod.append(template_stats[2])

        info_list.append(str(template_stats[3])) # skewness
        skewness.append(template_stats[3])

        info_list.append(str(template_stats[4])) # kurtosis
        kurtosis.append(template_stats[4])

        info_list.append(str(template_stats[5])) # outlier ratio
        outlier_ratio.append(template_stats[5])

        if test_l2_regularisation:
          for i in range(len(l2_lambdas)):
            info_list.append(str((sad_eip_l2[i] - sad_eip) / block_pixel_count))
            mad_change_l2[i].append((sad_eip_l2[i] - sad_eip) / block_pixel_count)
        if test_alternative_eip:
          info_list.append(str((sad_eip_alternative - sad_eip) / block_pixel_count))

        info_list.append("\n")
        eip_test_log_file.write(",".join(info_list))
      

      # summary
      print(sequence, qp)
      print("Pixel count: ", pixel_count)
      print("EIP total SAD: ", total_sad_eip)
      if test_l2_regularisation:
        for i in range(len(l2_lambdas)):
          print("Overall SAD change L2 lambda =", str(l2_lambdas[i]), ": ", overall_sad_change_l2[i])
        for i in range(len(l2_lambdas)):
          print("Overall SAD gain L2 lambda =", str(l2_lambdas[i]), ": ", overall_sad_gain_l2[i])
      if test_alternative_eip:
        print("Overall SAD change alternative EIP: ", overall_sad_change_alternative)
        print("Overall SAD gain alternative EIP: ", overall_sad_gain_alternative)

      # correlation/regression
      # cod
      if test_l2_regularisation:
        for i in range(len(l2_lambdas)):
          coeffs = np.polyfit(cod, mad_change_l2[i], 1)
          print(f"L2-{l2_lambdas[i]}:", "cod", coeffs)
        for i in range(len(l2_lambdas)):
          coeffs = np.polyfit([abs(i) for i in skewness], mad_change_l2[i], 1)
          print(f"L2-{l2_lambdas[i]}:", "magnitude of skewness", coeffs)
        for i in range(len(l2_lambdas)):
          coeffs = np.polyfit(kurtosis, mad_change_l2[i], 1)
          print(f"L2-{l2_lambdas[i]}:", "kurtosis", coeffs)
        for i in range(len(l2_lambdas)):
          coeffs = np.polyfit(outlier_ratio, mad_change_l2[i], 1)
          print(f"L2-{l2_lambdas[i]}:", "outlier ratio", coeffs)

      eip_test_log_csv_writer.writerow([sequence, qp])
      eip_test_log_csv_writer.writerow(["Method", "SAD change", "SAD change percentage", "SAD gain", "SAD gain percentage"])
      if test_alternative_eip:
        eip_test_log_csv_writer.writerow(["EIP Alternative: ", str(overall_sad_change_alternative), str(overall_sad_change_alternative/total_sad_eip), str(overall_sad_gain_alternative), str(overall_sad_gain_alternative/total_sad_eip)])
      if test_l2_regularisation:
        for i in range(len(l2_lambdas)):
          eip_test_log_csv_writer.writerow([f"L2 lambda ={l2_lambdas[i]}", str(overall_sad_change_l2[i]), str(overall_sad_change_l2[i]/total_sad_eip), str(overall_sad_gain_l2[i]), str(overall_sad_gain_l2[i]/total_sad_eip)])

if __name__ == "__main__":
  # simulation_use_case()
  simulate_eip_looped()
