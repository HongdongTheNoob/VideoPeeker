import os
import sys
import cv2
import numpy as np
import json
import csv
import CCP.CCCMSim as CCCMSim
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
  video_path = "/Data/xcy_test/ClassE/FourPeople_1280x720_60.yuv"
  test_video = VideoInformation(video_path, 1280, 720, 8)
  dimensions = (64, 96, 64, 32)
  position_string = '_(' + str(dimensions[0]) + ',' + str(dimensions[1]) + ')_' + str(dimensions[2]) + 'x' + str(dimensions[3])

  rec_video_path = "/Data/Bitstream/ECM12/ClassE/FourPeople/str-FourPeople-AI-22_1280x720_10bits.yuv"
  rec_video = VideoInformation(rec_video_path, 1280, 720, 10)

  y_block, y_template = GetBlock.get_block(rec_video, 0, dimensions, 'y', 12)
  GetBlock.print_block(replace_extension(video_path, '_y' + position_string + '.png'), y_block, bit_depth = rec_video.bit_depth, pixel_zoom = 8)
  GetBlock.print_block(replace_extension(video_path, '_y_template' + position_string + '_12.png'), y_template, bit_depth = rec_video.bit_depth, pixel_zoom = 8)

  cb_block, _ = GetBlock.get_block(test_video, 0, dimensions, 'cb', 12)
  _, cb_template = GetBlock.get_block(rec_video, 0, dimensions, 'cb', 12)
  GetBlock.print_block(replace_extension(video_path, '_cb' + position_string + '.png'), cb_block, bit_depth = test_video.bit_depth, pixel_zoom = 8)
  GetBlock.print_block(replace_extension(video_path, '_cb_template' + position_string + '_12.png'), cb_template, bit_depth = rec_video.bit_depth, pixel_zoom = 8)

  cr_block, _ = GetBlock.get_block(test_video, 0, dimensions, 'cr', 12)
  _, cr_template = GetBlock.get_block(rec_video, 0, dimensions, 'cr', 12)
  GetBlock.print_block(replace_extension(video_path, '_cr' + position_string + '.png'), cr_block, bit_depth = test_video.bit_depth, pixel_zoom = 8)
  GetBlock.print_block(replace_extension(video_path, '_cr_template' + position_string + '_12.png'), cr_template, bit_depth = rec_video.bit_depth, pixel_zoom = 8)

  predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs = CCCMSim.simulate_cccm(test_video, 0, dimensions, 6, reconstructed_video = rec_video)
  x_cb, x_cr = coeffs
  GetBlock.print_block(replace_extension(video_path, '_cb_predicted_cccm' + position_string + '.png'), predicted_cb_block, bit_depth = test_video.bit_depth, pixel_zoom = 8)
  GetBlock.print_block(replace_extension(video_path, '_cr_predicted_cccm' + position_string + '.png'), predicted_cr_block, bit_depth = test_video.bit_depth, pixel_zoom = 8)
  print("SADs CCCM: ", sad_cb, sad_cr)
  # print("Cb coeffs: ", x_cb.reshape(1, -1))
  # print("Cr coeffs: ", x_cr.reshape(1, -1))

  predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs = CCCMSim.simulate_cccm(test_video, 0, dimensions, 6, reconstructed_video = rec_video, l2_regularisation=512.0)
  x_cb, x_cr = coeffs
  GetBlock.print_block(replace_extension(video_path, '_cb_predicted_cccm_l2' + position_string + '.png'), predicted_cb_block, bit_depth = test_video.bit_depth, pixel_zoom = 8)
  GetBlock.print_block(replace_extension(video_path, '_cr_predicted_cccm_l2' + position_string + '.png'), predicted_cr_block, bit_depth = test_video.bit_depth, pixel_zoom = 8)
  print("SADs CCCM-L2: ", sad_cb, sad_cr)
  # print("Cb coeffs: ", x_cb.reshape(1, -1))
  # print("Cr coeffs: ", x_cr.reshape(1, -1))

  predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs, template_stats = CCCMSim.simulate_mm_cccm(test_video, 0, dimensions, 6, evaluate_on_template = True, reconstructed_video = rec_video)
  x_cb0, x_cb1, x_cr0, x_cr1 = coeffs
  mad_cb, mad_cr, mad_cb_template, mad_cr_template, cod = template_stats
  GetBlock.print_block(replace_extension(video_path, '_cb_predicted_mmlm' + position_string + '.png'), predicted_cb_block, bit_depth = test_video.bit_depth, pixel_zoom = 8)
  GetBlock.print_block(replace_extension(video_path, '_cr_predicted_mmlm' + position_string + '.png'), predicted_cr_block, bit_depth = test_video.bit_depth, pixel_zoom = 8)
  print("SADs MM-CCCM: ", sad_cb, sad_cr)
  print("MADs and COD: ", template_stats)
  print("Cb coeffs: ", x_cb0.reshape(1, -1)[0], x_cb1.reshape(1, -1)[0])
  print("Cr coeffs: ", x_cr0.reshape(1, -1)[0], x_cr1.reshape(1, -1)[0])

  # lbccp
  # lbccp_kernel = np.array([[1, 1, 1], [1, 8, 1], [1, 1, 1]]) / 16
  # filtered_cb_block = Filters.apply_filter(test_video, 0, dimensions, 'cb', predicted_cb_block, lbccp_kernel)
  # filtered_cr_block = Filters.apply_filter(test_video, 0, dimensions, 'cr', predicted_cr_block, lbccp_kernel)
  # sad_cb = np.sum(np.abs(filtered_cb_block.astype("int32") - cb_block))
  # sad_cr = np.sum(np.abs(filtered_cr_block.astype("int32") - cr_block))
  # GetBlock.print_block(replace_extension(video_path, '_cb_predicted_mmlm_lbccp' + position_string + '.png'), filtered_cb_block, bit_depth = test_video.bit_depth, pixel_zoom = 8)
  # GetBlock.print_block(replace_extension(video_path, '_cr_predicted_mmlm_lbccp' + position_string + '.png'), filtered_cr_block, bit_depth = test_video.bit_depth, pixel_zoom = 8)
  # print("SADs MM-CCCM with LBCCP: ", sad_cb, sad_cr)
  # # print("Cb coeffs: ", x_cb0.reshape(1, -1), x_cb1.reshape(1, -1))
  # # print("Cr coeffs: ", x_cr0.reshape(1, -1), x_cr1.reshape(1, -1))

  # predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs, mad = CCCMSim.simulate_mm_cccm(test_video, 0, dimensions, 6, l2_regularisation=128.0, evaluate_on_template = True, reconstructed_video = rec_video)
  # x_cb0, x_cb1, x_cr0, x_cr1 = coeffs
  # mad_cb, mad_cr, mad_cb_template, mad_cr_template = mad
  # GetBlock.print_block(replace_extension(video_path, '_cb_predicted_mmlm_l2' + position_string + '.png'), predicted_cb_block, bit_depth = test_video.bit_depth, pixel_zoom = 8)
  # GetBlock.print_block(replace_extension(video_path, '_cr_predicted_mmlm_l2' + position_string + '.png'), predicted_cr_block, bit_depth = test_video.bit_depth, pixel_zoom = 8)
  # print("SADs MM-CCCM-L2: ", sad_cb, sad_cr)
  # print("MADs for block and template: ", mad)
  # # print("Cb coeffs: ", x_cb0.reshape(1, -1), x_cb1.reshape(1, -1))
  # # print("Cr coeffs: ", x_cr0.reshape(1, -1), x_cr1.reshape(1, -1))

def simulate_mmlm_looped():
  load_block_stats = True
  print_blocks = True

  video_class = "E"
  qps = ["22", "27", "32", "37"] if not print_blocks else ["22"]

  test_lbccp = True
  test_l2_regularisation = True
  l2_lambdas = [4, 8, 16, 32, 64, 128, 256] if not print_blocks else [8, 32, 128]
  test_soft_mmlm = False

  if print_blocks:
    if not os.path.exists("./Tests/MMLM_Sim/Visualisation"):
      os.mkdir("./Tests/MMLM_Sim/Visualisation")
    if not os.path.exists("./Tests/MMLM_Sim/Visualisation/Class" + video_class):
      os.mkdir("./Tests/MMLM_Sim/Visualisation/Class" + video_class)
    # if not os.path.exists("./Tests/MMLM_Sim/Visualisation/Class" + video_class + "/mm-cccm"):
    #   os.mkdir("./Tests/MMLM_Sim/Visualisation" + video_class + "/mm-cccm")
    # if test_lbccp:
    #   if not os.path.exists("./Tests/MMLM_Sim/Visualisation/Class" + video_class + "/lbccp"):
    #     os.mkdir("./Tests/MMLM_Sim/Visualisation" + video_class + "/lbccp")
    # if test_l2_regularisation:
    #   if not os.path.exists("./Tests/MMLM_Sim/Visualisation/Class" + video_class + "/l2"):
    #     os.mkdir("./Tests/MMLM_Sim/Visualisation" + video_class + "/l2")

  mmlm_test_log_csv = open("./Tests/MMLM_Sim/MMLM_stats_class" + video_class + ".csv", mode = 'w', newline = '')
  mmlm_test_log_csv_writer = csv.writer(mmlm_test_log_csv)

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
        all_blocks = BmsStatsScanner.collect_blocks(bms_files[0], frame_range = range(10), target_string = "IsMmCCCMFull=1")
        with open("./Tests/MMLM_Sim/" + sequence + "-" + qp + ".json", "w") as file:
          json.dump(all_blocks, file)   
      else: 
        with open("./Tests/MMLM_Sim/" + sequence + "-" + qp + ".json", "r") as file:
          all_blocks = json.load(file)    

      print("test video:", file_path)
      print("rec video:", reconstructed_video_files[0])

      # Loop
      pixel_count = 0
      total_sad_mmlm = 0
      overall_sad_change_lbccp = 0
      overall_sad_gain_lbccp = 0
      overall_sad_change_l2 = [0 for _ in l2_lambdas]
      overall_sad_gain_l2 = [0 for _ in l2_lambdas]
      overall_sad_change_soft = 0
      overall_sad_gain_soft = 0
      sad_mmlm_l2 = [0 for _ in l2_lambdas]

      mmlm_test_log_file = open("./Tests/MMLM_Sim/" + sequence + "-" + qp + ".txt", "w")
      for block in all_blocks:
        frame, dimensions = block
        if dimensions[0] == 0 and dimensions[1] == 0:
          continue

        if print_blocks:
          if dimensions[2] * dimensions[3] < 128:
            continue

        position_string = '_frame_' + str(frame) + '_(' + str(dimensions[0]) + ',' + str(dimensions[1]) + ')_' + str(dimensions[2]) + 'x' + str(dimensions[3])
        # mmlm_test_log_file.write("frame " + str(frame) + " " + str(dimensions) + "\n")

        if print_blocks:
          rec_y_block, _ = GetBlock.get_block(rec_video, frame, dimensions, 'y', 0)
          if test_video.bit_depth == 8:
            rec_y_block = (rec_y_block + 2) // 4
          print_block_prefix = os.path.join("./Tests/MMLM_Sim/Visualisation/Class" + video_class, sequence + "_" + qp + position_string + "_")
        
        cb_block, _ = GetBlock.get_block(test_video, frame, dimensions, 'cb', 0)
        cr_block, _ = GetBlock.get_block(test_video, frame, dimensions, 'cr', 0)

        if print_blocks:
          GetBlock.print_block_yuv(print_block_prefix + "input.png", (rec_y_block, cb_block, cr_block), test_video.bit_depth, 8)

        predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs, template_stats = CCCMSim.simulate_mm_cccm(test_video, 0, dimensions, 6, reconstructed_video = rec_video, evaluate_on_template = True)
        sad_mmlm = sad_cb + sad_cr
        # mmlm_test_log_file.write(" ".join(["Template COD", str(template_stats[4])]))
        # mmlm_test_log_file.write(" ".join(["SADs MM-CCCM: ", str(sad_cb), str(sad_cr), "\n"]))
        total_sad_mmlm += sad_mmlm
        if print_blocks:
          GetBlock.print_block_yuv(print_block_prefix + "mm-cccm.png", (rec_y_block, predicted_cb_block, predicted_cr_block), test_video.bit_depth, 8)
          best_mode = "mm-cccm"
          best_mode_sad = sad_cb + sad_cr

        # lbccp
        if test_lbccp:
          lbccp_kernel = np.array([[1, 1, 1], [1, 8, 1], [1, 1, 1]]) / 16
          filtered_cb_block = Filters.apply_filter(test_video, frame, dimensions, 'cb', predicted_cb_block, lbccp_kernel)
          filtered_cr_block = Filters.apply_filter(test_video, frame, dimensions, 'cr', predicted_cr_block, lbccp_kernel)
          sad_cb = np.sum(np.abs(filtered_cb_block.astype("int32") - cb_block))
          sad_cr = np.sum(np.abs(filtered_cr_block.astype("int32") - cr_block))
          sad_mmlm_lbccp = sad_cb + sad_cr
          # mmlm_test_log_file.write(" ".join(["SADs MM-CCCM with LBCCP: ", str(sad_cb), str(sad_cr), " change ", str(sad_mmlm_lbccp - sad_mmlm), "\n"]))
          overall_sad_change_lbccp += sad_mmlm_lbccp - sad_mmlm
          overall_sad_gain_lbccp += max(0, sad_mmlm - sad_mmlm_lbccp)
          if print_blocks:
            GetBlock.print_block_yuv(print_block_prefix + "mm-cccm-lbccp.png", (rec_y_block, filtered_cb_block, filtered_cr_block), test_video.bit_depth, 8)
            if sad_cb + sad_cr < best_mode_sad:
              best_mode = "lbccp"
              best_mode_sad = sad_cb + sad_cr

        if test_l2_regularisation:
          for i in range(len(l2_lambdas)):
            predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs, _ = CCCMSim.simulate_mm_cccm(test_video, 0, dimensions, 6, l2_regularisation = l2_lambdas[i], reconstructed_video = rec_video)
            sad_mmlm_l2[i] = sad_cb + sad_cr
            # mmlm_test_log_file.write(" ".join(["SADs MM-CCCM-L2-100: ", str(sad_cb), str(sad_cr), " change ", str(sad_mmlm_l2[i] - sad_mmlm), "\n"]))
            overall_sad_change_l2[i] += sad_mmlm_l2[i] - sad_mmlm
            overall_sad_gain_l2[i] += max(0, sad_mmlm - sad_mmlm_l2[i])
            if print_blocks:
              GetBlock.print_block_yuv(print_block_prefix + f"mm-cccm-l2-{l2_lambdas[i]}.png", (rec_y_block, predicted_cb_block, predicted_cr_block), test_video.bit_depth, 8)
              if sad_cb + sad_cr < best_mode_sad:
                best_mode = f"l2-{l2_lambdas[i]}"
                best_mode_sad = sad_cb + sad_cr

        if test_soft_mmlm:
          predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs, _ = CCCMSim.simulate_soft_classified_mm_cccm(test_video, 0, dimensions, 6)
          sad_mmlm_soft = sad_cb + sad_cr
          # mmlm_test_log_file.write(" ".join(["SADs Soft MM-CCCM: ", str(sad_cb), str(sad_cr), " change ", str(sad_mmlm_soft - sad_mmlm), "\n"]))
          overall_sad_change_soft += sad_mmlm_soft - sad_mmlm
          overall_sad_gain_soft += max(0, sad_mmlm - sad_mmlm_soft)
          if print_blocks:
            GetBlock.print_block_yuv(print_block_prefix + "mm-cccm-soft.png", (rec_y_block, predicted_cb_block, predicted_cr_block), test_video.bit_depth, 8)
            if sad_cb + sad_cr < best_mode_sad:
              best_mode = "soft"
              best_mode_sad = sad_cb + sad_cr
      
        if print_blocks:
          best_mode_image = ResultVisualisation.text_to_image(best_mode)
          best_mode_image.save(print_block_prefix + "best.png")
          
        pixel_count += dimensions[2] * dimensions[3] // 4
        
        # write log per block
        info_list = []
        block_pixel_count = dimensions[2] * dimensions[3] // 4
        info_list.append(str(block_pixel_count))
        info_list.append(str(template_stats[4]))
        if test_lbccp:
          info_list.append(str((sad_mmlm_lbccp - sad_mmlm) / block_pixel_count))
        if test_l2_regularisation:
          for i in range(len(l2_lambdas)):
            info_list.append(str((sad_mmlm_l2[i] - sad_mmlm) / block_pixel_count))
        if test_soft_mmlm:
          info_list.append(str((sad_mmlm_soft - sad_mmlm) / block_pixel_count))

        info_list.append("\n")
        mmlm_test_log_file.write(",".join(info_list))

      print(sequence, qp)
      print("Pixel count: ", pixel_count)
      print("MMLM total SAD: ", total_sad_mmlm)
      if test_lbccp:
        print("Overall SAD change LBCCP: ", overall_sad_change_lbccp)
        print("Overall SAD gain LBCCP: ", overall_sad_gain_lbccp)
      if test_l2_regularisation:
        for i in range(len(l2_lambdas)):
          print("Overall SAD change L2 lambda =", str(l2_lambdas[i]), ": ", overall_sad_change_l2[i])
        for i in range(len(l2_lambdas)):
          print("Overall SAD gain L2 lambda =", str(l2_lambdas[i]), ": ", overall_sad_gain_l2[i])
      if test_soft_mmlm:
        print("Overall SAD change soft MMLM: ", overall_sad_change_soft)
        print("Overall SAD gain soft MMLM: ", overall_sad_gain_soft)

      mmlm_test_log_csv_writer.writerow([sequence, qp])
      mmlm_test_log_csv_writer.writerow(["Method", "SAD change", "SAD change percentage", "SAD gain", "SAD gain percentage"])
      if test_lbccp:
        mmlm_test_log_csv_writer.writerow(["LBCCP: ", str(overall_sad_change_lbccp), str(overall_sad_change_lbccp/total_sad_mmlm), str(overall_sad_gain_lbccp), str(overall_sad_gain_lbccp/total_sad_mmlm)])
      if test_l2_regularisation:
        for i in range(len(l2_lambdas)):
          mmlm_test_log_csv_writer.writerow([f"L2 lambda ={l2_lambdas[i]}", str(overall_sad_change_l2[i]), str(overall_sad_change_l2[i]/total_sad_mmlm), str(overall_sad_gain_l2[i]), str(overall_sad_gain_l2[i]/total_sad_mmlm)])
      if test_soft_mmlm:
        mmlm_test_log_csv_writer.writerow(["Soft MMLM: ", str(overall_sad_change_soft), str(overall_sad_change_soft/total_sad_mmlm), str(overall_sad_gain_soft), str(overall_sad_gain_soft/total_sad_mmlm)])
      

def simulate_cccm_looped():
  video_class = "D"
  qps = ["22", "27", "32", "37"]

  for i in range(len(VideoDataset.video_sequences[video_class])):
    sequence = VideoDataset.video_sequences[video_class][i]
    file_path = VideoDataset.video_file_names[video_class][i]
    video_width, video_height, video_bitdepth = VideoDataset.video_width_height_bitdepth[video_class][i]
    test_video = VideoInformation(file_path, video_width, video_height, video_bitdepth)

    video_folder = os.path.dirname(file_path)

    for qp in qps:
      bms_files = VideoDataset.find_stats_files(os.path.join(VideoDataset.bms_file_base_path, 'Class' + video_class), sequence, qp)

      if not bms_files:
        continue

      all_blocks = BmsStatsScanner.collect_blocks(bms_files[0], frame_range = range(10), target_string = "IsCCCMFull=1")
      with open("./Tests/CCCM_Sim/" + sequence + "-" + qp + ".json", "w") as file:
        json.dump(all_blocks, file)    

      # Loop
      pixel_count = 0
      total_sad_cccm = 0
      overall_sad_change_l2 = 0

      test_log_file = open("./Tests/CCCM_Sim/" + sequence + "-" + qp + ".txt", "w")
      for block in all_blocks:
        frame, dimensions = block
        position_string = '_frame_' + str(frame) + '_(' + str(dimensions[0]) + ',' + str(dimensions[1]) + ')_' + str(dimensions[2]) + 'x' + str(dimensions[3])
        test_log_file.write("frame " + str(frame) + " " + str(dimensions) + "\n")

        predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, _ = CCCMSim.simulate_cccm(test_video, 0, dimensions, 6)
        sad_cccm = sad_cb + sad_cr
        test_log_file.write(" ".join(["SADs MM-CCCM: ", str(sad_cb), str(sad_cr), "\n"]))

        predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, _ = CCCMSim.simulate_cccm(test_video, 0, dimensions, 6, l2_regularisation = 1000)
        sad_cccm_l2_100 = sad_cb + sad_cr
        test_log_file.write(" ".join(["SADs MM-CCCM-L2-100: ", str(sad_cb), str(sad_cr), " change ", str(sad_cccm_l2_100 - sad_cccm), "\n"]))

        pixel_count += dimensions[2] * dimensions[3] // 4
        total_sad_cccm += sad_cccm
        overall_sad_change_l2 += sad_cccm_l2_100 - sad_cccm
      
      print(sequence, qp)
      print("Pixel count: ", pixel_count)
      print("CCCM total SAD: ", total_sad_cccm)
      print("Overall SAD change L2: ", overall_sad_change_l2)

if __name__ == "__main__":
  # simulation_use_case()
  simulate_mmlm_looped()
  # simulate_cccm_looped()
