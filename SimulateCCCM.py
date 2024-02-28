import os
import cv2
import numpy as np
import json
import CCP.CCCMSim as CCCMSim
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
  dimensions = (64, 64, 64, 64)
  position_string = '_(' + str(dimensions[0]) + ',' + str(dimensions[1]) + ')_' + str(dimensions[2]) + 'x' + str(dimensions[3])

  y_block, y_template = GetBlock.get_block(my_video, 0, dimensions, 'y', 12)
  cv2.imwrite(replace_extension(file_path, '_y' + position_string + '.png'), np.kron(y_block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_y_template' + position_string + '_12.png'), np.kron(y_template, np.ones((8, 8))))

  cb_block, cb_template = GetBlock.get_block(my_video, 0, dimensions, 'cb', 12)
  cv2.imwrite(replace_extension(file_path, '_cb' + position_string + '.png'), np.kron(cb_block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_cb_template' + position_string + '_12.png'), np.kron(cb_template, np.ones((8, 8))))

  cr_block, cr_template = GetBlock.get_block(my_video, 0, dimensions, 'cr', 12)
  cv2.imwrite(replace_extension(file_path, '_cr' + position_string + '.png'), np.kron(cr_block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_cr_template' + position_string + '_12.png'), np.kron(cr_template, np.ones((8, 8))))

  predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs = CCCMSim.simulate_cccm(my_video, 0, dimensions, 6)
  x_cb, x_cr = coeffs
  cv2.imwrite(replace_extension(file_path, '_cb_predicted_cccm' + position_string + '.png'), np.kron(predicted_cb_block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_cr_predicted_cccm' + position_string + '.png'), np.kron(predicted_cr_block, np.ones((8, 8))))
  print("SADs CCCM: ", sad_cb, sad_cr)
  print("Cb coeffs: ", x_cb.reshape(1, -1))
  print("Cr coeffs: ", x_cr.reshape(1, -1))

  predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs = CCCMSim.simulate_cccm(my_video, 0, dimensions, 6, l2_regularisation=1000.0)
  x_cb, x_cr = coeffs
  cv2.imwrite(replace_extension(file_path, '_cb_predicted_cccm_l2' + position_string + '.png'), np.kron(predicted_cb_block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_cr_predicted_cccm_l2' + position_string + '.png'), np.kron(predicted_cr_block, np.ones((8, 8))))
  print("SADs CCCM-L2: ", sad_cb, sad_cr)
  print("Cb coeffs: ", x_cb.reshape(1, -1))
  print("Cr coeffs: ", x_cr.reshape(1, -1))

  predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs, mad = CCCMSim.simulate_mm_cccm(my_video, 0, dimensions, 6, evaluate_on_template = True)
  x_cb0, x_cb1, x_cr0, x_cr1 = coeffs
  mad_cb, mad_cr, mad_cb_template, mad_cr_template = mad
  cv2.imwrite(replace_extension(file_path, '_cb_predicted_mmlm' + position_string + '.png'), np.kron(predicted_cb_block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_cr_predicted_mmlm' + position_string + '.png'), np.kron(predicted_cr_block, np.ones((8, 8))))
  print("SADs MM-CCCM: ", sad_cb, sad_cr)
  print("MADs for block and template: ", mad)
  print("Cb coeffs: ", x_cb0.reshape(1, -1), x_cb1.reshape(1, -1))
  print("Cr coeffs: ", x_cr0.reshape(1, -1), x_cr1.reshape(1, -1))
  # lbccp
  lbccp_kernel = np.array([[1, 1, 1], [1, 8, 1], [1, 1, 1]]) / 16
  filtered_cb_block = Filters.apply_filter(my_video, 0, dimensions, 'cb', predicted_cb_block, lbccp_kernel)
  filtered_cr_block = Filters.apply_filter(my_video, 0, dimensions, 'cr', predicted_cr_block, lbccp_kernel)
  sad_cb = np.sum(np.abs(filtered_cb_block.astype("int32") - cb_block))
  sad_cr = np.sum(np.abs(filtered_cr_block.astype("int32") - cr_block))
  cv2.imwrite(replace_extension(file_path, '_cb_predicted_mmlm_lbccp' + position_string + '.png'), np.kron(predicted_cb_block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_cr_predicted_mmlm_lbccp' + position_string + '.png'), np.kron(predicted_cr_block, np.ones((8, 8))))
  print("SADs MM-CCCM with LBCCP: ", sad_cb, sad_cr)
  print("Cb coeffs: ", x_cb0.reshape(1, -1), x_cb1.reshape(1, -1))
  print("Cr coeffs: ", x_cr0.reshape(1, -1), x_cr1.reshape(1, -1))

  predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs, mad = CCCMSim.simulate_mm_cccm(my_video, 0, dimensions, 6, l2_regularisation=500.0, evaluate_on_template = True)
  x_cb0, x_cb1, x_cr0, x_cr1 = coeffs
  mad_cb, mad_cr, mad_cb_template, mad_cr_template = mad
  cv2.imwrite(replace_extension(file_path, '_cb_predicted_mmlm_l2' + position_string + '.png'), np.kron(predicted_cb_block, np.ones((8, 8))))
  cv2.imwrite(replace_extension(file_path, '_cr_predicted_mmlm_l2' + position_string + '.png'), np.kron(predicted_cr_block, np.ones((8, 8))))
  print("SADs MM-CCCM-L2: ", sad_cb, sad_cr)
  print("MADs for block and template: ", mad)
  print("Cb coeffs: ", x_cb0.reshape(1, -1), x_cb1.reshape(1, -1))
  print("Cr coeffs: ", x_cr0.reshape(1, -1), x_cr1.reshape(1, -1))

def simulate_mmlm_looped():
  video_class = "E"
  qps = ["22"]

  test_lbccp = True
  test_l2_regularisation = True
  l2_lambdas = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
  test_soft_mmlm = True

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
        all_blocks = BmsStatsScanner.collect_blocks(bms_files[0], frame_range = range(1), target_string = "Chroma_IntraMode=68")
        with open("./Tests/MMLM_Sim/" + sequence + "-" + qp + ".json", "w") as file:
          json.dump(all_blocks, file)   
      else: 
        with open("./Tests/MMLM_Sim/" + sequence + "-" + qp + ".json", "r") as file:
          all_blocks = json.load(file)    

      # Loop
      pixel_count = 0
      total_sad_mmlm = 0
      overall_sad_change_lbccp = 0
      overall_sad_gain_lbccp = 0
      overall_sad_change_l2 = [0 for _ in l2_lambdas]
      overall_sad_gain_l2 = [0 for _ in l2_lambdas]
      overall_sad_change_soft = 0
      overall_sad_gain_soft = 0

      mmlm_test_log_file = open("./Tests/MMLM_Sim/" + sequence + "-" + qp + ".txt", "w")
      for block in all_blocks:
        frame, dimensions = block
        if dimensions[0] == 0 and dimensions[1] == 0:
          continue

        position_string = '_frame_' + str(frame) + '_(' + str(dimensions[0]) + ',' + str(dimensions[1]) + ')_' + str(dimensions[2]) + 'x' + str(dimensions[3])
        mmlm_test_log_file.write("frame " + str(frame) + " " + str(dimensions) + "\n")

        cb_block, _ = GetBlock.get_block(my_video, frame, dimensions, 'cb', 0)
        cr_block, _ = GetBlock.get_block(my_video, frame, dimensions, 'cr', 0)

        predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs, mad = CCCMSim.simulate_mm_cccm(my_video, 0, dimensions, 6)
        sad_mmlm = sad_cb + sad_cr
        mmlm_test_log_file.write(" ".join(["SADs MM-CCCM: ", str(sad_cb), str(sad_cr), "\n"]))
        pixel_count += dimensions[2] * dimensions[3] // 4
        total_sad_mmlm += sad_mmlm

        # lbccp
        if test_lbccp:
          lbccp_kernel = np.array([[1, 1, 1], [1, 8, 1], [1, 1, 1]]) / 16
          filtered_cb_block = Filters.apply_filter(my_video, frame, dimensions, 'cb', predicted_cb_block, lbccp_kernel)
          filtered_cr_block = Filters.apply_filter(my_video, frame, dimensions, 'cr', predicted_cr_block, lbccp_kernel)
          sad_cb = np.sum(np.abs(filtered_cb_block.astype("int32") - cb_block))
          sad_cr = np.sum(np.abs(filtered_cr_block.astype("int32") - cr_block))
          sad_mmlm_lbccp = sad_cb + sad_cr
          mmlm_test_log_file.write(" ".join(["SADs MM-CCCM with LBCCP: ", str(sad_cb), str(sad_cr), " change ", str(sad_mmlm_lbccp - sad_mmlm), "\n"]))
          overall_sad_change_lbccp += sad_mmlm_lbccp - sad_mmlm
          overall_sad_gain_lbccp += max(0, sad_mmlm - sad_mmlm_lbccp)

        if test_l2_regularisation:
          for i in range(len(l2_lambdas)):
            predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs, mad = CCCMSim.simulate_mm_cccm(my_video, 0, dimensions, 6, l2_regularisation = l2_lambdas[i])
            sad_mmlm_l2 = sad_cb + sad_cr
            mmlm_test_log_file.write(" ".join(["SADs MM-CCCM-L2-100: ", str(sad_cb), str(sad_cr), " change ", str(sad_mmlm_l2 - sad_mmlm), "\n"]))
            overall_sad_change_l2[i] += sad_mmlm_l2 - sad_mmlm
            overall_sad_gain_l2[i] += max(0, sad_mmlm - sad_mmlm_l2)

        if test_soft_mmlm:
          predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, coeffs = CCCMSim.simulate_soft_classified_mm_cccm(my_video, 0, dimensions, 6)
          sad_new_mmlm = sad_cb + sad_cr
          mmlm_test_log_file.write(" ".join(["SADs Soft MM-CCCM: ", str(sad_cb), str(sad_cr), " change ", str(sad_new_mmlm - sad_mmlm), "\n"]))
          overall_sad_change_soft += sad_new_mmlm - sad_mmlm
          overall_sad_gain_soft += max(0, sad_mmlm - sad_new_mmlm)

      
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


def simulate_cccm_looped():
  video_class = "D"
  qps = ["22", "27", "32", "37"]

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

      all_blocks = BmsStatsScanner.collect_blocks(bms_files[0], frame_range = range(10), target_string = "Chroma_IntraMode=67")
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

        predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, _ = CCCMSim.simulate_cccm(my_video, 0, dimensions, 6)
        sad_cccm = sad_cb + sad_cr
        test_log_file.write(" ".join(["SADs MM-CCCM: ", str(sad_cb), str(sad_cr), "\n"]))

        predicted_cb_block, predicted_cr_block, sad_cb, sad_cr, _ = CCCMSim.simulate_cccm(my_video, 0, dimensions, 6, l2_regularisation = 1000)
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
