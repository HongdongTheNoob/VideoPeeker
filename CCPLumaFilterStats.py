from collections import defaultdict
import re
import os
import math
import pandas as pd

def process_file(filename):
  """
  Processes a text file, builds a table with A, B, and extracted_value as keys, counting zeros and non-zeros.

  Args:
    filename: The path to the text file.

  Returns:
    A dictionary where keys are tuples of (A, B, extracted_value) and values are dictionaries with "zeros" and "non_zeros" counts.
  """
  data_block_size = defaultdict(lambda: {"zeros": 0, "non_zeros": 0})
  data_chroma_mode = defaultdict(lambda: {"zeros": 0, "non_zeros": 0})
  block_is_located = False
  chroma_mode = 0
  block_width = 0
  block_height = 0
  block_x = 0
  block_y = 0

  with open(filename, "r") as f:
    for line in f:
      line = line.rstrip()

      if not block_is_located:
        if "Chroma_IntraMode=" not in line:
          continue

        match = re.search(r"\((\s*\d+),(\s*\d+)\)", line)
        if match is None:
          continue 
        block_x, block_y = int(match.group(1)), int(match.group(2))

        match = re.search(r"\[(\s*\d+)x(\s*\d+)\]", line)
        if match is None:
          continue 
        block_width, block_height = int(match.group(1)), int(match.group(2))

        chroma_mode_pos = line.find("Chroma_IntraMode=")
        if chroma_mode_pos == -1:
          continue
        try:
          chroma_mode = int(line[chroma_mode_pos + len("Chroma_IntraMode="):])
        except ValueError:
          print(f"Invalid integer format after 'ccModelFilter=' in line: {line}")
          continue

        block_is_located = True

      else:
        if "ccModelFilter" not in line or not re.search(r"\[(\s*\d+)x(\s*\d+)\]", line) or "ccModelFilter=" not in line:
          continue

        match = re.search(r"\((\s*\d+),(\s*\d+)\)", line)
        if match is None:
          continue 
        X, Y = int(match.group(1)), int(match.group(2))

        match = re.search(r"\[(\s*\d+)x(\s*\d+)\]", line)
        if match is None:
          continue 
        W, H = int(match.group(1)), int(match.group(2))

        if (not W == block_width) or (not H == block_height) or (not X == block_x) or (not Y == block_y):
          block_is_located = False
          continue

        model_filter_pos = line.find("ccModelFilter=")
        if model_filter_pos == -1:
          continue

        block_is_located = False

        try:
          ccModelFilter = int(line[model_filter_pos + len("ccModelFilter="):])
        except ValueError:
          print(f"Invalid integer format after 'ccModelFilter=' in line: {line}")
          continue

        key = (W*H, W, H)

        if ccModelFilter == 0:
          data_block_size[key]["zeros"] += 1
          data_block_size[(0, 0, 0)]["zeros"] += 1
        else:
          data_block_size[key]["non_zeros"] += 1
          data_block_size[(0, 0, 0)]["non_zeros"] += 1

        key = chroma_mode
        if ccModelFilter == 0:
          data_chroma_mode[key]["zeros"] += 1
        else:
          data_chroma_mode[key]["non_zeros"] += 1

  return data_block_size, data_chroma_mode

qps = [22, 27, 32, 37]
use_rate = [0 for _ in range(4)]
build = "ECM11_LF0-LM"
video_class = "D"
# seq = "FourPeople"
sequences = {
  "B": {"MarketPlace", "RitualDance", "Cactus", "BasketballDrive", "BQTerrace"},
  "C": {"BasketballDrill", "BQMall", "PartyScene", "RaceHorsesC"}, 
  "D": {"BasketballPass", "BlowingBubbles", "BQSquare", "RaceHorses"},
  "E": {"FourPeople", "Johnny", "KristenAndSara"}
}

for seq in sequences[video_class]:
  with open("./block-stats/" + build + "/" + video_class + "-" + seq  + ".txt", 'w') as stats_file:
    for q in range(4):
      qp = qps[q]
      file_name = os.path.join("D:/Data/Bitstream", build, "Class" + video_class, seq, "str-" + seq + "-AI-" + str(qp) + ".vtmbmsstats")

      table = [["".ljust(8) for _ in range(7)] for _ in range(7)]

      for i in range(1, 7):
        table[0][i] = ("W=" + str(2 << i)).ljust(8)
        table[i][0] = ("H=" + str(2 << i)).ljust(8)

      # Line by line process
      data_block_size, data_chroma_mode = process_file(file_name)

      # Use rate vs block size table
      for key, value in sorted(data_block_size.items()):
        area, W, H = key
        rate = value['non_zeros']/(value['zeros']+value['non_zeros'])
        stats_file.write(f"({W}x{H}), zeros: {value['zeros']}, non-zeros: {value['non_zeros']}, use rate: {rate:.3f}" + "\n")

        if W > 0 and H > 0:
          table[int(H).bit_length() - 2][int(W).bit_length() - 2] = str(round(rate, 3)).ljust(8)
        else:
          use_rate[q] = round(rate, 3)

      stats_file.write(f"QP={qp}" + "\n")
      for row in table:
        stats_file.write(f"{''.join([value for value in row])}" + "\n")
      stats_file.write("\n")

      # Use rate versus chroma mode
      for key, value in sorted(data_chroma_mode.items()):
        chroma_mode = key
        rate = value['non_zeros']/(value['zeros']+value['non_zeros'])
        stats_file.write(f"{chroma_mode}, zeros: {value['zeros']}, non-zeros: {value['non_zeros']}, use rate: {rate:.3f}" + "\n")
      stats_file.write("\n")

    stats_file.write("Overall use rates\n")
    stats_file.write(' '.join([f'{value:.3f}' for value in use_rate]) + "\n")
    