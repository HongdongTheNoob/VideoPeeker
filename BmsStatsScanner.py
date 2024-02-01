from collections import defaultdict
import re
import os
import math
import pandas as pd

class BlockStats:
  def __init__(self):
    self.block_dimensions = ()
    self.lines = []

  def clear(self):
    self.block_dimensions = ()
    self.lines = []

  def is_same_block(self, dimensions):
    if dimensions[2] == 0 and dimensions[3] == 0:
      return True
    
    return self.block_dimensions == dimensions
  
class GlobalStatsContainer:
  def __init__(self):
    self.chromaIntraModeCounter = defaultdict(int) 

def process_block_stats(global_container, block_stats):
  for line in block_stats.lines:
    # find chroma intra mode
    model_filter_pos = line.find("Chroma_IntraMode=")
    if model_filter_pos == -1:
      continue
    try:
      chroma_intra_mode = int(line[model_filter_pos + len("Chroma_IntraMode="):])
    except ValueError:
      print(f"Invalid integer format after 'Chroma_IntraMode=' in line: {line}")
      continue

    global_container.chromaIntraModeCounter[chroma_intra_mode] += 1

def bms_stats_scan(file_path):
  with open(file_path, "r") as bms_stats_file:
    block_is_located = False
    current_block_stats = BlockStats()
    global_container = GlobalStatsContainer()

    for line in bms_stats_file:
      line = line.rstrip()

      if not line.startswith('BlockStat'): # ignore non-info lines
        continue

      match = re.search(r"\((\s*\d+),(\s*\d+)\)", line)
      if match is None:
        continue 
      block_x, block_y = int(match.group(1)), int(match.group(2))
      match = re.search(r"\[(\s*\d+)x(\s*\d+)\]", line)
      if match is None:
        continue 
      block_width, block_height = int(match.group(1)), int(match.group(2))
      
      block_dimensions = (block_x, block_y, block_width, block_height)
    
      if current_block_stats.is_same_block(block_dimensions):
        current_block_stats.lines.append(line)
      else:
        if len(current_block_stats.block_dimensions) == 0: # initialise
          current_block_stats.block_dimensions = block_dimensions
          current_block_stats.lines.append(line)
        else: # process this stats and re-initialise
          process_block_stats(global_container, current_block_stats) 
          current_block_stats.clear()
          current_block_stats.block_dimensions = block_dimensions
          current_block_stats.lines.append(line)

    if len(current_block_stats.block_dimensions) > 0: # process remaining
      process_block_stats(global_container, current_block_stats) 

  for key, value in sorted(global_container.chromaIntraModeCounter.items()):
    print("mode: ", key, " count: ", value)
