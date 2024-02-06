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

  def pixel_count(self):
    return self.block_dimensions[2] * self.block_dimensions[3]
  
class GlobalStatsContainer:
  def __init__(self):
    self.luma_intra_mode_counter = defaultdict(lambda: [0, 0])
    self.chroma_intra_mode_counter = defaultdict(lambda: [0, 0]) 

def get_luma_or_chroma_intra_mode(global_container, block_stats):
  for line in block_stats.lines:
    # find luma/chroma intra mode
    luma_mode_pos = line.find("Luma_IntraMode=")
    chroma_mode_pos = line.find("Chroma_IntraMode=")
    if luma_mode_pos == -1 and chroma_mode_pos == -1:
      continue

    if luma_mode_pos != -1:
      luma_intra_mode = int(line[luma_mode_pos + len("Luma_IntraMode="):])
      global_container.luma_intra_mode_counter[luma_intra_mode][0] += 1
      global_container.luma_intra_mode_counter[luma_intra_mode][1] += block_stats.pixel_count()

    if chroma_mode_pos != -1:
      chroma_intra_mode = int(line[chroma_mode_pos + len("Chroma_IntraMode="):])
      global_container.chroma_intra_mode_counter[chroma_intra_mode][0] += 1
      global_container.chroma_intra_mode_counter[chroma_intra_mode][1] += block_stats.pixel_count()

def scan_stats(file_path):
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
          get_luma_or_chroma_intra_mode(global_container, current_block_stats) 
          current_block_stats.clear()
          current_block_stats.block_dimensions = block_dimensions
          current_block_stats.lines.append(line)

    if len(current_block_stats.block_dimensions) > 0: # process remaining
      get_luma_or_chroma_intra_mode(global_container, current_block_stats) 

  return global_container