from collections import defaultdict
import re
import os
import math
import pandas as pd

class BlockStats:
  def __init__(self):
    self.frame_number = -1
    self.block_dimensions = ()
    self.lines = []
    self.info = defaultdict(int)

  def clear(self):
    self.frame_number = -1
    self.block_dimensions = ()
    self.lines = []
    self.info = defaultdict(int)

  def is_same_block(self, frame_number, dimensions):
    if self.frame_number != frame_number:
      return False

    if dimensions[2] == 0 and dimensions[3] == 0:
      return True
    
    return self.block_dimensions == dimensions

  def pixel_count(self):
    return self.block_dimensions[2] * self.block_dimensions[3]
  
  def add_info(self, line):
    self.lines.append(line)
    pattern = r'(\w+)\s*=\s*(\d+)'
    matches = re.findall(pattern, line)
    for match in matches:
        word = match[0]
        number = int(match[1])
        self.info[word] = number

  def print_info(self):
    print("frame number: ", self.frame_number)
    print("block dimensions: ", self.block_dimensions)
    for key, value in self.info.items():
      print(key, value)

class GlobalStatsContainer:
  def __init__(self):
    self.luma_intra_mode_counter = defaultdict(lambda: [0, 0])
    self.chroma_intra_mode_counter = defaultdict(lambda: [0, 0]) 

def is_in_block(location, dimensions):
  if (location[0] < dimensions[0]) or (location[0] >= dimensions[0] + dimensions[2]):
    return False
  if (location[1] < dimensions[1]) or (location[1] >= dimensions[1] + dimensions[3]):
    return False
  return True

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

def scan_stats(file_path, frame_range = []):
  with open(file_path, "r") as bms_stats_file:
    block_is_located = False
    current_block_stats = BlockStats()
    global_container = GlobalStatsContainer()

    for line in bms_stats_file:
      line = line.rstrip()

      if not line.startswith('BlockStat'): # ignore non-info lines
        continue

      # match frame number
      match = re.search(r"POC (\d+)", line)
      if match is None:
        continue 
      frame_number = int(match.group(1))
      if len(frame_range) > 0 and (frame_number not in frame_range):
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
    
      if current_block_stats.is_same_block(frame_number, block_dimensions):
        current_block_stats.add_info(line)
      else:
        if len(current_block_stats.block_dimensions) == 0: # initialise
          current_block_stats.frame_number = frame_number
          current_block_stats.block_dimensions = block_dimensions
          current_block_stats.add_info(line)
        else: # process this stats and re-initialise
          get_luma_or_chroma_intra_mode(global_container, current_block_stats) 
          current_block_stats.clear()
          current_block_stats.frame_number = frame_number
          current_block_stats.block_dimensions = block_dimensions
          current_block_stats.add_info(line)

    if len(current_block_stats.block_dimensions) > 0: # process remaining
      get_luma_or_chroma_intra_mode(global_container, current_block_stats) 

  return global_container

def find_stats(file_path, frame_number, location):
  block_is_located = False
  current_block_stats = BlockStats()
  with open(file_path, "r") as bms_stats_file:
    for line in bms_stats_file:
      line = line.rstrip()

      if not line.startswith('BlockStat'): # ignore non-info lines
        continue

      # match frame number
      match = re.search(r"POC (\d+)", line)
      if match is None:
        continue 
      if frame_number != int(match.group(1)):
        continue

      # match block location
      match = re.search(r"\((\s*\d+),(\s*\d+)\)", line)
      if match is None:
        continue 
      block_x, block_y = int(match.group(1)), int(match.group(2))
      match = re.search(r"\[(\s*\d+)x(\s*\d+)\]", line)
      if match is None:
        continue 
      block_width, block_height = int(match.group(1)), int(match.group(2))
      
      block_dimensions = (block_x, block_y, block_width, block_height)

      if not is_in_block(location, block_dimensions):
        continue

      if current_block_stats.is_same_block(frame_number, block_dimensions):
        current_block_stats.add_info(line)
      else:
        if len(current_block_stats.block_dimensions) == 0: # initialise
          current_block_stats.frame_number = frame_number
          current_block_stats.block_dimensions = block_dimensions
          current_block_stats.add_info(line)

  return current_block_stats
