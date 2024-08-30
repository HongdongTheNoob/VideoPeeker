import os
import sys
import cv2
import numpy as np
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom

import IPM.IPMSim as IPMSim
import FillReferencePatterns

blockSizes = [
  # (4, 4), 
  # (4, 8), 
  # (8, 4), 
  # (4, 16), 
  # (16, 4), 
  # (8, 8), 
  # (8, 16), 
  # (16, 8), 
  (16, 16), 
  # (16, 32), 
  # (32, 16), 
  # (32, 32),
]

for blockSize in blockSizes:
  blockWidth, blockHeight = blockSize
  RL = 1
  
  saveFolder = f'./IPM/IpmVisualisation/{blockWidth}x{blockHeight}'
  if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)
    
  for invocationType in FillReferencePatterns.invocationTypes:
    if RL == 1:
      if invocationType in ['top_bar0', 'top_bar1', 'left_bar0', 'left_bar1']:
        continue
    
    print(f'Simulating: {blockWidth}x{blockHeight}', invocationType)
    blockWithReferenceSamples = np.full((blockHeight * 2 + RL, blockWidth * 2 + RL), 128)
    blockWithReferenceSamples = FillReferencePatterns.FillReferenceSamples(blockWithReferenceSamples, blockSize, invocationType)

    for modeId in range(2, 67):
      predictionBlock = IPMSim.PredIntraAngular(blockWithReferenceSamples, blockSize, modeId)
      blockWithReferenceSamples[RL:RL+blockHeight, RL:RL+blockWidth] = predictionBlock

