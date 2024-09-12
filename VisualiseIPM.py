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
from itertools import chain

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

invocationTypes = [
  # 'top_bar',
  # 'top_zebra2',
  # 'left_bar', 
  # 'left_bar0', 
  # 'left_bar1', 
  # 'left_zebra', 
  # 'left_zebra2', 
  # 'zebra', 
  'zebra2', 
  # 'corner_TL', 
  # 'corner_BL',
  # 'gradient'
]

for blockSize in blockSizes:
  blockWidth, blockHeight = blockSize
  RL = 1
  
  saveFolder = f'./IPM/IpmVisualisation/{blockWidth}x{blockHeight}_PDPC'
  if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)
    
  for invocationType in invocationTypes:
    if RL == 1:
      if invocationType in ['top_bar0', 'top_bar1', 'left_bar0', 'left_bar1']:
        continue
    
    print(f'Simulating: {blockWidth}x{blockHeight}', invocationType)
    blockWithReferenceSamples = np.full((blockHeight * 8 + RL, blockWidth * 8 + RL), 128)
    blockWithReferenceSamples = FillReferencePatterns.FillReferenceSamples(blockWithReferenceSamples, blockSize, invocationType)

    for modeId in chain(range(-14, 0), range(2, 81)):
    # for modeId in range(51, 56):
      print("Angular mode", modeId)
      predictionBlock = IPMSim.PredIntraAngular(blockWithReferenceSamples, blockSize, modeId)
      blockWithReferenceSamples[RL:RL+blockHeight, RL:RL+blockWidth] = predictionBlock

      # Save
      blockWithReferenceSamplesToSave = blockWithReferenceSamples[0:2*blockHeight+RL, 0:2*blockWidth+RL]
      plt.imshow(blockWithReferenceSamplesToSave, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
      rect = Rectangle((RL-0.5, RL-0.5), blockWidth, blockHeight, linewidth=RL, edgecolor='black', facecolor='none')
      plt.gca().add_patch(rect)
      fileName = f'{saveFolder}/{invocationType}_{(modeId+14):02d}_mode{(modeId):02d}.png'
      plt.savefig(fileName, dpi=300, bbox_inches='tight')
      plt.close()
