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

from scipy.linalg import hadamard

blockSize = (16, 16)
invocationType = 'zebra2'
blockWidth, blockHeight = blockSize
RL = 1

saveFolder = f'./IPM/HadamardMix'
if not os.path.exists(saveFolder):
  os.mkdir(saveFolder)
  
blockWithReferenceSamples = np.full((blockHeight * 8 + RL, blockWidth * 8 + RL), 128)
blockWithReferenceSamples = FillReferencePatterns.FillReferenceSamples(blockWithReferenceSamples, blockSize, invocationType)

modes = [42, 58]
predictionBlocks = []
hadamardBlocks = []
H = hadamard(blockWidth)

w0 = np.zeros(blockSize)
for y in range(blockHeight):
  for x in range(blockWidth):
    b = y / blockHeight + x / blockWidth
    w0[y, x] = 1.0 if b >= 1.5 else (0 if b < 0.5 else 0.5)
w1 = 1.0 - w0

for i in range(len(modes)):
  predictionBlocks.append(IPMSim.PredIntraAngular(blockWithReferenceSamples, blockSize, modes[i]))
  hadamardBlocks.append(H @ predictionBlocks[i] @ H.T)

fusionHadamard = np.multiply(hadamardBlocks[0], w0) + np.multiply(hadamardBlocks[1], w1)
hadamardFusionBlock = (H @ fusionHadamard @ H.T) / (blockHeight * blockWidth)
blockWithReferenceSamples[RL:RL+blockHeight, RL:RL+blockWidth] = hadamardFusionBlock
blockWithReferenceSamplesToSave = blockWithReferenceSamples[0:2*blockHeight+RL, 0:2*blockWidth+RL]
plt.imshow(blockWithReferenceSamplesToSave, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
rect = Rectangle((RL-0.5, RL-0.5), blockWidth, blockHeight, linewidth=RL, edgecolor='black', facecolor='none')
plt.gca().add_patch(rect)
fileName = f'{saveFolder}/fusion_lf_{modes[0]}_hf_{modes[1]}.png'
plt.savefig(fileName, dpi=300, bbox_inches='tight')
plt.close()

fusionBlock = 0.5 * predictionBlocks[0] + 0.5 * predictionBlocks[1]
blockWithReferenceSamples[RL:RL+blockHeight, RL:RL+blockWidth] = fusionBlock
blockWithReferenceSamplesToSave = blockWithReferenceSamples[0:2*blockHeight+RL, 0:2*blockWidth+RL]
plt.imshow(blockWithReferenceSamplesToSave, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
rect = Rectangle((RL-0.5, RL-0.5), blockWidth, blockHeight, linewidth=RL, edgecolor='black', facecolor='none')
plt.gca().add_patch(rect)
fileName = f'{saveFolder}/fusion_0.5_{modes[0]}_{modes[1]}.png'
plt.savefig(fileName, dpi=300, bbox_inches='tight')
plt.close()
