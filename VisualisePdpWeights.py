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

import MIP.PdpData
import FillReferencePatterns

ROUNDER = np.power(2, 13)
DIVISOR = np.power(2, 14)

# get weights
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
  'zebra', 
  'zebra2', 
  # 'corner_TL', 
  # 'corner_BL'
]

longWeights = [{
  (4, 4): MIP.PdpData.g_weights4x4,
  (4, 8): MIP.PdpData.g_weights4x8,
  (8, 4): MIP.PdpData.g_weights8x4,
  (4, 16): MIP.PdpData.g_weights4x16,
  (16, 4): MIP.PdpData.g_weights16x4,
  (8, 8): MIP.PdpData.g_weights8x8,
  (8, 16): MIP.PdpData.g_weights8x16,
  (16, 8): MIP.PdpData.g_weights16x8,
  (16, 16): MIP.PdpData.g_weights16x16,
  (16, 32): MIP.PdpData.g_weights16x32,
  (32, 16): MIP.PdpData.g_weights32x16,
  (32, 32): MIP.PdpData.g_weights32x32,
}, {
  (4, 4): MIP.PdpData.g_weights4x4,
  (4, 8): MIP.PdpData.g_weights8x4,
  (8, 4): MIP.PdpData.g_weights4x8,
  (4, 16): MIP.PdpData.g_weights16x4,
  (16, 4): MIP.PdpData.g_weights4x16,
  (8, 8): MIP.PdpData.g_weights8x8,
  (8, 16): MIP.PdpData.g_weights16x8,
  (16, 8): MIP.PdpData.g_weights8x16,
  (16, 16): MIP.PdpData.g_weights16x16,
  (16, 32): MIP.PdpData.g_weights32x16,
  (32, 16): MIP.PdpData.g_weights16x32,
  (32, 32): MIP.PdpData.g_weights32x32,
}
]

shortWeights = [{
  (4, 4): MIP.PdpData.g_weightsShort4x4,
  (4, 8): MIP.PdpData.g_weightsShort4x8,
  (8, 4): MIP.PdpData.g_weightsShort8x4,
  (4, 16): MIP.PdpData.g_weightsShort4x16,
  (16, 4): MIP.PdpData.g_weightsShort16x4,
  (8, 8): MIP.PdpData.g_weightsShort8x8,
  (8, 16): MIP.PdpData.g_weightsShort8x16,
  (16, 8): MIP.PdpData.g_weightsShort16x8,
  (16, 16): MIP.PdpData.g_weightsShort16x16,
  (16, 32): MIP.PdpData.g_weightsShort16x32,
  (32, 16): MIP.PdpData.g_weightsShort32x16,
  (32, 32): MIP.PdpData.g_weightsShort32x32,
}, {
  (4, 4): MIP.PdpData.g_weightsShort4x4,
  (4, 8): MIP.PdpData.g_weightsShort8x4,
  (8, 4): MIP.PdpData.g_weightsShort4x8,
  (4, 16): MIP.PdpData.g_weightsShort16x4,
  (16, 4): MIP.PdpData.g_weightsShort4x16,
  (8, 8): MIP.PdpData.g_weightsShort8x8,
  (8, 16): MIP.PdpData.g_weightsShort16x8,
  (16, 8): MIP.PdpData.g_weightsShort8x16,
  (16, 16): MIP.PdpData.g_weightsShort16x16,
  (16, 32): MIP.PdpData.g_weightsShort32x16,
  (32, 16): MIP.PdpData.g_weightsShort16x32,
  (32, 32): MIP.PdpData.g_weightsShort32x32,
}
]

def FillBlock(blockWithReferenceSamples, outputBlock, blockSize):
  blockWidth, blockHeight = blockSize
  RL = blockWithReferenceSamples.shape[0] % blockHeight
  if blockWidth == 32:
    if outputBlock.shape[0] < blockHeight:
      paddedOutputBlock = np.hstack((blockWithReferenceSamples[RL:RL+blockHeight:2, RL-1:RL], outputBlock))
    else:
      paddedOutputBlock = np.hstack((blockWithReferenceSamples[RL:RL+blockHeight, RL-1:RL], outputBlock))
    outputBlock = zoom(paddedOutputBlock, (1, 33/paddedOutputBlock.shape[1]), order=1)
    outputBlock = outputBlock[:, 1:]
  if blockHeight == 32:
    paddedOutputBlock = np.vstack((blockWithReferenceSamples[RL-1:RL, RL:RL+blockWidth], outputBlock))
    outputBlock = zoom(paddedOutputBlock, (33/paddedOutputBlock.shape[0], 1), order=1)
    outputBlock = outputBlock[1:, :]
  blockWithReferenceSamples[RL:RL+blockHeight,RL:RL+blockWidth] = outputBlock

  return blockWithReferenceSamples

for blockSize in blockSizes:
  # blockWidth = 32
  # blockHeight = 32
  blockWidth, blockHeight = blockSize
  RL = 2 if blockWidth * blockHeight <= 256 else 1

  saveFolder = f'./MIP/PdpVisualisation/{blockWidth}x{blockHeight}'
  if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)

  for invocationType in invocationTypes:
    if RL == 1:
      if invocationType in ['top_bar0', 'top_bar1', 'left_bar0', 'left_bar1']:
        continue

    print(f'Simulating: {blockWidth}x{blockHeight}', invocationType)
    blockWithReferenceSamples = np.full((blockHeight * 2 + RL, blockWidth * 2 + RL), 128)
    blockWithReferenceSamples = FillReferencePatterns.FillReferenceSamples(blockWithReferenceSamples, blockSize, invocationType)

    # Long reference samples
    stride = blockWidth * 2 + RL
    mirrorStride = blockHeight * 2 + RL
    referenceLength = (blockWidth * 2 + RL) * RL + blockHeight * 2 * RL
    referenceSamples = [np.zeros(referenceLength), np.zeros(referenceLength)]

    # Fill reference samples
    for i in range(RL):
      referenceSamples[0][(i*stride):((i+1)*stride)] = blockWithReferenceSamples[i,:]
      referenceSamples[1][(i*mirrorStride):((i+1)*mirrorStride)] = blockWithReferenceSamples.transpose()[i,:]
    for i in range(blockHeight*2):
      referenceSamples[0][RL*stride+i*RL:RL*stride+(i+1)*RL] = blockWithReferenceSamples[i+RL,0:RL]
    for i in range(blockWidth*2):
      referenceSamples[1][RL*mirrorStride+i*RL:RL*mirrorStride+(i+1)*RL] = blockWithReferenceSamples.transpose()[i+RL,0:RL]

    referenceSamples[0] = np.reshape(referenceSamples[0], (referenceLength, 1))
    referenceSamples[1] = np.reshape(referenceSamples[1], (referenceLength, 1))

    for mirror in range(2):
      for modeId in range(11 if RL == 2 else 7):

        if (modeId < 2 or modeId == (10 if RL == 2 else 6)) and mirror == 1:
          continue # don't need this
        
        weightMatrix = np.array(longWeights[mirror][(blockWidth, blockHeight)][modeId])

        # Compute matrix
        outputBlock = (np.matmul(weightMatrix, referenceSamples[mirror]) + ROUNDER) // DIVISOR
        if mirror == 0:
          outputBlock = np.reshape(outputBlock, (min(blockHeight, 16), min(blockWidth, 16)))
        else:
          outputBlock = np.reshape(outputBlock, (min(blockWidth, 16), min(blockHeight, 16))).transpose()

        # Fill back
        blockWithReferenceSamples = FillBlock(blockWithReferenceSamples, outputBlock, blockSize)

        # Save
        plt.imshow(blockWithReferenceSamples, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        rect = Rectangle((RL-0.5, RL-0.5), blockWidth, blockHeight, linewidth=RL, edgecolor='black', facecolor='none')
        plt.gca().add_patch(rect)
        modeNumber = modeId if modeId <= 1 else (modeId * 2 - 2 if RL == 2 else modeId * 4 - 6)
        if mirror:
          modeNumber = 68 - modeNumber
        fileName = f'{saveFolder}/{invocationType}_{modeNumber:02d}.png'
        plt.savefig(fileName, dpi=300, bbox_inches='tight')
        plt.close()

    # Short reference samples
    stride = blockWidth + RL
    mirrorStride = blockHeight + RL
    referenceLength = (blockWidth + RL) * RL + blockHeight * RL
    referenceSamples = [np.zeros(referenceLength), np.zeros(referenceLength)]

    # Fill reference samples
    for i in range(RL):
      referenceSamples[0][(i*stride):((i+1)*stride)] = blockWithReferenceSamples[i,0:stride]
      referenceSamples[1][(i*mirrorStride):((i+1)*mirrorStride)] = blockWithReferenceSamples.transpose()[i,0:mirrorStride]
    for i in range(blockHeight):
      referenceSamples[0][RL*stride+i*RL:RL*stride+(i+1)*RL] = blockWithReferenceSamples[i+RL,0:RL]
    for i in range(blockWidth):
      referenceSamples[1][RL*mirrorStride+i*RL:RL*mirrorStride+(i+1)*RL] = blockWithReferenceSamples.transpose()[i+RL,0:RL]

    referenceSamples[0] = np.reshape(referenceSamples[0], (referenceLength, 1))
    referenceSamples[1] = np.reshape(referenceSamples[1], (referenceLength, 1))
    for mirror in range(2):
      for modeId in range(8 if RL == 2 else 4):

        if (modeId == (7 if RL == 2 else 3)) and mirror == 1:
          continue # don't need this

        weightMatrix = np.array(shortWeights[mirror][(blockWidth, blockHeight)][modeId])

        # Compute matrix
        outputBlock = (np.matmul(weightMatrix, referenceSamples[mirror]) + ROUNDER) // DIVISOR
        if mirror == 0:
          outputBlock = np.reshape(outputBlock, (min(blockHeight, 16), min(blockWidth, 16)))
        else:
          outputBlock = np.reshape(outputBlock, (min(blockWidth, 16), min(blockHeight, 16))).transpose()

        # Fill back
        blockWithReferenceSamples = FillBlock(blockWithReferenceSamples, outputBlock, blockSize)

        # Save
        plt.imshow(blockWithReferenceSamples, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        rect = Rectangle((RL-0.5, RL-0.5), blockWidth, blockHeight, linewidth=RL, edgecolor='black', facecolor='none')
        plt.gca().add_patch(rect)
        modeNumber = modeId * 2 + 20 if RL == 2 else modeId * 4 + 22
        if mirror:
          modeNumber = 68 - modeNumber
        fileName = f'{saveFolder}/{invocationType}_{modeNumber:02d}.png'
        plt.savefig(fileName, dpi=300, bbox_inches='tight')
        plt.close()
