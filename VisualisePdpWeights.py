import os
import sys
import cv2
import numpy as np
import json
import csv
import MIP.PdpData
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom

ROUNDER = np.power(2, 13)
DIVISOR = np.power(2, 14)

# get weights
blockSizes = [
  # (4, 4), 
  (4, 8), 
  (8, 4), 
  (4, 16), 
  (16, 4), 
  # (8, 8), 
  (8, 16), 
  (16, 8), 
  # (16, 16), 
  (16, 32), 
  (32, 16), 
  # (32, 32)
]

longWeights = {
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
}

shortWeights = {
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
}

invocation_types = [
  'left_bar', 
  'left_bar0', 
  'left_bar1', 
  'left_zebra', 
  'left_zebra2', 
  'zebra', 
  'corner_topleft', 
  'corner_bottomleft'
]

for blockSize in blockSizes:
  # blockWidth = 32
  # blockHeight = 32
  blockWidth, blockHeight = blockSize
  RL = 2 if blockWidth * blockHeight <= 256 else 1

  saveFolder = f'./MIP/PdpVisualisation/{blockWidth}x{blockHeight}'
  if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)

  for invocation_type in invocation_types:
    print(f'Simulating: {blockWidth}x{blockHeight}', invocation_type)
    blockWithReferenceSamples = np.full((blockHeight * 2 + RL, blockWidth * 2 + RL), 128)

    match invocation_type:
      case 'corner_topleft':
        blockWithReferenceSamples[0:RL,0:RL] = 192
      case 'corner_topleft1x1far':
        blockWithReferenceSamples[0,0] = 192
      case 'corner_bottomleft':
        blockWithReferenceSamples[blockHeight:blockHeight+RL,0:RL] = 192
      case 'zebra':
        blockWithReferenceSamples[0:RL,0:RL] = 192
        blockWithReferenceSamples[0:RL,1::2] = 192
        blockWithReferenceSamples[1::2,0:RL] = 192
      case 'zebra2':
        blockWithReferenceSamples[0:RL,::4] = 192
        blockWithReferenceSamples[0:RL,1::4] = 192
        blockWithReferenceSamples[::4,0:RL] = 192
        blockWithReferenceSamples[1::4,0:RL] = 192
      case 'top_bar':
        blockWithReferenceSamples[0:RL,:] = 192
      case 'top_bar0':
        if RL == 1:
          continue
        blockWithReferenceSamples[1,:] = 192
      case 'top_bar1':
        if RL == 1:
          continue
        blockWithReferenceSamples[0,:] = 192
      case 'left_bar':
        blockWithReferenceSamples[:,0:RL] = 192
      case 'left_bar0':
        if RL == 1:
          continue
        blockWithReferenceSamples[:,1] = 192
      case 'left_bar1':
        if RL == 1:
          continue
        blockWithReferenceSamples[:,0] = 192
      case 'left_zebra':
        blockWithReferenceSamples[::2,0:RL] = 192
      case 'left_zebra2':
        blockWithReferenceSamples[::4,0:RL] = 192
        blockWithReferenceSamples[1::4,0:RL] = 192

    # Long reference samples
    stride = blockWidth * 2 + RL
    referenceLength = (blockWidth * 2 + RL) * RL + blockHeight * 2 * RL
    referenceSamples = np.zeros(referenceLength)

    # Fill reference samples
    for i in range(RL):
      referenceSamples[(i*stride):((i+1)*stride)] = blockWithReferenceSamples[i,:]
    for i in range(blockHeight*2):
      referenceSamples[RL*stride+i*RL:RL*stride+(i+1)*RL] = blockWithReferenceSamples[i+RL,0:RL]

    referenceSamples = np.reshape(referenceSamples, (referenceLength, 1))
    for modeId in range(11 if RL == 2 else 7):

      weightMatrix = np.array(longWeights[(blockWidth, blockHeight)][modeId])

      # Compute matrix
      # outputBlock = np.right_shift((np.matmul(weightMatrix, referenceSamples).astype(int) + ROUNDER), 14)
      outputBlock = (np.matmul(weightMatrix, referenceSamples) + ROUNDER) // DIVISOR
      outputBlock = np.reshape(outputBlock, (min(blockHeight, 16), min(blockWidth, 16)))

      # Fill back
      if blockWidth == 32:
        if outputBlock.shape[0] == 16:
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

      # Display
      plt.imshow(blockWithReferenceSamples, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
      rect = Rectangle((RL-0.5, RL-0.5), blockWidth, blockHeight,
                      linewidth=2, edgecolor='black', facecolor='none')
      plt.gca().add_patch(rect)
      # plt.show()

      modeNumber = modeId if modeId <= 1 else (modeId * 2 - 2 if RL == 2 else modeId * 4 - 6)
      fileName = f'{saveFolder}/{invocation_type}_{modeNumber:02d}.png'
      plt.savefig(fileName, dpi=300, bbox_inches='tight')

    # Short reference samples
    stride = blockWidth + RL
    referenceLength = (blockWidth + RL) * RL + blockHeight * RL
    referenceSamples = np.zeros(referenceLength)

    # Fill reference samples
    for i in range(RL):
      referenceSamples[(i*stride):((i+1)*stride)] = blockWithReferenceSamples[i,0:stride]
    for i in range(blockHeight):
      referenceSamples[RL*stride+i*RL:RL*stride+(i+1)*RL] = blockWithReferenceSamples[i+RL,0:RL]

    referenceSamples = np.reshape(referenceSamples, (referenceLength, 1))
    for modeId in range(8 if RL == 2 else 4):

      weightMatrix = np.array(shortWeights[(blockWidth, blockHeight)][modeId])

      # Compute matrix
      # outputBlock = np.right_shift((np.matmul(weightMatrix, referenceSamples).astype(int) + ROUNDER), 14)
      outputBlock = (np.matmul(weightMatrix, referenceSamples) + ROUNDER) // DIVISOR
      outputBlock = np.reshape(outputBlock, (min(blockHeight, 16), min(blockWidth, 16)))

      # Fill back
      if blockWidth == 32:
        if outputBlock.shape[0] == 16:
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

      # Display
      plt.imshow(blockWithReferenceSamples, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
      rect = Rectangle((RL-0.5, RL-0.5), blockWidth, blockHeight,
                      linewidth=2, edgecolor='black', facecolor='none')
      plt.gca().add_patch(rect)
      # plt.show()

      modeNumber = modeId * 2 + 20 if RL == 2 else modeId * 4 + 22
      fileName = f'{saveFolder}/{invocation_type}_{modeNumber:02d}.png'
      plt.savefig(fileName, dpi=300, bbox_inches='tight')