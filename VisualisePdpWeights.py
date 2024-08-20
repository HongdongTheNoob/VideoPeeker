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

blockWidth = 8
blockHeight = 8
referenceLines = 2

blockWithReferenceSamples = np.full((blockHeight * 2 + referenceLines, blockWidth * 2 + referenceLines), 128)
invocation_type = 'corner_topleft1x1far'

if invocation_type == 'corner_topleft':
  blockWithReferenceSamples[0:2,0:2] = 192
if invocation_type == 'corner_topleft1x1far':
  blockWithReferenceSamples[0,0] = 192
if invocation_type == 'top_bar':
  blockWithReferenceSamples[0:2,:] = 192
if invocation_type == 'top_bar0':
  blockWithReferenceSamples[1,:] = 192
if invocation_type == 'top_bar1':
  blockWithReferenceSamples[0,:] = 192
if invocation_type == 'left_bar':
  blockWithReferenceSamples[:,0:2] = 192
if invocation_type == 'left_bar0':
  blockWithReferenceSamples[:,1] = 192
if invocation_type == 'left_bar1':
  blockWithReferenceSamples[:,0] = 192

# Long reference samples
stride = blockWidth * 2 + referenceLines
referenceLength = (blockWidth * 2 + referenceLines) * referenceLines + blockHeight * 2 * referenceLines
referenceSamples = np.zeros(referenceLength)

# Fill reference samples
for i in range(referenceLines):
  referenceSamples[(i*stride):((i+1)*stride)] = blockWithReferenceSamples[i,:]
for i in range(blockHeight*2):
  referenceSamples[referenceLines*stride+i*referenceLines:referenceLines*stride+(i+1)*referenceLines] = blockWithReferenceSamples[i+referenceLines,0:referenceLines]

referenceSamples = np.reshape(referenceSamples, (referenceLength, 1))
for modeId in range(11):

  weightMatrix = np.array(MIP.PdpData.g_weights8x8[modeId])

  # Compute matrix
  outputBlock = np.matmul(weightMatrix, referenceSamples) // np.power(2, 14)
  outputBlock = np.reshape(outputBlock, (blockHeight, blockWidth))

  # Fill back
  blockWithReferenceSamples[referenceLines:referenceLines+blockHeight,referenceLines:referenceLines+blockWidth] = outputBlock

  # Display
  plt.imshow(blockWithReferenceSamples, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
  rect = Rectangle((referenceLines-0.5, referenceLines-0.5), blockWidth, blockHeight,
                  linewidth=2, edgecolor='black', facecolor='none')
  plt.gca().add_patch(rect)
  # plt.show()

  modeNumber = (modeId * 2 - 2) if modeId > 1 else modeId
  fileName = f'./MIP/PdpVisualisation/{invocation_type}_{modeNumber:02d}.png'
  plt.savefig(fileName, dpi=300, bbox_inches='tight')

# Short reference samples
stride = blockWidth + referenceLines
referenceLength = (blockWidth + referenceLines) * referenceLines + blockHeight * referenceLines
referenceSamples = np.zeros(referenceLength)

# Fill reference samples
for i in range(referenceLines):
  referenceSamples[(i*stride):((i+1)*stride)] = blockWithReferenceSamples[i,0:stride]
for i in range(blockHeight):
  referenceSamples[referenceLines*stride+i*referenceLines:referenceLines*stride+(i+1)*referenceLines] = blockWithReferenceSamples[i+referenceLines,0:referenceLines]

referenceSamples = np.reshape(referenceSamples, (referenceLength, 1))
for modeId in range(8):

  weightMatrix = np.array(MIP.PdpData.g_weightsShort8x8[modeId])

  # Compute matrix
  outputBlock = np.matmul(weightMatrix, referenceSamples) // np.power(2, 14)
  outputBlock = np.reshape(outputBlock, (blockHeight, blockWidth))

  # Fill back
  blockWithReferenceSamples[referenceLines:referenceLines+blockHeight,referenceLines:referenceLines+blockWidth] = outputBlock

  # Display
  plt.imshow(blockWithReferenceSamples, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
  rect = Rectangle((referenceLines-0.5, referenceLines-0.5), blockWidth, blockHeight,
                  linewidth=2, edgecolor='black', facecolor='none')
  plt.gca().add_patch(rect)
  # plt.show()

  modeNumber = modeId * 2 + 20
  fileName = f'./MIP/PdpVisualisation/{invocation_type}_{modeNumber:02d}.png'
  plt.savefig(fileName, dpi=300, bbox_inches='tight')