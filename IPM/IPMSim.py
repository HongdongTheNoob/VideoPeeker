import numpy as np
from . import IntraFilters

angTable = [ 
   0,    1,    2,    3,    4,    6,    8,   10,   
  12,   14,   16,   18,   20,   23,   26,   29,   
  32,   35,   39,   45,   51,   57,   64,   73,
  86,  102,  128,  171,  256,  341,  512, 1024 
]
invAngTable = [
  0,   16384, 8192, 5461, 4096, 2731, 2048, 1638, 
  1365, 1170, 1024, 910, 819, 712, 630, 565,
  512, 468,   420,  364,  321,  287,  256,  224,  
  191,  161,  128,  96,  64,  48,  32,  16
]

def FillMainReferenceLine(blockWidthReferenceLine, blockSize, modeId, filterTaps = 6):
  blockWidth, blockHeight = blockSize
  intraAnglePredMode = modeId - 50

  # Top part
  middlePart = np.array(blockWidthReferenceLine[0, 0:8*blockWidth+1]).reshape((1, 8*blockWidth+1))

  # Extend to the right
  extendedRight = np.full((1, filterTaps), middlePart[0, -1])

  # Left part
  leftPart = np.zeros((1, blockHeight + 1))

  if intraAnglePredMode >= 0:
    maxLeftOffset = (filterTaps - 1) // 2
    leftPart[0, -(maxLeftOffset-1):0] = middlePart[0, 0]
  else:
    for deltaX in range(-blockHeight-1, 0):
      idx = blockHeight + 1 + deltaX
      pixelPosition = (-deltaX * invAngTable[abs(intraAnglePredMode)] + 8) // 16
      integerPel = pixelPosition // 32
      fractionalPel = pixelPosition % 32

      referenceSamples = [
        blockWidthReferenceLine[min(max(0, integerPel-1), blockHeight), 0],
        blockWidthReferenceLine[min(max(0, integerPel), blockHeight), 0],
        blockWidthReferenceLine[min(max(0, integerPel+1), blockHeight), 0],
        blockWidthReferenceLine[min(max(0, integerPel+2), blockHeight), 0]
      ]
      interpolatedValue = np.dot(np.array(referenceSamples), np.array(IntraFilters.weak_4tap_filter[fractionalPel]))
      leftPart[0, idx] = (interpolatedValue + 32) // 64

      # referenceSamples = [
      #   blockWidthReferenceLine[min(max(0, integerPel-2), blockHeight), 0],
      #   blockWidthReferenceLine[min(max(0, integerPel-1), blockHeight), 0],
      #   blockWidthReferenceLine[min(max(0, integerPel), blockHeight), 0],
      #   blockWidthReferenceLine[min(max(0, integerPel+1), blockHeight), 0],
      #   blockWidthReferenceLine[min(max(0, integerPel+2), blockHeight), 0]
      # ]
      # interpolatedValue = np.dot(np.array(referenceSamples), np.array(IntraFilters.test_5tap_filter[fractionalPel]))
      # leftPart[0, idx] = (interpolatedValue + 32) // 64

      # referenceSamples = [
      #   blockWidthReferenceLine[min(max(0, integerPel-2), blockHeight), 0],
      #   blockWidthReferenceLine[min(max(0, integerPel-1), blockHeight), 0],
      #   blockWidthReferenceLine[min(max(0, integerPel), blockHeight), 0],
      #   blockWidthReferenceLine[min(max(0, integerPel+1), blockHeight), 0],
      #   blockWidthReferenceLine[min(max(0, integerPel+2), blockHeight), 0],
      #   blockWidthReferenceLine[min(max(0, integerPel+3), blockHeight), 0]
      # ]
      # interpolatedValue = np.dot(np.array(referenceSamples), np.array(IntraFilters.luma_intra_filter[fractionalPel]))
      # leftPart[0, idx] = (interpolatedValue + 128) // 256

    
  # concat
  mainReferenceLineLine = np.hstack((leftPart, middlePart, extendedRight)).flatten()

  return mainReferenceLineLine

def PredIntraAngular(blockWidthReferenceLine, blockSize, modeId, forcePDPC = 0):
  blockWidth, blockHeight = blockSize

  # if modeId < 2 or modeId > 66:
  if modeId < -14 or modeId > 80: # support WAIP
    print("Wrong mode ID")
    return

  transposed = (modeId < 34)
  waip = (modeId < 2) or (modeId > 66)
  predModeIndex = modeId
  if modeId < 34: # transpose
    blockWidthReferenceLine = blockWidthReferenceLine.transpose()
    blockHeight, blockWidth = blockWidth, blockHeight
    predModeIndex = 68 - modeId if modeId >= 2 else 68 - (modeId + 2)
  
  predModeSign = 1 if predModeIndex > 50 else (-1 if predModeIndex < 50 else 0)
  
  mainReferenceLine = FillMainReferenceLine(blockWidthReferenceLine, (blockWidth, blockHeight), predModeIndex)
  xx, yy = np.meshgrid(range(1,blockWidth+1), range(1,blockHeight+1))

  referenceLinePosition = (xx * 32) + yy * predModeSign * angTable[abs(predModeIndex - 50)]
  referenceLinePosition = referenceLinePosition + ((blockHeight + 1) * 32)
  referenceLinePosition = np.clip(referenceLinePosition, -np.inf, (blockWidth * 8 + blockHeight) * 32).astype(int)
  referenceLinePositionInteger = referenceLinePosition // 32
  referenceLinePositionFractional = referenceLinePosition % 32

  # const TFilterCoeff* const f = (useCubicFilter) ? ( bExtIntraDir ? InterpolationFilter::getIntraLumaFilterTableExt(deltaFract) : InterpolationFilter::getIntraLumaFilterTable(deltaFract)) : (width >=32 && height >=32)? (bExtIntraDir ? intraSmoothingFilter2Ext : intraSmoothingFilter2) : (bExtIntraDir ? intraSmoothingFilterExt : intraSmoothingFilter);
  # if useCubicFilter: !m_ipaParam.interpolationFlag >> reference Filter >> integer slope modes >> angTable value is multiples of 32
  #   InterpolationFilter::getIntraLumaFilterTable(deltaFract)
  # else:
  #   if width >=32 && height >=32: 
  #     intraSmoothingFilter2
  #   else: 
  #     intraSmoothingFilter
  # if blockWidth >=32 and blockHeight >=32:
  #   smoothingFilter = [-(referenceLinePositionFractional // 2) + 16, 
  #                           -3 * (referenceLinePositionFractional // 2) + 64,
  #                           -referenceLinePositionFractional + 96,
  #                           referenceLinePositionFractional + 64,
  #                           3 * (referenceLinePositionFractional // 2) + 16,
  #                           referenceLinePositionFractional // 2]
  # else:
  #   smoothingFilter = [referenceLinePositionFractional * 0, 
  #                           -2 * referenceLinePositionFractional + 64,
  #                           -2 * referenceLinePositionFractional + 128,
  #                           2 * referenceLinePositionFractional + 64,
  #                           2 * referenceLinePositionFractional,
  #                           referenceLinePositionFractional * 0]
    
  predictionFilter = [np.zeros((blockHeight, blockWidth)) for _ in range(6)]

  if predModeIndex > 66:
    for y in range(blockHeight):
      for x in range(blockWidth):
          for f in range(6):
            predictionFilter[f][y][x] = IntraFilters.luma_intra_filter_ext[referenceLinePositionFractional[y][x]][f]
      predictionBlock = np.zeros((blockHeight, blockWidth)) + 128
      for filterTap in range(6):
        predictionBlock += np.multiply(predictionFilter[filterTap], mainReferenceLine[(referenceLinePositionInteger - 2 + filterTap)])
      predictionBlock = predictionBlock // 256
  else:
    if angTable[abs(predModeIndex - 50)] % 32 == 0:
      predictionBlock = mainReferenceLine[referenceLinePositionInteger]
    else:
      for y in range(blockHeight):
        for x in range(blockWidth):
            for f in range(6):
              predictionFilter[f][y][x] = IntraFilters.luma_intra_filter[referenceLinePositionFractional[y][x]][f]
      predictionBlock = np.zeros((blockHeight, blockWidth)) + 128
      for filterTap in range(6):
        predictionBlock += np.multiply(predictionFilter[filterTap], mainReferenceLine[(referenceLinePositionInteger - 2 + filterTap)])
      predictionBlock = predictionBlock // 256

  if forcePDPC > 0 or (forcePDPC == 0 and predModeSign > 0):
    # xx, yy = np.meshgrid(range(1,blockWidth+1), range(1,blockHeight+1))
    invAngle = invAngTable[abs(predModeIndex - 50)]
    yIntercept = (yy - 1) + (xx * invAngle + 256) // 512
    yIntercept[yIntercept > blockHeight * 8 - 1] = blockHeight * 8 - 1
    scale = min(2, np.log2(blockHeight - np.floor(np.log2(3 * invAngle - 2))) + 8)
    weightLeftShiftBit = (xx * 2) // np.power(2, scale)
    weightLeft = 32 // np.power(2, weightLeftShiftBit)
    oppositeBlock = blockWidthReferenceLine[yIntercept+1, 0]
    oppositeBlock[xx >= 3 * np.power(2,scale)] = 0
    predictionBlock = (np.multiply(oppositeBlock, weightLeft) + np.multiply(predictionBlock, 64 - weightLeft) + 32) // 64

  if transposed: # transpose back
    predictionBlock = predictionBlock.transpose()

  return predictionBlock
