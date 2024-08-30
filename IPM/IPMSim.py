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
  middlePart = np.array(blockWidthReferenceLine[0, 0:blockWidth*2+1]).reshape((1, blockWidth*2+1))

  # Extend to the right
  extendedRight = np.full((1, filterTaps), middlePart[0, -1])

  # Left part
  leftPart = np.zeros((1, blockHeight + 1))

  print(modeId, leftPart.shape, middlePart.shape, extendedRight.shape)

  if intraAnglePredMode >= 0:
    maxLeftOffset = (filterTaps - 1) // 2
    leftPart[-(maxLeftOffset-1):0] = middlePart[0, 0]
  else:
    for deltaX in range(-blockHeight+1, 0):
      idx = blockHeight + 1 + deltaX
      pixelPosition = (-deltaX * invAngTable[abs(intraAnglePredMode)] + 8) // 16
      integerPel = pixelPosition // 32
      fractionalPel = pixelPosition % 32

      referenceSamples = [
        blockWidthReferenceLine[max(0, integerPel-1), 0],
        blockWidthReferenceLine[max(0, integerPel), 0],
        blockWidthReferenceLine[max(0, integerPel+1), 0],
        blockWidthReferenceLine[max(0, integerPel+2), 0]
      ]

      interpolatedValue = np.dot(np.array(referenceSamples), np.array(IntraFilters.weak_4tap_filter[fractionalPel]))
      leftPart[idx] = (interpolatedValue + 32) // 64
    
  # concat
  mainReferenceLineLine = np.hstack((leftPart, middlePart, extendedRight))

  return mainReferenceLineLine

def PredIntraAngular(blockWidthReferenceLine, blockSize, modeId):
  blockWidth, blockHeight = blockSize

  if modeId < 2 or modeId > 66:
    print("Wrong mode ID")
    return

  transposed = (modeId < 34)
  predModeIndex = modeId
  if predModeIndex < 34: # transpose
    blockWidthReferenceLine = blockWidthReferenceLine.transpose()
    predModeIndex = 68 - modeId
  
  predModeSign = 1 if predModeIndex > 50 else (-1 if predModeIndex < 50 else 0)
  
  mainReferenceLine = FillMainReferenceLine(blockWidthReferenceLine, blockSize, predModeIndex)

  yy, xx = np.meshgrid(blockHeight, blockWidth, indexing='ij')

  referenceLinePosition = (xx * 32) + yy * predModeSign * angTable[abs(predModeIndex - 50)]
  referenceLinePosition = referenceLinePosition + ((blockHeight + 1) * 32)
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
  if angTable[referenceLinePositionFractional] % 32 == 0:
    if blockWidth >=32 and blockHeight >=32:
      predictionFilter = [-(referenceLinePositionFractional // 2) + 16, 
                              -3 * (referenceLinePositionFractional // 2) + 64,
                              -referenceLinePositionFractional + 96,
                              referenceLinePositionFractional + 64,
                              3 * (referenceLinePositionFractional // 2) + 16,
                              referenceLinePositionFractional // 2]
    else:
      predictionFilter = [referenceLinePositionFractional * 0, 
                              -2 * referenceLinePositionFractional + 64,
                              -2 * referenceLinePositionFractional + 128,
                              2 * referenceLinePositionFractional + 64,
                              2 * referenceLinePositionFractional,
                              referenceLinePositionFractional * 0]
  else:
    predictionFilter = IntraFilters.luma_intra_filter[referenceLinePositionFractional, 0:6]
    
  predictionBlock = np.zeros(predictionFilter[0].shape) + 128
  for filterTap in range(6):
    predictionBlock += np.multiply(predictionFilter[0], mainReferenceLine[referenceLinePositionInteger])
  predictionBlock = predictionBlock // 128

  return predictionBlock
