import numpy as np

modeIdx_to_angleIdx = [0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 8, 8, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 16, 16, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 24, 24, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30] 
modeIdx_to_distanceIdx = [1, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
angleIdx_to_intraIdx = [50, 0, 44, 41, 34, 27, 0, 0, 18, 0, 0, 9, 66, 59, 56, 0, 50, 0, 44, 41, 34, 27, 0, 0, 18, 0, 0, 9, 66, 59, 56, 0]

g_dis = [ 8, 8, 8, 8, 4, 4, 2, 1, 0, -1, -2, -4, -4, -8, -8, -8, -8, -8, -8, -8, -4, -4, -2, -1, 0, 1, 2, 4, 4, 8, 8, 8 ]
g_bld2Width = [ 1, 2, 4, 8, 16, 32 ]
g_angle2mask = [ 0, -1, 1, 2, 3, 4, -1, -1, 5, -1, -1, 4, 3, 2, 1, -1, 0, -1, 1, 2, 3, 4, -1, -1, 5, -1, -1, 4, 3, 2, 1, -1 ]
g_angle2mirror = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2 ]
GEO_NUM_ANGLES = 32
GEO_MAX_CU_LOG2 = 6
GEO_MAX_CU_SIZE = 64
GEO_WEIGHT_MASK_SIZE = 112

# Define constants
GEO_NUM_CU_SIZE = 5
GEO_MIN_CU_LOG2 = 2
GEO_NUM_PARTITION_MODE = 64

def generate_geo_weights():
  g_geoWeights = np.zeros((6, 6, GEO_WEIGHT_MASK_SIZE * GEO_WEIGHT_MASK_SIZE))

  for angleIdx in range(0, 9):
    if g_angle2mask[angleIdx] == -1:
      continue
    for bldIdx in range(0, 6):
      distanceX = angleIdx
      distanceY = (distanceX + (GEO_NUM_ANGLES >> 2)) % GEO_NUM_ANGLES
      rho = (g_dis[distanceX] << (GEO_MAX_CU_LOG2 + 1)) + (g_dis[distanceY] << (GEO_MAX_CU_LOG2 + 1))
      maskOffset = (2 * GEO_MAX_CU_SIZE - GEO_WEIGHT_MASK_SIZE) >> 1
      index = 0

      for y in range(GEO_WEIGHT_MASK_SIZE):
        lookUpY = (((y + maskOffset) << 1) + 1) * g_dis[distanceY]
        for x in range(GEO_WEIGHT_MASK_SIZE):
          sxi = ((x + maskOffset) << 1) + 1
          weightIdx = sxi * g_dis[distanceX] + lookUpY - rho
          if g_bld2Width[bldIdx] > 1:
            weightLinearIdx = 8 * g_bld2Width[bldIdx] + weightIdx
            g_geoWeights[bldIdx][g_angle2mask[angleIdx]][index] = max(0, min(32, (weightLinearIdx + (g_bld2Width[bldIdx] >> 2)) // (g_bld2Width[bldIdx] >> 1)))
          else:
            weightLinearIdx = 8 + weightIdx
            g_geoWeights[bldIdx][g_angle2mask[angleIdx]][index] = max(0, min(32, weightLinearIdx << 1))
          index += 1

  return g_geoWeights

def generate_weight_offset():

  # Initialize g_weightOffset array
  g_weightOffset = np.zeros((GEO_NUM_PARTITION_MODE, GEO_NUM_CU_SIZE, GEO_NUM_CU_SIZE, 2), dtype=np.int16)

  # Iterate over height and width indices
  for hIdx in range(GEO_NUM_CU_SIZE):
    height = 1 << (hIdx + GEO_MIN_CU_LOG2)
    for wIdx in range(GEO_NUM_CU_SIZE):
      width = 1 << (wIdx + GEO_MIN_CU_LOG2)
      # Iterate over split directions
      for splitDir in range(GEO_NUM_PARTITION_MODE):
        angle = modeIdx_to_angleIdx[splitDir]
        distance = modeIdx_to_distanceIdx[splitDir]
        offsetX = (GEO_WEIGHT_MASK_SIZE - width) >> 1
        offsetY = (GEO_WEIGHT_MASK_SIZE - height) >> 1
        if distance > 0:
          if angle % 16 == 8 or (angle % 16 != 0 and height >= width):
            offsetY += ((distance * height) >> 3) if angle < 16 else -((distance * height) >> 3)
          else:
            offsetX += ((distance * width) >> 3) if angle < 16 else -((distance * width) >> 3)
        g_weightOffset[splitDir, hIdx, wIdx, 0] = offsetX
        g_weightOffset[splitDir, hIdx, wIdx, 1] = offsetY

  return g_weightOffset