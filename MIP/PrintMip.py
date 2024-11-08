import numpy as np
import MipData
import PdpData

numpy_arrays = np.array(MipData.data16x16)

for sample in range(0, 7):
  extract_element = numpy_arrays[:, :, sample]  # Extract the element corresponding to n-th sample along axis 0
  extract_element = extract_element.reshape((-1, 8, 8))
  for matrix in extract_element:
  for row in matrix:
    print(','.join(map(str, row)))
  print()

def ComputeMIP(reducedBoundary, dataMatrix):
  dataArrays = np.array(dataMatrix)

  pTemp0 = reducedBoundary[0]

  if(dataArrays.shape[2] == 7):
    reducedBoundary[1:] = [x - reducedBoundary[0] for x in reducedBoundary[1:]]
    reducedBoundary.pop(0)
  else:
    reducedBoundary[1:] = [x - reducedBoundary[0] for x in reducedBoundary[1:]]
    reducedBoundary[0] = 128 - reducedBoundary[0]

  reducedBoundary = np.array(reducedBoundary)

  oW = 32 - 32 * sum(reducedBoundary)

  returnValues = []
  for i in range(0, dataArrays.shape[0]):
    matrix = dataArrays[i, :, :]
    output = matrix.dot(reducedBoundary)
    output += oW
    output //= 64
    output += pTemp0
    if(output.size == 16):
      output = output.reshape(-1, 4, 4)
    else:
      output = output.reshape(-1, 8, 8)
    returnValues.append(output)

  return returnValues

def ComputePDP(referenceSamples, weightMatrix, modeIndex):
  # convert mode index


  weightArray = np.array(weightMatrix[modeId])

    # int sum = 0;
    # const int16_t* f = filter[y >> yShift][(x >> xShift)>>2];
    # for (int i = 0; i < refLen; ++i)
    # {
    # auto xs = x>>xShift;
    # sum += ref[i] * f[(i / 8) * 32 + (xs % 4) * 8 + ( i % 8 )];
    # }
    # pred[x] = (Pel)Clip3(clpRng.min, clpRng.max, (std::max(0, sum) + addShift) >> 14);

  output = weightArray.mul(referenceSamples)


  return returnValues

# boundary = [138, 128, 128, 128]
# predictors = ComputeMIP(boundary, data4x4)
# for p in predictors:
#   for row in p:
#     for elem in row:
#       print("{}".format(elem).rjust(5), end="\n")
#   print()
