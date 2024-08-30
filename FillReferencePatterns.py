invocationTypes = [
  # 'top_bar',
  'left_bar', 
  'left_bar0', 
  'left_bar1', 
  'left_zebra', 
  'left_zebra2', 
  'zebra', 
  'corner_TL', 
  'corner_BL'
]

def FillReferenceSamples(blockWithReferenceSamples, blockSize, invocationType):
  blockWidth, blockHeight = blockSize
  RL = blockWithReferenceSamples.shape[0] - blockHeight
  match invocationType:
    case 'corner_TL':
      blockWithReferenceSamples[0:RL,0:RL] = 192
    case 'corner_TL1x1far':
      blockWithReferenceSamples[0,0] = 192
    case 'corner_BL':
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
      blockWithReferenceSamples[1,:] = 192
    case 'top_bar1':
      blockWithReferenceSamples[0,:] = 192
    case 'left_bar':
      blockWithReferenceSamples[:,0:RL] = 192
    case 'left_bar0':
      blockWithReferenceSamples[:,1] = 192
    case 'left_bar1':
      blockWithReferenceSamples[:,0] = 192
    case 'left_zebra':
      blockWithReferenceSamples[::2,0:RL] = 192
    case 'left_zebra2':
      blockWithReferenceSamples[::4,0:RL] = 192
      blockWithReferenceSamples[1::4,0:RL] = 192
  
  return blockWithReferenceSamples
