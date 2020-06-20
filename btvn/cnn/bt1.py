import numpy as np

'''
'''

def padding_zeros(matrix, number_padding):
  matrix = np.pad(matrix, [(number_padding, number_padding), (number_padding, number_padding)], mode='constant')
  print(matrix)
  return matrix

def convolution(x, w, s, p):
  height_begin = x.shape[0]
  width_begin = x.shape[1]
  # add padding
  if(p>0):
    x = padding_zeros(x,p)

  height = x.shape[0]
  width = x.shape[1]
  len_kernel = len(w)

  r_x = int(((height_begin-len_kernel + 2*p) / s)+1)
  r_y = int(((width_begin-len_kernel + 2*p) / s)+1)
  print('r_x, r_y = ', r_x,r_y)

  r = np.zeros((r_x,r_y))
  
  rangeRX = 0
  rangeRY = 0
  for i in range (0,height,s):
    for j in range (0,width,s):
      stx = i
      sty = j
      endx = i+len_kernel
      endy = j+len_kernel

      if (stx > height or endx > height or sty > width or endy > width):
        print('done x = ',i)
        break

      r[rangeRX][rangeRY] = np.sum(x[stx:endx,sty:endy] * w)
      rangeRY+=1

    rangeRX+=1
    rangeRY=0

  return r   

x = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
s = 2
p = 1
w = np.array([[1,0,1],[0,1,0],[1,0,1]])
r = convolution(x,w,s,p)
print(r)