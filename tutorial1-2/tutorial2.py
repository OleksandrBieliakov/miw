import numpy as np
import matplotlib.pyplot as mp

#import tensorflow as tw

matrix = np.loadtxt('data/tutorial1-2/Sharp_char.txt')

x = matrix[:,[0]]
y = matrix[:,[1]]

mp.plot(x, y, 'r*')
mp.xlabel('voltage')
mp.ylabel('distance')
mp.title('Sensor characteristic')
mp.show()

x, y = y, x
mp.plot(x, y, 'r*')
mp.xlabel('distance')
mp.ylabel('voltage')
mp.title('Sensor characteristic - reverse function')
mp.show()
