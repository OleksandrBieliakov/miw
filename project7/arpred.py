import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

a = np.loadtxt('alior.txt')

y_train = a[:50, [0]]
train = a[:50, 1:]

y_test = a[50:100, [0]]
test = a[50:100, 1:]

v = np.linalg.pinv(train) @ y_train

print(v)

plt.plot(y_train, 'r-')
plt.plot(v[0] * train[:, [0]] + v[1] * train[:, [1]] + v[2] * train[:, [2]] + v[3] * train[:, [3]])
plt.show()

plt.plot(y_test, 'r-')
plt.plot(v[0] * test[:, [0]] + v[1] * test[:, [1]] + v[2] * test[:, [2]] + v[3] * test[:, [3]])
plt.show()
