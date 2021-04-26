import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def print_zip(x, y):
    for n, m in zip(x, y):
        print(n, m)


a = np.loadtxt('../data/project4/dane2.txt')

x_all = a[:, [1]]
y_all = a[:, [0]]

x, x_test, y, y_test = train_test_split(x_all, y_all, test_size=0.5, random_state=3)

c = np.hstack([x, np.ones(x.shape)])
c3 = np.hstack([np.cbrt(x), np.ones(x.shape)])

v = np.linalg.pinv(c) @ y
v3 = np.linalg.pinv(c3) @ y

e = sum((y - (v[0] * x + v[1])) ** 2)/len(y)
e3 = sum((y - (v3[0] * np.cbrt(x) + v3[1])) ** 2)/len(y)

e_test = sum((y_test - (v[0] * x_test + v[1])) ** 2)/len(y_test)
e3_test = sum((y_test - (v3[0] * np.cbrt(x_test) + v3[1])) ** 2)/len(y_test)

print('TRAINING SET ERROR:', 'linear -', e, 'cubic root -', e3)
print('TESTING SET ERROR:', 'linear -', e_test, 'cubic root -', e3_test)

plt.plot(x, y, 'ro')
x = sorted(x)
plt.plot(x, v[0] * x + v[1])
plt.plot(x, v3[0] * np.cbrt(x) + v3[1])
plt.show()

plt.plot(x_test, y_test, 'ro')
x_test = sorted(x_test)
plt.plot(x_test, v[0] * x_test + v[1])
plt.plot(x_test, v3[0] * np.cbrt(x_test) + v3[1])
plt.show()
