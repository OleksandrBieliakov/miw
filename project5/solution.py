import numpy as np
import matplotlib.pyplot as plt
import numpy.random as r

P = np.arange(-2, 2, 0.1)
T = P**2 + r.random(len(P))

S1 = 4
W1 = r.rand(S1, 1) - 0.5
B1 = r.rand(S1, 1) - 0.5
W2 = r.rand(1, S1) - 0.5
B2 = r.rand(1, 1) - 0.5
lr = 0.01

for epoch in range(1):

    X = W1 * P + B1 * np.ones(len(P))
    A1 = np.tanh(X)
    A2 = W2 * A1 + B2
    print(A2.shape)

    # E2 = T - A2
    # E1 = W2'*E2;
    #
    # dW2 = lr * E2 * A1'
    # dB2 = lr * E2 * ones(size(E2))'
    # dW1 = lr * (exp(X). / (exp(X) + 1)). * E1 * P'
    # dB1 = lr * (exp(X). / (exp(X) + 1)). * E1 * ones(size(P))'
    #
    # W2 = W2 + dW2
    # B2 = B2 + dB2
    # W1 = W1 + dW1
    # B1 = B1 + dB1

# plt.plot(P, T, 'ro')
# plt.plot(P, A2)
# plt.show()

