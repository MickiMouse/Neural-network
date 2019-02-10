import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from perceptron import Perceptron

df = load_digits()
X, y = df.data, df.target

# create train sample y like [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y_ = np.zeros([X.shape[0], 10])

for i in range(y.shape[0]):
    y_[i, y[i]] = 1

# normalize
X /= 15

model = Perceptron(X, y_, learning_rate=0.7)
model.add_layer(64, 30)
model.add_layer(30, 10)
model.train(800)

print("Precision of model: ", accuracy_score(y, np.ravel(model.predict(X))))

plt.plot(model.mse_for_plot)
plt.show()
