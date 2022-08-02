import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

x = diabetes.data[:, 2]
y = diabetes.target

w = 1.0
b = 1.0

for i in range(100):
    for x_i, y_i in zip(x, y):
        y_hat = w * x_i + b
        err = y_i - y_hat

        w_rate = x_i
        w = w + w_rate * err

        b = b + err

#위 모델을 바탕으로 새로운 데이터 예측
x_new = 0.18
y_pred = w * x_new + b

plt.scatter(x, y)
pt1 = (-0.1, w * -0.1 + b)
pt2 = (0.15, w * 0.15 + b)

plt.scatter(x_new, y_pred)

plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')

plt.show()