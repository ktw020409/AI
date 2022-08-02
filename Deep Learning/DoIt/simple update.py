from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

x = diabetes.data[:, 2]
y = diabetes.target

w = 1.0
b = 1.0

print("변화율로 가중치 업데이트하기")

y_hat = w * x[0] + b
print("y : " + str(y[0]) + ", y_hat : " + str(y_hat))

w_inc = w + 0.1
y_hat_inc = w_inc * x[0] + b
print("y : " + str(y[0]) + ", y_hat_inc : " + str(y_hat_inc))

w_rate = (y_hat_inc - y_hat) / (w_inc - w)
#y_hat_inc - y_hat = (w_inc - w) * x[0] + b - b = (w_inc - w) * x[0]
#(y_hat_inc - y_hat) / (w_inc - w) = (w_inc - w) * x[0] / (w_inc - w) = x[0]
#so, w_rate == x[0]
print("w_rate : " + str(w_rate) + ", x[0] : " + str(x[0]))

w_new = w + w_rate
print("w_new : " + str(w_new))

print()
print("변화율로 절편 업데이트하기")

b_inc = b + 0.1
y_hat_inc = w * x[0] + b_inc
print("y : " + str(y[0]) + ", y_hat_inc : " + str(y_hat_inc))

b_rate = (y_hat_inc - y_hat) / (b_inc - b)
print("b_rate : " + str(b_rate))

b_new = b + 1
print("b_new : " + str(b_new))