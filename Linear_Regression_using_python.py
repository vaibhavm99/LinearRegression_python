from statistics import mean
from matplotlib import pyplot as plt 
import numpy as np 
import pandas as pd

df = pd.read_csv("weather.csv")
print(df.columns)

X = df["Temperature "]
y = df['Relative Humidity ']
X = np.array(X, dtype=np.float64) 
y = np.array(y, dtype=np.float64)


def best_fit_slope(X, y):
    X_mean = mean(X)
    y_mean = mean(y)
    Xy = X*y
    Xy_mean = mean(Xy)
    X_2 = X**2
    X_2_mean = mean(X_2)
    m = ((X_mean * y_mean) - Xy_mean)/(X_mean ** 2 - X_2_mean)
    return m

def y_intersept(X, y, m):
    X_mean = mean(X)
    y_mean = mean(y)
    b = y_mean - (m * X_mean)
    return b


def squared_error(y_orig, y_line):
    return sum((y_line - y_orig) ** 2)


def coefficient_of_determination(y_orig, y_line):
    y_mean_line = [mean(y_orig) for y in y_orig]
    squared_error_regrr_line = squared_error(y_orig, y_line)
    squared_error_mean_line = squared_error(y_orig, y_mean_line)
    return 1 - (squared_error_regrr_line / squared_error_mean_line)

m = best_fit_slope(X, y)
b = y_intersept(X, y, m)






#R2 error
y_predict = m * X + b 
r2 = coefficient_of_determination(y, y_predict)
print(f"r2 = {r2}")


#Plotting

line_x = np.linspace(X.min(), X.max()) 
line_y = m * line_x + b

predict_x = 33
predict_y = m * predict_x + b

plt.scatter(X, y)
plt.scatter(predict_x, predict_y, color = 'black')
plt.plot(line_x, line_y, color = 'red')
plt.show()