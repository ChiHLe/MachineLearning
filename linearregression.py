__author__ = 'Chi Le'

# Linear Regression for salary prediction based on features
# Features: sex, current position, years in position, highest degree, years since highest degree
# w = (F^T F)^-1 F^T t, F = input matrix, t = output vector, T indicates transpose
# Root mean square error on training set 2233.63400334, average percent error 6.34713543942

import numpy as np

data = []

# Read the data set and add each line to data
with open('salary.txt') as file:
    header = file.readline().split()
    for l in file:
        line = l.split()
        # first term is the bias weight
        row = [1]
        # categorical variable into numerical
        row += ([1] if line[0] == 'female' else [0])
        row += ([3] if line[1] == 'full' else ([2] if line[1] == 'associate' else [1]))
        row += [float(line[2])]
        row += ([1] if line[3] == 'doctorate' else [0])
        row += [float(i) for i in line[4:]]
        data.append(row)

# Create numpy arrays
# features array
input = np.array([line[0:-1] for line in data])
# salary vector
output = np.array([line[-1] for line in data])

# find the weight vector
weight = np.linalg.inv(np.dot(input.T, input)).dot(np.dot(input.T, output))

# print linear regression model
print("sex, current position, years in position, highest degree, years since highest degree")
print(weight)

# calculate errors (root mean square and percent error)
pred = []
sq_error = 0
per_error = 0

# Calculate error for each data point
for i in range(len(data)):
    pred.append(weight.dot(input[i]))
    sq_error += (pred[i] - output[i])**2
    per_error += abs((pred[i] - output[i])/output[i])

avg_pe = (per_error/len(data))*100
rmse = (sq_error/len(data))**0.5

# Print out error
print("Root mean square error " + str(rmse))
print("Average percent error " + str(avg_pe))