from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import BernoulliRBM
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from pandas.tseries.offsets import BDay
from compiler.ast import flatten
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import collections
import math

def RMSD(y_true, y_pred):
	errors = 0
	print y_true[:6], y_pred[:6]
	exit()
	for i in range(len(y_true)):
		errors = errors + math.sqrt(math.pow((y_true[i] - y_pred[i]), 2))
	return errors / len(y_pred)

def MAPE(y_true, y_pred):
	errors = 0
	for i in range(len(y_true)):
		errors = errors + abs((y_true[i] - y_pred[i]) / y_true[i])
	return errors / len(y_pred)

# Loading Data
df = pd.read_csv("pems.csv", header=0)

MIN_P = 5
MAX_P = 15
STEP_P = 5
MIN_Q = 3
MAX_Q = 5
STEP_Q = 1
MIN_N = 60
MAX_N = 120
STEP_N = 10

# Indexing the data
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
del df['date']

with open('./PEMS/RBM.txt', 'wb') as rbm_file:
	with open('./PEMS/NN.txt', 'wb') as nn_file:
		# Pre-processing
		# Getting only business days
		df = df[df.index.weekday < 5]

		# Using historic data from the same weekday
		for i in range (1, MAX_Q + 1):
			df['count-{}'.format(i)] = df['count'].shift(7 * 24 * i)

		df1 = df.fillna(0)

		# Highest values of the series
		df1_max = df1.max()['count']
		df1_min = df1.min()['count']
		df1 = abs(df1 - df1.min()) / (df1.max() - df1.min())

		for p in xrange(MIN_P, MAX_P + 1, STEP_P):
			for q in xrange(MIN_Q, MAX_Q + 1, STEP_Q):
				for n in xrange(MIN_N, MAX_N + 1, STEP_N):
					print('Running for params P = {}, Q = {}, N = {}'.format(p, q, n))
					print('Pre-processing...')

					nn_file.write('Running for params P = {}, Q = {}, N = {}\n'.format(p, q, n))
					rbm_file.write('Running for params P = {}, Q = {}, N = {}\n'.format(p, q, n))

					# Initializing the data data
					X1 = list()
					Y1 = list()

					for i in range(len(df1) - p):
						X = list()
						for j in range (1, MAX_Q + 1):
							X.append(df1['count-{}'.format(j)][i])
						X1.append(flatten(X + flatten(df1['count'][i:(i + p)])))
						Y1.append(df1['count'][i + p])
					
					print('   Splitting in train-test...')
					# Train/test/validation split
					rows1 = random.sample(range(len(X1)), int(len(X1)/3))

					X1_test = [X1[j] for j in rows1]
					Y1_test = [Y1[j] for j in rows1]
					X1_train = [X1[j] for j in list(set(range(len(X1))) - set(rows1))]
					Y1_train = [Y1[j] for j in list(set(range(len(Y1))) - set(rows1))]

					# Denormalizing the data
					for i in range(0, len(Y1_test)):
						Y1_test[i] = Y1_test[i] * (df1_max - df1_min) + df1_min

					print('   Initializing the models...')
					# Initializing the models
					MLP1 = MLPRegressor(hidden_layer_sizes=n, activation='logistic')
					SVR1 = SVR()
					RBM1 = BernoulliRBM(verbose=False, n_components=n)
					regressor1 = Pipeline(steps=[('rbm', RBM1), ('SVR', SVR1)])

					results_nn1 = list()
					results_rbm1 = list()
					print('Running tests...')
					for test in range(0, 30):
						if(test % 6 == 5):
							print('T = {}%'.format(int(((test + 1)*100)/30)))
						MLP1.fit(X1_train, Y1_train)
						predicted1 = list(MLP1.predict(X1_test))

						for i in range(0, len(predicted1)):
							predicted1[i] = predicted1[i] * (df1_max - df1_min) + df1_min

						results_nn1.append(MAPE(Y1_test, predicted1))
						
						for i in range(0, 9):
							plt.plot(Y1_test[i * 100: i * 100 + 100], color='black')
							plt.plot(predicted1[i * 100:i * 100 + 100], color='red')
							plt.savefig('./PEMS/{}_{}_{}_{}_{}.eps'.format(p, q, n, test, i), figsize=(8, 4))
							plt.gcf().clear()

						regressor1.fit(X1_train, Y1_train)
						predicted1 = regressor1.predict(X1_test)
						results_rbm1.append(MAPE(Y1_test, predicted1))

					nn_file.write('Min: {}\n'.format(min(results_nn1)))
					nn_file.write('Avg MAPE: {}\n'.format(np.mean(results_nn1)))
					nn_file.flush()

					rbm_file.write('Min: {}\n'.format(min(results_rbm1)))
					rbm_file.write('Avg MAPE: {}\n'.format(np.mean(results_rbm1)))
					rbm_file.flush()
					print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
				print('> > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >')