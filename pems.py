from sklearn import metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold
from pandas.tseries.offsets import BDay
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import collections
import math
import csv
import time

def MAPE(y_true, y_pred):
	errors = 0
	for i in range(len(y_true)):
		errors = errors + (abs((float(y_true[i]) - float(y_pred[i]))) / float(y_true[i]))
	return errors / len(y_pred)

def RSquared(y_true, y_pred_nn, y_pred_rbm):
	SStot = 0
	SSres = 0

	for i in range(len(y_true)):
		SStot = SStot + (float(y_true[i]) - float(y_pred_rbm[i]))**2
		SSres = SSres + (float(y_true[i]) - float(y_pred_nn[i]))**2

	return 1 - SSres/SStot

# Loading Data
df = pd.read_csv("pems.csv", header=0)

kf = RepeatedKFold(n_splits=3, n_repeats=5)

MIN_P = 10
MAX_P = 15
STEP_P = 1
MIN_Q = 0
MAX_Q = 5
STEP_Q = 1
MIN_N = 80
MAX_N = 120
STEP_N = 10

# Indexing the data
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
del df['date']

with open('RBM_PEMS.csv', 'w') as rbm_file:
	with open('NN_PEMS.csv', 'w') as nn_file:
		# Reading CSV
		rbmwriter = csv.writer(rbm_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
		nnwriter  = csv.writer(nn_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

		# Writing results headers
		rbmwriter.writerow(['P', 'Q', 'N', 'H', 'Avg Mape', 'Min MAPE', 'Avg time'])
		nnwriter.writerow(['P', 'Q', 'N', 'H', 'Avg Mape', 'Min MAPE', 'R2', 'Avg time'])

		# Pre-processing

		# Using historic data (Q) from the same time and weekday
		for i in range (1, MAX_Q + 1):
			df['count-{}'.format(i)] = df['count'].shift(i * 7 * 24)

		# Change n/a to 1
		df = df.fillna(0)

		# Normalizing the data
		df_max = max(df['count'])
		df_min = min(df['count'])
		df['count'] = df['count'] / (df_max - df_min)

		for i in range (1, MAX_Q + 1):
			df['count-{}'.format(i)] = df['count-{}'.format(i)] / (df_max - df_min)

		for p in range(MIN_P, MAX_P + 1, STEP_P):
			for q in range(MIN_Q, MAX_Q + 1, STEP_Q):
				aux_df = df

				# Shifiting the data set by Q weeks
				df = df[q * 7 * 24:]

				for n in range(MIN_N, MAX_N + 1, STEP_N):
					print('Running for params P = {}, Q = {}, N = {}'.format(p, q, n))
					print('Pre-processing...')

					# Initializing the data
					X1 = list()
					Y1 = list()

					# Mapping each set of variables (P and Q) to their correspondent value
					for i in range(len(df) - p - 1):
						X = list()
						for j in range (1, q + 1):
							X.append(df['count-{}'.format(j)][i + p + 1])
						X1.append(X + list(df['count'][i:(i + p)]))
						Y1.append(df['count'][i + p + 1])

					print('   Initializing the models...')
					# Initializing the models
					MLP1 = MLPRegressor(hidden_layer_sizes=(n, n, n,), activation='logistic')
					regressor1 = Pipeline(steps=[('rbm1', BernoulliRBM(verbose=False, n_components=n)),
								     ('SVR', SVR())])

					results_nn1 = list()
					results_rbm1 = list()
					results_r21 = list()

					avg_mlp_time1 = 0
					avg_rbm_time1 = 0

					X1 = np.asarray(X1)
					Y1 = np.asarray(Y1)

					print('Running tests...')
					for train, test in kf.split(X1):
						X1_train, X1_test, Y1_train, Y1_test = X1[train], X1[test], Y1[train], Y1[test]

						start_time = time.time()
						MLP1.fit(X1_train, Y1_train)
						predicted1_nn = MLP1.predict(X1_test)
						avg_mlp_time1 = avg_mlp_time1 + time.time() - start_time
						results_nn1.append(MAPE(Y1_test, predicted1_nn))
						
						start_time = time.time()
						regressor1.fit(X1_train, Y1_train)
						predicted1_rbm = regressor1.predict(X1_test)
						avg_rbm_time1 = avg_rbm_time1 + time.time() - start_time
						results_rbm1.append(MAPE(Y1_test, predicted1_rbm))

						results_r21.append(RSquared(Y1_test, predicted1_nn, predicted1_rbm))

					nnwriter.writerow([p, q, n, 1, np.mean(results_nn1), min(results_nn1), np.mean(results_r21), avg_mlp_time1 / 15])
					rbmwriter.writerow([p, q, n, 1, np.mean(results_rbm1), min(results_rbm1), avg_rbm_time1 / 15])
					print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
				print('> > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >')
				df = aux_df
