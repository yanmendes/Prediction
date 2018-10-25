from sklearn import metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold
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
df = pd.read_csv("ts.csv", header=0)

MIN_T = 5
MAX_T = 15
STEP_T = 5
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

# Test/train split
kf = RepeatedKFold(n_splits=3, n_repeats=5)

for t in range(MIN_T, MAX_T + 1, STEP_T):
	with open('./{}/RBM.csv'.format(t, t), 'w', 1) as rbm_file:
		with open('./{}/NN.csv'.format(t, t), 'w', 1) as nn_file:
			# Reading CSV
			rbmwriter = csv.writer(rbm_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
			nnwriter  = csv.writer(nn_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

			# Writing results headers
			rbmwriter.writerow(['P', 'Q', 'N', 'H', 'Avg Mape', 'Min MAPE', 'Avg time'])
			nnwriter.writerow(['P', 'Q', 'N', 'H', 'Avg Mape', 'Min MAPE', 'R2', 'Avg time'])

			# Pre-processing

			# Resampling to aggregate in time windows of T minutes
			df = df.resample('{}Min'.format(t)).sum()

			# Changing n/a to 0
			df = df.fillna(0)			

			# Laplace smoothing
			df['count'] = df['count'] + 1

			# Using historic data (Q) from the same time and weekday
			for i in range (1, MAX_Q + 1):
				df['count-{}'.format(i)] = df['count'].shift(i * 7 * 24 * 60 // t)

			# Change n/a to 1
			df = df.fillna(1)

			# Getting only business days
			df_weekday = df[df.index.weekday < 5]
			df_weekend = df[df.index.weekday >= 5]
			
			# Splitting the two networks
			df1 = df_weekday.between_time('7:00','20:00')
			df2 = df_weekend.between_time('7:00','20:00')

			# Normalizing the data
			df1_max = max(df1['count'])
			df1_min = min(df1['count'])
			df1['count'] = df1['count'] / (df1_max - df1_min)

			df2_max = max(df2['count'])
			df2_min = min(df2['count'])
			df2['count'] = df2['count'] / (df2_max - df2_min)

			for i in range (1, MAX_Q + 1):
				df1['count-{}'.format(i)] = df1['count-{}'.format(i)] / (df1_max - df1_min)
				df2['count-{}'.format(i)] = df2['count-{}'.format(i)] / (df2_max - df2_min)

			for p in range(MIN_P, MAX_P + 1, STEP_P):
				for q in range(MIN_Q, MAX_Q + 1, STEP_Q):
					aux_df1 = df1
					aux_df2 = df2

					# Shifiting the data set by Q weeks
					df1 = df1[q * (5 * 13 * 60 // t + 5):]
					df2 = df2[q * (2 * 13 * 60 // t + 5):]

					for n in range(MIN_N, MAX_N + 1, STEP_N):
						print('Running for params P = {}, Q = {}, N = {}'.format(p, q, n))
						print('Pre-processing...')

						# Initializing the data
						X1 = list()
						Y1 = list()
						X2 = list()
						Y2 = list()

						# Mapping each set of variables (P and Q) to their correspondent value
						for i in range(len(df1) - p - 1):
							X = list()
							for j in range (1, q + 1):
								X.append(df1['count-{}'.format(j)][i + p + 1])
							X1.append(X + list(df1['count'][i:(i + p)]))
							Y1.append(df1['count'][i + p + 1])

						for i in range(len(df2) - p - 1):
							X = list()
							for j in range (1, q + 1):
								X.append(df2['count-{}'.format(j)][i + p + 1])
							X2.append(X + list(df2['count'][i:(i + p)]))
							Y2.append(df2['count'][i + p + 1])

						print('   Initializing the models...')
						# Initializing the models
						MLP1 = MLPRegressor(hidden_layer_sizes=(n, n, n,), activation='relu')
						regressor1 = Pipeline(steps=[('rbm1', BernoulliRBM(verbose=False, n_components=n)),
 									     ('SVR', SVR())])

						MLP2 = MLPRegressor(hidden_layer_sizes=(n, n, n,), activation='relu')
						regressor2 = Pipeline(steps=[('rbm2', BernoulliRBM(verbose=False, n_components=n)),
 									     ('SVR', SVR())])

						results_nn1 = list()
						results_rbm1 = list()
						results_r21 = list()

						avg_mlp_time1 = 0
						avg_rbm_time1 = 0

						results_nn2 = list()
						results_rbm2 = list()
						results_r22 = list()

						avg_mlp_time2 = 0
						avg_rbm_time2 = 0

						X1 = np.asarray(X1)
						X2 = np.asarray(X2)
						Y1 = np.asarray(Y1)
						Y2 = np.asarray(Y2)

						print('   Running weekday tests...')
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

						print('   Running weekend tests...')
						for train, test in kf.split(X2):
							X2_train, X2_test, Y2_train, Y2_test = X2[train], X2[test], Y2[train], Y2[test]

							start_time = time.time()
							MLP2.fit(X2_train, Y2_train)
							predicted2_nn = MLP2.predict(X2_test)
							avg_mlp_time2 = avg_mlp_time2 + time.time() - start_time
							results_nn2.append(MAPE(Y2_test, predicted2_nn))
							
							start_time = time.time()
							regressor2.fit(X2_train, Y2_train)
							predicted2_rbm = regressor2.predict(X2_test)
							avg_rbm_time2 = avg_rbm_time2 + time.time() - start_time
							results_rbm2.append(MAPE(Y2_test, predicted2_rbm))

							results_r22.append(RSquared(Y2_test, predicted2_nn, predicted2_rbm))

						nnwriter.writerow([p, q, n, 2, np.mean(results_nn2), min(results_nn2), np.mean(results_r22), avg_mlp_time2 / 15])
						rbmwriter.writerow([p, q, n, 2, np.mean(results_rbm2), min(results_rbm2), avg_rbm_time2 / 15])

						print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
					print('> > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >')
					df1 = aux_df1
					df2 = aux_df2
