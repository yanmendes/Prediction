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

def MAPE(y_true, y_pred):
	errors = 0
	for i in range(len(y_true)):
		errors = errors + abs(y_true[i] - y_pred[i]/y_true[i])
	return 100 * errors / len(y_pred)

# Loading Data
df = pd.read_csv("ts.csv", header=0)

MIN_P = 5
MAX_P = 15
STEP_P = 5
MIN_Q = 3
MAX_Q = 5
STEP_P = 1
STEP_N = 5

# Indexing the data
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
del df['date']

for t in range (1, 4):
	with open('./{}/RBM_{}.txt'.format(5*t, 5*t), 'wb') as rbm_file:
		with open('./{}/NN_{}.txt'.format(5*t, 5*t), 'wb') as nn_file:
			# Pre-processing
			# Getting only business days
			df = df[df.index.weekday < 5]

			# Resampling to aggregate
			df = df.resample('{}Min'.format(5 * t)).sum()

			# Using historic data from the same weekday
			for i in range (1, MAX_Q + 1):
				df['count-{}'.format(i)] = df['count'].shift(7 * 24 * (60 / (5 * t)) * i)

			df = df.fillna(1)

			# Highest values of the series
			df1 = df.between_time('7:00','20:00')
			df1_max = df1.max()['count']
			df1_min = df1.min()['count']
			df1 = abs(df1 - df1.min()) / (df1.max() - df1.min())

			# lowest values of the series
			df2 = df.between_time('20:01','6:59')
			df2_max = df2.max()['count']
			df2_min = df2.min()['count']
			df2 = abs(df2 - df2.min()) / (df2.max() - df2.min())

			for p in range(MIN_P, MAX_P + 1):
				for q in range(MIN_Q, MAX_Q + 1):
					for n in range(60, 81):
						print('Running for params P = {}, Q = {}, N = {}'.format(p, q, n))
						print('Pre-processing...')

						nn_file.write('Running for params P = {}, Q = {}, N = {}\n'.format(p, q, n))
						rbm_file.write('Running for params P = {}, Q = {}, N = {}\n'.format(p, q, n))

						# Initializing the data data
						X1 = list()
						X2 = list()
						Y1 = list()
						Y2 = list()

						for i in range(len(df1) - p):
							X = list()
							for j in range (1, MAX_Q + 1):
								X.append(df1['count-{}'.format(j)][i])
							X1.append(flatten(X + flatten(df1['count'][i:(i + p)])))
							Y1.append(df1['count'][i + p])

						for i in range(len(df2) - p):
							X = list()
							for j in range (1, MAX_Q + 1):
								X.append(df2['count-{}'.format(j)][i])
							X2.append(flatten(X + flatten(df2['count'][i:(i + p)])))
							Y2.append(df2['count'][i + p])
						
						print('   Splitting in train-test...')
						# Train/test/validation split
						rows1 = random.sample(range(len(X1)), int(len(X1)/3))
						rows2 = random.sample(range(len(X2)), int(len(X2)/3))

						X1_test = [X1[j] for j in rows1]
						X2_test = [X2[j] for j in rows2]
						Y1_test = [Y1[j] for j in rows1]
						Y2_test = [Y2[j] for j in rows2]
						X1_train = [X1[j] for j in list(set(range(len(X1))) - set(rows1))]
						X2_train = [X2[j] for j in list(set(range(len(X2))) - set(rows2))]
						Y1_train = [Y1[j] for j in list(set(range(len(Y1))) - set(rows1))]
						Y2_train = [Y2[j] for j in list(set(range(len(Y2))) - set(rows2))]

						# Denormalizing the data
						for i in range(0, len(Y1_test)):
							Y1_test[i] = int(Y1_test[i] * (df1_max - df1_min) + df1_min)

						for i in range(0, len(Y2_test)):
							Y2_test[i] = int(Y2_test[i] * (df2_max - df2_min) + df2_min)

						print('   Initializing the models...')
						# Initializing the models
						MLP1 = MLPRegressor(hidden_layer_sizes=n, activation='logistic')
						MLP2 = MLPRegressor(hidden_layer_sizes=n, activation='logistic')
						SVR1 = SVR()
						SVR2 = SVR()
						RBM1 = BernoulliRBM(verbose=False, n_components=n)
						RBM2 = BernoulliRBM(verbose=False, n_components=n)
						regressor1 = Pipeline(steps=[('rbm', RBM1), ('SVR', SVR1)])
						regressor2 = Pipeline(steps=[('rbm', RBM2), ('SVR', SVR2)])

						results_nn1 = list()
						results_nn2 = list()
						results_rbm1 = list()
						results_rbm2 = list()
						print('Running tests...')
						for test in range(0, 30):
							if(test % 6 == 5):
								print('T = {}%'.format(int(((test + 1)*100)/30)))
							MLP1.fit(X1_train, Y1_train)
							predicted1 = MLP1.predict(X1_test)
							MLP2.fit(X2_train, Y2_train)
							predicted2 = MLP2.predict(X2_test)

							predicted1 = list(predicted1)
							predicted2 = list(predicted2)

							for i in range(0, len(predicted1)):
								predicted1[i] = int(predicted1[i] * (df1_max - df1_min) + df1_min)

							for i in range(0, len(predicted2)):
								predicted2[i] = int(predicted2[i] * (df2_max - df2_min) + df2_min)

							results_nn1.append(MAPE(Y1_test, predicted1))
							results_nn2.append(MAPE(Y2_test, predicted2))
							plt.switch_backend('agg')
							for i in range(0, 9):
								plt.plot(Y1_test[i * 100: i * 100 + 100], color='black')
								plt.plot(predicted1[i * 100:i * 100 + 100], color='red')
								plt.savefig('./{}/{}_{}_{}_{}_{}.eps'.format(5*t, p, q, n, test, i), figsize=(8, 4	))
								plt.gcf().clear()

							regressor1.fit(X1_train, Y1_train)
							predicted1 = regressor1.predict(X1_test)
							regressor2.fit(X2_train, Y2_train)
							predicted2 = regressor2.predict(X2_test)
							results_rbm1.append(MAPE(Y1_test, predicted1))
							results_rbm2.append(MAPE(Y2_test, predicted2))

						nn_file.write('Results for 07:00-20:00\n')
						nn_file.write('Min: {}\n'.format(min(results_nn1)))
						nn_file.write('Avg MAPE: {}\n'.format(np.mean(results_nn1)))

						nn_file.write('Results for 20:01-06:59\n')
						nn_file.write('Min: {}\n'.format(min(results_nn2)))
						nn_file.write('Avg MAPE: {}\n\n'.format(np.mean(results_nn2)))
						nn_file.flush()

						rbm_file.write('Results for 07:00-20:00\n')
						rbm_file.write('Min: {}\n'.format(min(results_rbm1)))
						rbm_file.write('Avg MAPE: {}\n'.format(np.mean(results_rbm1)))

						rbm_file.write('Results for 20:01-06:59\n')
						rbm_file.write('Min: {}\n'.format(min(results_rbm2)))
						rbm_file.write('Avg MAPE: {}\n\n'.format(np.mean(results_rbm2)))
						rbm_file.flush()

						n = n + STEP_N - 1
						print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
					print('> > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >')
					q = q + STEP_Q - 1
				p = p + STEP_P - 1
