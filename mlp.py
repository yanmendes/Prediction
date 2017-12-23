from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import BernoulliRBM
from sklearn.utils import check_array
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from pandas.tseries.offsets import BDay
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

def MAPE(y_true, y_pred):
	errors = list()
	for i in range(len(y_true)):
		errors.append(abs(y_true[i] - y_pred[i]/y_true[i]))
	return np.mean(errors)

# Loading Data
df = pd.read_csv("ts.csv", header=0)

# Indexing the data
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
del df['date']

for t in range (1, 4):
	with open('./{}/NN_{}.txt'.format(5*t, 5*t), 'wb') as output_file:
		# Pre-processing
		# Getting only business days
		df = df[df.index.weekday < 5]

		# Resampling to aggregate
		df = df.resample('{}Min'.format(5 * t)).sum().fillna(0)

		# Highest values of the series
		df1 = df.between_time('7:00','20:00')
		df1 = abs(df1 - df1.mean()) / (df1.max() - df1.min())

		# lowest values of the series
		df2 = df.between_time('20:01','6:59')
		df2 = abs(df2 - df2.mean()) / (df2.max() - df2.min())

		for p in range(3, 16):
			for n in random.sample(xrange(40, 100), 10):
				print 'Running for params P = {}, N = {}'.format(p, n)
				print 'Pre-processing...'

				output_file.write('Running for params P = {}, N = {}\n'.format(p, n))

				# Initializing the data data
				X1 = list()
				X2 = list()
				Y1 = list()
				Y2 = list()

				for i in range(len(df1) - p):
					X1.append(df1['count'][i:(i + p)])
					Y1.append(df1['count'][i + p])

				for i in range(len(df2) - p):
					X2.append(df2['count'][i:(i + p)])
					Y2.append(df2['count'][i + p])
				
				print '   Splitting in train-test...'
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

				print '   Initializing the models...'
				# Initializing the models
				MLP1 = MLPRegressor(hidden_layer_sizes=n)
				MLP1transformed = MLPRegressor(hidden_layer_sizes=n)
				MLP2 = MLPRegressor(hidden_layer_sizes=n)
				MLP2transformed = MLPRegressor(hidden_layer_sizes=n)

				# Grid search to find best params
				# regressor = GridSearchCV(MLP, dict(rbm__learning_rate=np.logspace(0., -3., num=5), rbm__n_components=random.sample(range(100, 1000), 5), logistic__C=random.sample(range(1, 10000), 5)))
				# regressor.fit(X_train, Y_train)
				# print(regressor.best_params_)

				results1 = list()
				results2 = list()
				print 'Running tests...'
				for test in range(0, 30):
					if(test % 6 == 5):
						print 'T = {}%'.format(int(((test + 1)*100)/30))
					MLP1.fit(X1_train, Y1_train)
					predicted1 = MLP1.predict(X1_test)
					MLP2.fit(X2_train, Y2_train)
					predicted2 = MLP2.predict(X2_test)
					results1.append(MAPE(Y1_test, predicted1))
					results2.append(MAPE(Y2_test, predicted2))

				output_file.write('Results for 07:00-20:00\n')
				output_file.write('Min: {}\n'.format(min(results1)))
				output_file.write('Avg MAPE: {}\n'.format(np.mean(results1)))

				output_file.write('Results for 20:01-06:59\n')
				output_file.write('Min: {}\n'.format(min(results2)))
				output_file.write('Avg MAPE: {}\n\n'.format(np.mean(results2)))
				output_file.flush()

				print '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -'
			print '> > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >'