import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

#GENERATOR FOR COMPARISON

if(True):
	comp = pandas.read_csv('comp_pems.csv', header=0)

	X = range(0, len(comp['LOGI']))
	nn = list()
	rbm = list()
	relu = list()

	for i in xrange(0, len(comp['LOGI']), 10):
		nn.append(np.mean(comp['LOGI'][i:i+10]))
		relu.append(np.mean(comp['RELU'][i:i+10]))
		rbm.append(np.mean(comp['RBM'][i:i+10]))

	f, (ax1, ax2) = plt.subplots(1, 2)
	ax1.set_title('All evaluations')
	ax2.set_title('Average evaluation')
	ax1.set_xlabel('Test #')
	ax1.set_ylabel('MAPE')
	ax2.set_xlabel('Test #')
	ax1.scatter(X, comp['LOGI'], color='blue', marker='x', alpha=0.4, label='ST (Logistic)')
	ax1.scatter(X, comp['RELU'], color='#c8f442', marker='+', alpha=0.1, label='ST (ReLU)')
	ax1.scatter(X, comp['RBM'], color='red', marker='v', alpha=0.3, label='Huang et. al')
	ax2.scatter(range(0, len(nn)), nn, color='blue', marker='x', alpha=0.5, label='ST (Logistic)')
	ax2.scatter(range(0, len(relu)), relu, color='#c8f442', marker='+', alpha=0.1, label='ST (ReLU)')
	ax2.scatter(range(0, len(rbm)), rbm, color='red', marker='v', alpha=0.5, label='Huang et. al')
	ax2.legend()
	ax1.legend()

	f.savefig('comp.eps')

#GENERATOR FOR PARAMETER P

if (False):
	nn = pandas.read_csv('NN.csv', header=0)
	rbm = pandas.read_csv('RBM.csv', header=0)

	f, axarr = plt.subplots(1, sharey=True)
	axarr.set_title('SmartTraffic Parameters Evaluation')
	axarr.set_ylabel('MAPE')
	axarr.set_xlabel('P')
	df2 = nn.pivot(columns='P', values='Avg Mape')
	df2.boxplot(ax=axarr)
	plt.savefig('nn-p.eps')

	f, axarr = plt.subplots(1, sharey=True)
	axarr.set_title('Huang et. al Parameters Evaluation')
	axarr.set_ylabel('MAPE')
	axarr.set_xlabel('P')
	df2 = rbm.pivot(columns='P', values='Avg Mape')
	df2.boxplot(ax=axarr)
	plt.savefig('rbm-p.eps')

#GENERATOR FOR PARAMETERS Q-N

if (False):
	nn = pandas.read_csv('NN.csv', header=0)
	rbm = pandas.read_csv('RBM.csv', header=0)

	f, axarr = plt.subplots(2, sharey=True, figsize=(8, 7))
	axarr[0].set_title('SmartTraffic Parameters Evaluation')
	axarr[0].set_ylabel('MAPE')
	axarr[1].set_ylabel('MAPE')
	axarr[0].set_xlabel('Q')
	axarr[1].set_xlabel('N')
	df2 = nn.pivot(columns='Q', values='Avg Mape')
	df2.boxplot(ax=axarr[0])
	df2 = nn.pivot(columns='N', values='Avg Mape')
	df2.boxplot(ax=axarr[1])
	plt.savefig('nn-q-n.eps')

	f, axarr = plt.subplots(2, sharey=True, figsize=(8, 7))
	axarr[0].set_title('Huang et. al Parameters Evaluation')
	axarr[0].set_ylabel('MAPE')
	axarr[1].set_ylabel('MAPE')
	axarr[0].set_xlabel('Q')
	axarr[1].set_xlabel('N')
	df2 = rbm.pivot(columns='Q', values='Avg Mape')
	df2.boxplot(ax=axarr[0])
	df2 = rbm.pivot(columns='N', values='Avg Mape')
	df2.boxplot(ax=axarr[1])
	plt.savefig('rbm-q-n.eps')
