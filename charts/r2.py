import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

comp = pandas.read_csv('r2.csv', header=0)

#plt.hist([comp['NN'], comp['RBM']], histtype='bar', color=['red', 'black'], label=['MLP', 'RBM'])
#plt.legend(prop={'size': 10})
#plt.title('Models comparison')
#plt.xlabel('MAPE')
#plt.ylabel('Number of occurences')

#plt.savefig('comp.eps')

#exit()

L = ['ST (Logistic)', 'ST (ReLU)']
X = [comp['LOGI'], comp['RELU']]

fig, ax1 = plt.subplots()
ax1.set_title('R2 distribution over all runs')
ax1.set_xlabel('Approach')
ax1.set_ylabel('Value')
	
ax1.boxplot(X, labels=L)
fig.savefig('r2.eps')
