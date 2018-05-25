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

X = comp['R2']
nn = list()
rbm = list()

plt.boxplot(X)
plt.ylabel('R2 distribution over all runs')
plt.savefig('r2.eps')
