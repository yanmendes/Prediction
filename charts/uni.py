import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

comp = pandas.read_csv('comp-5.csv', header=0)

#plt.hist([comp['NN'], comp['RBM']], histtype='bar', color=['red', 'black'], label=['MLP', 'RBM'])
#plt.legend(prop={'size': 10})
#plt.title('Models comparison')
#plt.xlabel('MAPE')
#plt.ylabel('Number of occurences')

#plt.savefig('comp.eps')

#exit()

X = range(0, len(comp['NN']))
nn = list()
rbm = list()

for i in xrange(0, len(comp['NN']), 10):
	nn.append(np.mean(comp['NN'][i:i+10]))
	rbm.append(np.mean(comp['RBM'][i:i+10]))

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title('All evaluations')
ax2.set_title('Average evaluation')
ax1.set_xlabel('Test #')
ax1.set_ylabel('MAPE')
ax2.set_xlabel('Test #')
ax1.scatter(X, comp['NN'], color='blue', marker='x', alpha=0.4, label='MLP')
ax1.scatter(X, comp['RBM'], color='#ffcf11', marker='v', alpha=0.3, label='RBM')
ax2.scatter(range(0, len(nn)), nn, color='blue', marker='x', alpha=0.5, label='MLP')
ax2.scatter(range(0, len(rbm)), rbm, color='#ffcf11', marker='v', alpha=0.5, label='RBM')
ax2.legend()
ax1.legend()

f.savefig('comp.eps')

exit()

nn = list()
rbm = list()

plt.scatter(X, comp['NN'], color='blue', marker='x', alpha=0.5)
plt.scatter(X, comp['RBM'], color='red', marker='v', alpha=0.5)

plt.savefig('comp.png')

exit()

nn = pandas.read_csv('NN.csv', header=0)
rbm = pandas.read_csv('RBM.csv', header=0)

f, axarr = plt.subplots(2, sharey=True)
axarr[0].set_title('MLP Evaluation')
axarr[0].set_ylabel('MAPE')
axarr[1].set_ylabel('MAPE')
axarr[0].set_xlabel('Q')
axarr[1].set_xlabel('N')
df2 = nn.pivot(columns='Q', values='Avg Mape')
df2.boxplot(ax=axarr[0])
df2 = nn.pivot(columns='N', values='Avg Mape')
df2.boxplot(ax=axarr[1])
plt.savefig('nn.eps')

f, axarr = plt.subplots(2, sharey=True)
axarr[0].set_title('MLP Evaluation')
axarr[0].set_ylabel('MAPE')
axarr[1].set_ylabel('MAPE')
axarr[0].set_xlabel('Q')
axarr[1].set_xlabel('N')
df2 = rbm.pivot(columns='Q', values='Avg Mape')
df2.boxplot(ax=axarr[0])
df2 = rbm.pivot(columns='N', values='Avg Mape')
df2.boxplot(ax=axarr[1])
plt.savefig('rbm.eps')

exit()

f, axarr = plt.subplots(2, sharey=True)
axarr[0].boxplot(nn['q'].values(), 0, positions=nn['q'].keys())
axarr[0].set_title("Evaluation of MLP")
axarr[0].set_xlabel("Q")
axarr[1].set_xlabel("N")
axarr[0].set_ylabel("MAPE")
axarr[1].set_ylabel("MAPE")
axarr[1].boxplot(nn['n'].values(), 0, positions=nn['n'].keys())

f.savefig('nn.eps')

f, axarr = plt.subplots(2, sharey=True)
axarr[0].boxplot(rbm['q'].values(), 0, positions=rbm['q'].keys())
axarr[0].set_title("Evaluation of RBM")
axarr[0].set_xlabel("Q")
axarr[1].set_xlabel("N")
axarr[0].set_ylabel("MAPE")
axarr[1].set_ylabel("MAPE")
axarr[1].boxplot(rbm['n'].values(), 0, positions=rbm['n'].keys())

f.savefig('rbm.eps')
