import re
import numpy as np
import matplotlib.pyplot as plt
from compiler.ast import flatten

rbm = dict()
nn = dict()

rbm['p'] = dict()
rbm['q'] = dict()
rbm['n'] = dict()

nn['p'] = dict()
nn['q'] = dict()
nn['n'] = dict()

with open('RBM.txt', 'r') as f:
	for line in f:
		if "Min" in line:
			continue
		elif "params" in line:
			m = re.match(r".* P = (\d+).* Q = (\d+).* N = (\d+)", line, re.M|re.I)
			p = int(m.group(1))
			q = int(m.group(2))
			n = int(m.group(3))
			if (p not in rbm['p']):
				rbm['p'][p] = list()
			if (q not in rbm['q']):
				rbm['q'][q] = list()
			if (n not in rbm['n']):
				rbm['n'][n] = list()
		else:
			rbm['p'][p].append(float(line[10:-1]))
			rbm['q'][q].append(float(line[10:-1]))
			rbm['n'][n].append(float(line[10:-1]))

	f.close()

with open('NN.txt', 'r') as f:
	for line in f:
		if "Min" in line:
			continue
		elif "params" in line:
			m = re.match(r".* P = (\d+).* Q = (\d+).* N = (\d+)", line, re.M|re.I)
			p = int(m.group(1))
			q = int(m.group(2))
			n = int(m.group(3))
			if (p not in nn['p']):
				nn['p'][p] = list()
			if (q not in nn['q']):
				nn['q'][q] = list()
			if (n not in nn['n']):
				nn['n'][n] = list()
		else:
			nn['p'][p].append(float(line[10:-1]))
			nn['q'][q].append(float(line[10:-1]))
			nn['n'][n].append(float(line[10:-1]))

	f.close()

#FAZER SOLO PRA P E SUBPLOT PRA N, Q E FALAR JUNTO NO ARTIGO DE N E Q

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
