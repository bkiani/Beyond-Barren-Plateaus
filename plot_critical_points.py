import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

params = {'axes.labelsize': MEDIUM_SIZE,'axes.titlesize':BIGGER_SIZE, 'legend.fontsize': SMALL_SIZE, 'xtick.labelsize': SMALL_SIZE, 'ytick.labelsize': SMALL_SIZE}
sns.set_context("paper", rc=params)


def get_distribution(x ,m = 128, p = 4):
	y = np.exp(-m*x) * (x**(m-p/2)) * (1-2*x)**p
	y = y / np.sum(y)
	# y = y / np.max(y)
	return y

def make_plot(save_name = 'critical_points.pdf'):
	sns.set_theme(style="whitegrid", palette="muted")
	sns.set_context("paper", rc=params)

	f, a0 = plt.subplots(1,1)
	colors = sns.color_palette("rocket")

	x = np.linspace(0,0.5,200)
	sns.lineplot(x, get_distribution(x,128,4), color = colors[0], alpha = 0.75)
	sns.lineplot(x, get_distribution(x,128,32), color = colors[2], alpha = 0.75)
	sns.lineplot(x, get_distribution(x,128,256),  color = colors[-1], alpha = 0.75)

	a0.fill_between(x,get_distribution(x,128,4), color=colors[0], alpha=0.35)
	a0.fill_between(x,get_distribution(x,128,32), color=colors[2], alpha=0.35)
	a0.fill_between(x,get_distribution(x,128,256), color=colors[-1], alpha=0.35)

	text_size = SMALL_SIZE
	a0.text(0.5,0.078,r"very underparameterized $l \ll 2m$", color = colors[0], horizontalalignment = 'right', fontsize = text_size, fontweight = 'bold')
	a0.text(0.4,0.05,r"underparameterized $l < 2m$", color = colors[2], horizontalalignment = 'right', fontsize = text_size, fontweight = 'bold')
	a0.text(0.01,0.125,r"overparameterized $l = 2m$", color = colors[-1], horizontalalignment = 'left', fontsize = text_size, fontweight = 'bold')

	a0.set_xlabel('Energy (Normalized)')
	a0.set_ylabel(r"Local Minima Density $Crt_0(E)$")
	a0.set_yticklabels([])
	a0.set_ylim([0,0.15])

	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(3.5,3.)
	fig.tight_layout()
	plt.savefig('./figures/'+save_name)


if __name__ == '__main__':
	make_plot()