from QCNN import QCNN,get_random_computational_ids,convert_id_to_state
from torch.optim import Adam, SGD, RMSprop, lr_scheduler
import torch
import torch.nn as nn
import numpy as np
from state_simulator import state_simulator, state_mixture_simulator, conjugate_transpose
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title 
params = {'axes.labelsize': MEDIUM_SIZE,'axes.titlesize':BIGGER_SIZE, 'legend.fontsize': SMALL_SIZE, 'xtick.labelsize': SMALL_SIZE, 'ytick.labelsize': SMALL_SIZE}
sns.set_context("paper", rc=params)
matplotlib.rcParams.update(params)


def plot_surface(x,y,z, save_name = 'surface_plot_test', normalize = True):
	if normalize:
		z = z/np.max(z)
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	ax.plot_surface(x,y,z,cmap = plt.cm.viridis,
					antialiased=False, shade = True, alpha = 0.5, vmin = 0.0,vmax = 0.8)

	fig = plt.gcf()
	fig.set_size_inches(3.0,2.5)
	fig.tight_layout()
	plt.grid(False)
	plt.axis('off')
	directory = './figures/3d_plots/'+save_name
	if not os.path.isdir(directory):
		os.mkdir(directory)

	for a1 in range(0,360,30):
		for a2 in range(-30,40,10):
			ax.view_init(a2,a1)
			plt.savefig(directory+'/'+save_name+'_'+str(a1)+'_'+str(a2)+'.pdf')

def plot_contour(x,y,z, save_name = 'surface_plot_test', normalize = True):
	if normalize:
		z = z/np.max(z)
	# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	fig = plt.figure()
	ax = plt.contourf(x,y,z,100,cmap = plt.cm.viridis, alpha = 0.8, vmin = 0.0,vmax = 0.8)
	plt.colorbar()

	fig = plt.gcf()
	fig.set_size_inches(3.0,2.5)
	fig.tight_layout()
	# plt.grid(False)
	plt.axis('off')
	directory = './figures/3d_plots/'+save_name
	if not os.path.isdir(directory):
		os.mkdir(directory)
	plt.savefig(directory+'/'+save_name+'_contour.pdf')


def QCNN_main(save_name = '3d_data_qcnn', random_initialization = False):
	n = 14
	batch_size = 128
	device = 'cuda:0'

	target_QCNN = QCNN(n).to(device)


	data_ids = get_random_computational_ids(n, batch_size)
	data = convert_id_to_state(data_ids, n).to(device)
	with torch.no_grad():
		state = state_mixture_simulator(n, state_init = data, device = device)
		y_tar = target_QCNN(state)

	optim = Adam(target_QCNN.parameters())
	criterion = nn.MSELoss()	

	def get_loss():
		data = convert_id_to_state(data_ids, n).to(device)
		state = state_mixture_simulator(n, state_init = data, device = device)
		return criterion(target_QCNN(state), y_tar)

	def get_direction(k):
		direction = torch.randn_like(target_QCNN.params[k])
		direction = direction - conjugate_transpose(direction)
		direction = direction / torch.sqrt(torch.sum(direction*direction.conj()))
		return direction

	if random_initialization:
		target_QCNN = QCNN(n).to(device)
	pdict = ['H0','H1','H2']
	dir1 = [get_direction(k) for k in pdict]
	dir1 = dict(zip(pdict,dir1))
	dir2 = [get_direction(k) for k in pdict]
	dir2 = dict(zip(pdict,dir2))
	start = [torch.clone(target_QCNN.params[k]) for k in pdict]
	start = dict(zip(pdict,start))

	grid_size = 128
	data = np.zeros((grid_size,grid_size))
	xs = np.linspace(-8,8,grid_size)
	ys = np.linspace(-8,8,grid_size)
	x_grid, y_grid = np.meshgrid(xs, ys)
	for i,x in enumerate(xs):
		print('i = ' + str(i))
		for j,y in enumerate(ys):
			for k in pdict:
				target_QCNN.params[k].data = start[k]+x*dir1[k]+y*dir2[k]
			print(get_loss().item())
			data[i,j] = get_loss().item()


	with open('./data/'+save_name+'.npz', 'wb') as f:
		np.savez(f, data, xs, ys)

	return data, xs, ys
	
def load_data(save_name):
	# with open('./data/'+save_name+'.npz', 'rb') as f:
	# 	npz = np.load(f)
	npz = np.load('./data/'+save_name+'.npz')
	return npz['arr_0'], npz['arr_1'], npz['arr_2']



if __name__ == '__main__':
	data, x_grid, y_grid = QCNN_main(save_name = '3d_data_qcnn', random_initialization = False)
	data, x_grid, y_grid = load_data('3d_data_qcnn')
	plot_surface(x_grid,y_grid,data,save_name = 'surface_plot_test')
	plot_contour(x_grid,y_grid,data,save_name = 'surface_plot_test')

	data, x_grid, y_grid = QCNN_main(save_name = '3d_data_qcnn_random_init', random_initialization = True)
	data, x_grid, y_grid = load_data('3d_data_qcnn_random_init')
	plot_surface(x_grid,y_grid,data,save_name = 'surface_plot_test_init')
	plot_contour(x_grid,y_grid,data,save_name = 'surface_plot_test_init')
