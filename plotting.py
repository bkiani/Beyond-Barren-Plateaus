import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os, glob


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
# matplotlib.rcParams.update(params)

def hist_and_swarm_plot(identifier, save_name = 'test.pdf', field = 'acc', xlabel = 'training accuracy', xlim = None, hue = None):
	sns.set_theme(style="whitegrid", palette="muted")
	sns.set_context("paper", rc=params)

	def get_data():
		files = glob.glob('./data/'+identifier)
		data = np.zeros(len(files))
		for i, file in enumerate(files):
			if i == 0:
				data = pd.read_csv(file)
				data = data.iloc[-1:,:]
			else:
				df = pd.read_csv(file)
				data = pd.concat([data, df.iloc[-1:,:]])
		data.reset_index(inplace = True)
		return data
	data = get_data()

	f, (a1, a0) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 5]},sharex = True)
	plt.subplots_adjust(wspace=0, hspace=0)
	sns.swarmplot(data = data, x = field, ax = a1, size=4, alpha = 0.5, hue = hue)
	sns.histplot(data = data, x = field, ax = a0, hue = hue) #kde = True 

	a0.set_xlabel(xlabel)
	if xlim is not None:
		a1.set_xlim(xlim)

	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(3.0,3.0)
	fig.tight_layout()
	plt.savefig('./figures/'+save_name)

def hist_and_swarm_plot_multiple(identifier, save_name = 'test.pdf', field = 'acc', y_field = 'qubits',
								 xlabel = 'training accuracy', xlim = None, hue = 'qubits', n_filter = None, order = None):
	sns.set_theme(style="whitegrid", palette="muted")
	sns.set_context("paper", rc=params)

	def get_data():
		files = glob.glob('./data/'+identifier)
		data = np.zeros(len(files))
		for i, file in enumerate(files):
			if i == 0:
				data = pd.read_csv(file)
				data = data.iloc[-1:,:]
			else:
				df = pd.read_csv(file)
				data = pd.concat([data, df.iloc[-1:,:]])
		data.reset_index(inplace = True)
		return data
	data = get_data()

	if n_filter is not None:
		data['filter'] = 0
		for n in n_filter:
			data['filter'] += (data['n_qubits'] ==n)
		data = data[data['filter']>0]

	data['qubits'] = 'n='+data['n_qubits'].astype('str')
	print(data['qubits'])


	# f, (a1, a0) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 5]},sharex = True)
	# plt.subplots_adjust(wspace=0, hspace=0)
	# print('here')
	# sns.swarmplot(data = data, x = field, ax = a1, size=4, alpha = 0.5, hue = hue)
	# print('here')
	# # sns.histplot(data = data, x = field, ax = a0, hue = hue, element="step") #kde = True 
	# sns.histplot(data = data, x = field, y = y_field, ax = a0, hue = hue) #kde = True 
	# print('here')
	# a0.set_xlabel(xlabel)
	# if xlim is not None:
	# 	a1.set_xlim(xlim)


	f, a0 = plt.subplots(1,1,)
	plt.subplots_adjust(wspace=0, hspace=0)
	sns.swarmplot(data = data, x = field, y = 'qubits', ax = a0, alpha = 0.5, size = 4., order = order)
	a0.set_xlabel(xlabel)
	if xlim is not None:
		a0.set_xlim(xlim)
	a0.set_ylabel('Number of Qubits')


	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(5.0,3.0)
	fig.tight_layout()
	plt.savefig('./figures/'+save_name)




def swarm_scatter_plot(identifier, save_name = 'scatter_test.pdf', x_field = 'loss', y_field = 'trace distance', L_filter = None, n_filter = None, leg_title = 'qubits',
						ylabel = 'Trace Distance', xlabel = 'Loss', xlim = None, hue = None, divide_loss = True):

	sns.set_theme(style="whitegrid", palette="muted")
	sns.set_context("paper", rc=params)

	def get_data():
		files = glob.glob('./data/'+identifier)
		data = np.zeros(len(files))
		for i, file in enumerate(files):
			if i == 0:
				data = pd.read_csv(file)
				data = data.iloc[-1:,:]
			else:
				df = pd.read_csv(file)
				data = pd.concat([data, df.iloc[-1:,:]])
		data.reset_index(inplace = True)
		return data

	data = get_data()
	if L_filter is not None:
		data['filter'] = 0
		for L in L_filter:
			data['filter'] += (data['layers_learn'] ==L)
		data = data[data['filter']>0]
	if n_filter is not None:
		data['filter'] = 0
		for n in n_filter:
			data['filter'] += (data['n_qubits'] ==n)
		data = data[data['filter']>0]


	if divide_loss:
		data['loss'] = data['loss']/data['n_qubits']

	# print(data['n_qubits'].unique())
	# print(data)
	data['trace distance'] = np.sqrt(data['ground_state_loss'])
	# data.to_csv('data.csv')

	f, a0 = plt.subplots(1,1)
	plt.subplots_adjust(wspace=0, hspace=0)
	palette = 'coolwarm' #'icefire' another option
	sns.scatterplot(data = data, x = x_field, y =y_field, hue = hue, alpha = 0.66,palette=palette, size = 0.25)
	sns.rugplot(data = data, x = x_field, y =y_field, hue = hue, alpha = 0.8,palette=palette)
	# sns.displot(data = data, x = x_field, y =y_field, hue = hue, alpha = 0.5, levels = 4, fill = True,palette="crest", kind="kde")

	a0.set_xlabel(xlabel)
	a0.set_ylabel(ylabel)
	if xlim is not None:
		a0.set_xlim(xlim)
	a0.get_legend().set_title(leg_title)

	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(3.0,3.0)
	fig.tight_layout()
	plt.savefig('./figures/'+save_name)


def loss_line_plot(identifier, save_name = 'line_test.pdf', field = 'loss', xlabel = 'Epoch',
 xlim = None, n_filter = None, plot_L = True, L_filter = None, ylabel = None):
	sns.set_theme(style="whitegrid", palette = "deep")
	sns.set_context("paper", rc=params)


	def get_data():
		files = glob.glob('./data/'+identifier)
		for i, file in enumerate(files):
			if i==0:
				df = pd.read_csv(file)
			else:
				df = pd.concat([df, pd.read_csv(file)], ignore_index = True)
		return df

	def add_param_field(data):
		data['# params'] = 'null'
		data['# params'][data['layers_student']==4] = 'L=4 (equal)'
		data['# params'][data['layers_student']==16] = 'L=16 (4 times more)'
		data['# params'][data['layers_student']==40] = 'L=40 (10 times more)'
		data['# params'][data['layers_student']==400] = 'L=400 (exponential)'
		data['# params'][data['layers_student']==1600] = 'L=1600 (exponential)'
		return data

	def add_n_field(data):
		data['num. qubits'] = 'null'
		data['num. qubits'][data['n_qubits']==4] = '4 qubits'
		data['num. qubits'][data['n_qubits']==8] = '8 qubits'
		data['num. qubits'][data['n_qubits']==12] = '12 qubits'
		data['num. qubits'][data['n_qubits']==16] = '16 qubits'
		return data

	
	fig = plt.figure()
	data = get_data()
	if n_filter is not None:
		data['filter'] = 0
		for n in n_filter:
			data['filter'] += (data['n_qubits'] ==n)
		data = data[data['filter']>0]

	if L_filter is not None:
		data['filter'] = 0
		for L in L_filter:
			data['filter'] += (data['layers_student'] ==L)
		data = data[data['filter']>0]

	if plot_L:
		data = add_param_field(data)
	else:
		data = add_n_field(data)

	# f, (a1, a0) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 5]},sharex = True)
	colors = sns.color_palette("rocket")
	if plot_L:
		data.sort_values(by=['layers_student'])
		sort_order = data['# params'].unique()
		sns.lineplot(data=data, x="step", y=field, units = 'name', palette = [colors[0], colors[2], colors[1], colors[-1]],
					hue="# params", estimator = None, alpha = 0.5)
	else:
		print(data['num. qubits'].unique())
		sns.lineplot(data=data, x="step", y=field, palette = colors[:4],
					hue="num. qubits", alpha = 0.5)

	ax = plt.gca()
	ax.set_xlabel(xlabel)
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylabel is not None:
		ax.set_ylabel(ylabel)

	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles=handles[0:], labels=labels[0:])
	handles, labels = plt.gca().get_legend_handles_labels()
	if plot_L:
		order = [3,1,2,0]
		ax.set_ylim([0,1.1])
		# plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'upper right', fontsize = 'xx-small', bbox_to_anchor=(1.05,1.1))
		plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])


	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(3.0,2.5)
	fig.tight_layout()
	plt.savefig('./figures/'+save_name)


def layerwise_line_plot(identifier, save_name = 'layerwise_test.pdf', xlabel = 'epoch', xlim = None):
	sns.set_theme(style="whitegrid", palette = "deep")
	sns.set_context("paper", rc=params)
	# plt.rc('legend',**{'fontsize':6})


	def get_data():
		files = glob.glob('./data/'+identifier)
		for i, file in enumerate(files):
			if i==0:
				df = pd.read_csv(file)
			else:
				df = pd.concat([df, pd.read_csv(file)], ignore_index = True)
		return df

	fig = plt.figure()
	data = get_data()
	
	data['loss'] = np.convolve(data['loss'], np.ones(10)/10, mode='same')
	data['layers_at_step'] += 1 # starts at 0 by python default and should really be one for plotting
	# data['ground state convergence'] = np.convolve(data['ground_state_loss'], np.ones(10)/10, mode='same')
	data['trace distance'] = np.convolve(np.sqrt(data['ground_state_loss']), np.ones(10)/10, mode='same')
	data['overparameterized'] = ( (data['n_qubits']//2)*16*2*data['layers_at_step'] >= 2**data['n_qubits'] )
	data['regime'] = 'not learnable'
	data['regime'][data['layers_at_step'] >= data['layers_target']] = 'learnable'
	data['regime'][data['overparameterized']] = 'overparameterized'

	data['convergence metric'] = 'loss'
	data['Value'] = data['loss']
	data2 = data.copy()
	data2['convergence metric'] = 'trace distance'	
	data2['Value'] = data['trace distance']
	data = pd.concat([data, data2], ignore_index = True)

	ax = sns.lineplot(data=data, x="step", y="value", hue="regime", style="convergence metric")

	# plt.legend(ncol=2)
	leg = ax.get_legend()
	leg.get_title().set_fontsize(10)

	ax = plt.gca()
	ax.set_yscale('log')
	plt.setp(ax.get_legend().get_texts(), fontsize='6') # for legend text
	plt.setp(ax.get_legend().get_title(), fontsize='8', weight='bold') # for legend title
	# ax.set_xlabel(xlabel)
	if xlim is not None:
		ax.set_xlim(xlim)

	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(3.0,2.5)
	fig.tight_layout()
	plt.savefig('./figures/'+save_name)

def layerwise_line_plot_updated(identifier, save_name = 'layerwise_test_updated.pdf', xlim = None):
	sns.set_theme(style="whitegrid", palette = "deep")
	sns.set_context("paper", rc=params)
	# plt.rc('legend',**{'fontsize':6})


	def get_data():
		files = glob.glob('./data/'+identifier)
		for i, file in enumerate(files):
			if i==0:
				df = pd.read_csv(file)
			else:
				df = pd.concat([df, pd.read_csv(file)], ignore_index = True)
		return df

	fig = plt.figure()
	data = get_data()


	data['loss'] = np.convolve(data['loss'], np.ones(10)/10, mode='same')
	# data['ground state convergence'] = np.convolve(data['ground_state_loss'], np.ones(10)/10, mode='same')
	data['trace distance'] = np.convolve(np.sqrt(data['ground_state_loss']), np.ones(10)/10, mode='same')
	per_step = np.min(data[data['layers_at_step']==1]['step']) - np.min(data[data['layers_at_step']==0]['step'])
	data['Layers in Ansatz'] = data['step']/per_step + 1
	if xlim is not None:
		data = data[data['Layers in Ansatz']<xlim[1]]
	
	data['layers_at_step'] += 1 # starts at 0 by python default and should really be one for plotting	
	learnable = (data['layers_at_step'])>=data['layers_target']
	learnable = np.min(data[learnable]['step'])
	overparameterized = ( (data['n_qubits']//2)*16*2*data['layers_at_step'] >= 2**data['n_qubits'] )
	overparameterized = np.min(data[overparameterized]['step'])
	total_steps = np.max(data['step'])

	data['convergence metric'] = 'loss'
	data['Value'] = data['loss']
	data2 = data.copy()
	data2['convergence metric'] = 'trace distance'	
	data2['Value'] = data['trace distance']
	data = pd.concat([data, data2], ignore_index = True)

	ax = sns.lineplot(data=data, x="Layers in Ansatz", y="Value", hue="convergence metric")
	ax.legend_.set_title(None)


	learnable_layer = data['layers_target'][0]
	overparameterized_layer = np.ceil(overparameterized/per_step)
	total_layers = np.max(data['Layers in Ansatz'])
	min_y = np.min(data['Value'])

	ax.axvspan(learnable_layer, overparameterized_layer, color='#728ab0', alpha=0.25)
	ax.axvspan(overparameterized_layer, total_layers, color='#6a9e84', alpha=0.25)
	ax.text( (learnable_layer+overparameterized_layer)/2,min_y*0.8,'learnable',
			 horizontalalignment='center',fontsize = 6)
	ax.text( (total_layers+overparameterized_layer)/2,min_y*0.8,'overparameterized',
			 horizontalalignment='center',fontsize = 6)

	# plt.legend(ncol=2)
	leg = ax.get_legend()
	ax = plt.gca()
	ax.set_yscale('log')
	plt.setp(ax.get_legend().get_texts(), fontsize='6') # for legend text
	# ax.set_xlabel(xlabel)
	if xlim is not None:
		ax.set_xlim(xlim)
		ticks = list(ax.get_xticks()) + [1]
		ticks.remove(0)
		ax.set_xticks(ticks)

	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(3.0,2.5)
	fig.tight_layout()
	plt.savefig('./figures/'+save_name)




if __name__ == '__main__':
	layerwise_line_plot_updated('adapt_VQE_qaoa_lradjust_11*', 'adapt_VQE_11qubits_lrchange_updated.pdf', xlim = [1,25])

	swarm_scatter_plot('updated_fast_local_VQE_4layers*.csv', save_name = 'updated_scatter_4_qaoa_4layers.pdf', hue = 'n_qubits', L_filter = [4], n_filter = [4,8,12,16,20,24], divide_loss = False)
	swarm_scatter_plot('SGD_updated_fast_local_VQE_4layers*.csv', save_name = 'SGD_updated_scatter_4_qaoa_4layers.pdf', hue = 'n_qubits', L_filter = [4], n_filter = [4,8,12,16,20,24], divide_loss = False)

	hist_and_swarm_plot_multiple('nvary_qcnn_*', save_name = 'nvary_qcnn_all_together_swarm.pdf', field = 'acc', y_field = 'qubits',
								 xlabel = 'Training Accuracy', xlim = [0.5,1.02], hue = 'qubits', n_filter = [4,6,8,10,12,14,16], order = ['n=4','n=6', 'n=8','n=10', 'n=12','n=14', 'n=16'])

	loss_line_plot('checkerboard_8*','checkerboard_8qubits_varyL.pdf', L_filter = [4,16,40,1600], ylabel = 'Loss')

