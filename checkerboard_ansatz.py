import argparse
import torch
import torch.nn as nn
import numpy as np
from state_simulator import state_simulator, state_mixture_simulator, conjugate_transpose
import pandas as pd 





def get_args():
	parser = argparse.ArgumentParser()

	# optimizer/circuit arguments
	parser.add_argument('--n', default = 6, type=int)
	parser.add_argument('--layers_teacher', default = 4, type=int)
	parser.add_argument('--layers_student', default = 64, type=int)
	parser.add_argument('--train-size', default = 512, type=int)
	parser.add_argument('--batch-size', default = 128, type=int)
	parser.add_argument('--lr', default = 0.01, type=float)
	parser.add_argument('--epochs', default = 5000, type=int)

	# other arguments
	parser.add_argument('--save-name', default = 'checkerboard_test', type=str)
	parser.add_argument('--device', default = 'cuda:1', type=str)
	parser.add_argument('--print-every', default = 20, type=int)
	parser.add_argument('--verbose', default = True, type=bool)

	return parser.parse_args()


ZZ = torch.tensor( [[1j,0,0,0],
					[0,-1j,0,0],
					[0,0,-1j,0],
					[0,0,0,1j]]
					).type(torch.complex64)

def random_init(A):
    rand_nums_1 = torch.randn(A.shape, dtype = A.dtype, device = A.device)
    rand_nums_2 = torch.randn(A.shape, dtype = A.dtype, device = A.device)
    with torch.no_grad():
        # A.copy_(torch.matrix_exp(rand_nums-conjugate_transpose(rand_nums)))
        A.copy_(rand_nums_1+1j*rand_nums_2)
        return A

def _2local_ZZ_block(state, U, offset = 0):
	for i in range(state.n//2):
		state.apply_unitary(torch.matrix_exp(ZZ.to(state.state.device)*U[i].type(torch.complex64)), [(2*i + offset)%state.n, (2*i + offset+1)%state.n])
	return state


def _2local_U_block(state, U, offset = 0):
	for i in range(state.n//2):
		state.apply_unitary(U[i], [(2*i + offset)%state.n, (2*i + offset+1)%state.n])
	return state


def _1local_U_block(state, U, offset = 0):
	for i in range(state.n):
		state.apply_unitary(U[i], [(i+offset)%state.n])
	return state


class checkerboard(nn.Module):
	def __init__(self,n,L):
		super().__init__()
		self.L = L

		self.params = nn.ParameterDict()
		for i in range(self.L):
			self.params["2local_1_{}".format(i)] = nn.Parameter(torch.randn(n//2,4,4).type(torch.complex64))
			self.params["2local_2_{}".format(i)] = nn.Parameter(torch.randn(n//2,4,4).type(torch.complex64))
		self.reset_parameters()


	def reset_parameters(self):
		for a in self.params:
			random_init(self.params[a])


	def forward(self,state):
		for i in range(self.L):
			H1 = self.params["2local_1_{}".format(i)]
			H2 = self.params["2local_2_{}".format(i)]
			state = _2local_U_block(state,torch.matrix_exp(H1-conjugate_transpose(H1)), offset = 0)
			state = _2local_U_block(state,torch.matrix_exp(H2-conjugate_transpose(H2)), offset = 1)
		return state.state



class checkerboard_ZZ(nn.Module):
	def __init__(self,n,L):
		super().__init__()
		self.L = L

		self.params = nn.ParameterDict()
		for i in range(self.L):
			self.params["1local_{}".format(i)] = nn.Parameter(torch.randn(n,2,2).type(torch.complex64))
			self.params["2local_1_{}".format(i)] = nn.Parameter(torch.randn(n//2).type(torch.float32))
			self.params["2local_2_{}".format(i)] = nn.Parameter(torch.randn(n//2).type(torch.float32))
		self.reset_parameters()


	def reset_parameters(self):
		for a in self.params:
			random_init(self.params[a])


	def forward(self,state):
		for i in range(self.L):
			H = self.params["1local_{}".format(i)]
			state = _1local_U_block(state,torch.matrix_exp(H - conjugate_transpose(H)))
			state = _2local_ZZ_block(state,self.params["2local_1_{}".format(i)], offset = 0)
			state = _2local_ZZ_block(state,self.params["2local_2_{}".format(i)], offset = 1)
		return state.state

def get_random_computational_ids(n_qubits, n_states=32):
	return np.random.randint(2**n_qubits, size = (n_states))

def get_random_computational_states(n_qubits, n_states = 32):
	ids = get_random_computational_ids(n_qubits,n_states)
	return convert_id_to_state(ids, n_qubits)

def convert_id_to_state(ids, n_qubits):
	out = np.zeros((len(ids), 1, 2**n_qubits))
	for i, id_i in enumerate(ids):
		out[i,0,id_i] = 1.
	return torch.tensor(out.reshape([len(ids),1] + [2]*(n_qubits))).type(torch.complex64)


# def accuracy(x, y):
# 	return torch.mean((torch.sgn(x-0.5)==torch.sgn(y-0.5)).type(torch.float32))

def fidelity_loss(x,y, normalize = True):
	# norms can sometimes be slightly off due to tiny computational error, normalize will fix this
	x = x.reshape(x.shape[0],-1)
	y = y.reshape(y.shape[0],-1)
	if normalize:
		x = x / torch.sqrt(torch.abs(torch.sum(x*x.conj(),-1, keepdim = True)))
		y = y / torch.sqrt(torch.abs(torch.sum(y*y.conj(),-1, keepdim = True)))
	inner_product = torch.sum(x.reshape(x.shape[0],-1)*y.reshape(y.shape[0],-1).conj(), -1)
	return 1 - torch.mean(torch.abs(inner_product)**2)

def main():
	args = get_args()
	n = args.n
	layers_student = args.layers_student
	layers_teacher = args.layers_teacher
	train_size = args.train_size
	batch_size = args.batch_size
	epochs = args.epochs
	print_every = args.print_every
	save_name = './data/' + args.save_name + '.csv'
	device = args.device
	lr = args.lr
	verbose = args.verbose

	target_net = checkerboard(n,layers_teacher).to(device)

	n_batches = train_size // batch_size

	data_ids = get_random_computational_ids(n, train_size)
	y = torch.zeros([len(data_ids),1]+[2]*n).type(torch.complex64).to(device)    # may need to store in cpu for large n
	for batch_i in range(n_batches):
		data = convert_id_to_state(data_ids[batch_i*batch_size:(batch_i+1)*batch_size], n).to(device)
		with torch.no_grad():
			state = state_mixture_simulator(n, state_init = data, device = device)
			y[batch_i*batch_size:(batch_i+1)*batch_size] = target_net(state)

	data_ids_test = get_random_computational_ids(n, 500)
	data_test = convert_id_to_state(data_ids_test, n).to(device)		
	with torch.no_grad():
		state = state_mixture_simulator(n, state_init = data_test, device = device)
		y_test = target_net(state)

	from torch.optim import Adam, SGD, RMSprop, lr_scheduler
	student_net = checkerboard(n,layers_student).to(device)
	optim = Adam(student_net.parameters(), lr = lr)

	# criterion = nn.MSELoss()	
	criterion = fidelity_loss	

	results = {	'name': save_name,
				'n_qubits': n,
				'layers_teacher': layers_teacher,
				'layers_student': layers_student, 
				'train_size': train_size,
				'epochs': epochs,
				'lr': lr,
				'step': [],
				'loss': [],
				'test_loss':[]}

	df = pd.DataFrame.from_dict(results)
	df.to_csv(save_name)


	for i in range(epochs):
		epoch_loss = 0.
		epoch_acc = 0.
		for batch_i in range(n_batches):
			optim.zero_grad()
			data = convert_id_to_state(data_ids[batch_i*batch_size:(batch_i+1)*batch_size], n).to(device)
			state = state_mixture_simulator(n, state_init = data, device = device)
			out = student_net(state)
			loss = criterion(out,y[batch_i*batch_size:(batch_i+1)*batch_size])
			loss.backward()
			optim.step()

			# acc = accuracy(out, y[batch_i*batch_size:(batch_i+1)*batch_size])
			# if verbose:
			# 	print('epoch {}, batch {}: loss is {}'.format(i, batch_i, loss.item()))
			epoch_loss += loss
			# epoch_acc += acc

		# epoch_acc = epoch_acc/n_batches
		epoch_loss = epoch_loss/n_batches
		# if verbose:
		# 	print('epoch {}, all batches: loss is {}, accuracy is {}'.format(i, epoch_loss.item(), epoch_acc))

		if (i%print_every) == 0:
			state = state_mixture_simulator(n, state_init = data_test, device = device)
			test_loss = criterion(student_net(state), y_test)
			results['step'].append(i)
			results['loss'].append(epoch_loss.item())
			results['test_loss'].append(test_loss.item())
			# results['acc'].append(epoch_acc.item())
			print('epoch {}, all batches: loss is {}, test loss is {}'.format(i, epoch_loss.item(), test_loss.item()))

			df = pd.DataFrame.from_dict(results)
			df.to_csv(save_name)

		if epoch_loss < 0.001:
			break


if __name__ == '__main__':
	main()