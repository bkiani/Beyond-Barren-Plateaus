import argparse
import torch
import torch.nn as nn
import numpy as np
from state_simulator import state_simulator, state_mixture_simulator, conjugate_transpose
import pandas as pd 





def get_args():
	parser = argparse.ArgumentParser()

	# optimizer/circuit arguments
	parser.add_argument('--n', default = 12, type=int)
	parser.add_argument('--train-size', default = 512, type=int)
	parser.add_argument('--batch-size', default = 128, type=int)
	parser.add_argument('--lr', default = 0.001, type=float)
	parser.add_argument('--epochs', default = 5000, type=int)

	# other arguments
	parser.add_argument('--save-name', default = 'qcnn_test', type=str)
	parser.add_argument('--device', default = 'cpu', type=str)
	parser.add_argument('--print-every', default = 25, type=int)
	parser.add_argument('--verbose', default = True, type=bool)

	return parser.parse_args()




def random_init(A):
    rand_nums = torch.randn(A.shape, dtype = A.dtype, device = A.device)
    with torch.no_grad():
        # A.copy_(torch.matrix_exp(rand_nums-conjugate_transpose(rand_nums)))
        A.copy_(rand_nums)
        return A

def _2local_conv(state, U):
	for i in range(state.n//2):
		state.apply_unitary(U, [2*i, 2*i+1])
	if state.n > 2:
		for i in range(state.n//2):
			state.apply_unitary(U, [2*i+1, (2*i+2)%state.n])
	return state


def _2local_pool(state):
	if state.n >= 2:
		for i in range(state.n//2):
			measurement = state.measure(i)
	return state


class QCNN(nn.Module):
	def __init__(self,n):
		super().__init__()
		self.L = int(np.ceil(np.log2(n)))

		self.params = nn.ParameterDict()
		for i in range(self.L):
			self.params["H{}".format(i)] = nn.Parameter(torch.randn(4,4).type(torch.complex64))
		self.reset_parameters()


	def reset_parameters(self):
		for a in self.params:
			random_init(self.params[a])


	def forward(self,state):
		for i,p in enumerate(self.params):
			state = _2local_conv(state, torch.matrix_exp(self.params[p] - conjugate_transpose(self.params[p])))
			state = _2local_pool(state)
			if i == 0:
				state = state.convert_to_density_matrix()

		out = state.measure(0)

		return out[:,0].real

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


# def get_random_computational_states(n_qubits, n_states = 32):
# 	ids = np.random.randint(2**n_qubits, size = (n_states))
# 	out = np.zeros((n_states,1, 2**n_qubits))
# 	for i, id_i in enumerate(ids):
# 		out[i,0,id_i] = 1.
# 	return torch.tensor(out.reshape([n_states,1] + [2]*(n_qubits))).type(torch.complex64)

def accuracy(x, y):
	return torch.mean((torch.sgn(x-0.5)==torch.sgn(y-0.5)).type(torch.float32))

def main():
	args = get_args()
	n = args.n
	train_size = args.train_size
	batch_size = args.batch_size
	epochs = args.epochs
	print_every = args.print_every
	save_name = './data/' + args.save_name + '.csv'
	device = args.device
	lr = args.lr
	verbose = args.verbose

	target_QCNN = QCNN(n).to(device)

	n_batches = train_size // batch_size

	data_ids = get_random_computational_ids(n, train_size)
	y = torch.zeros(len(data_ids)).to(device)
	for batch_i in range(n_batches):
		data = convert_id_to_state(data_ids[batch_i*batch_size:(batch_i+1)*batch_size], n).to(device)
		with torch.no_grad():
			state = state_mixture_simulator(n, state_init = data, device = device)
			y[batch_i*batch_size:(batch_i+1)*batch_size] = target_QCNN(state)

	from torch.optim import Adam, SGD, RMSprop, lr_scheduler
	student_QCNN = QCNN(n).to(device)
	optim = Adam(student_QCNN.parameters(), lr = lr)

	criterion = nn.MSELoss()	

	results = {	'name': save_name,
				'n_qubits': n, 
				'train_size': train_size,
				'epochs': epochs,
				'lr': lr,
				'step': [],
				'loss': [],
				'acc': [] }

	df = pd.DataFrame.from_dict(results)
	df.to_csv(save_name)


	for i in range(epochs):
		epoch_loss = 0.
		epoch_acc = 0.
		for batch_i in range(n_batches):
			optim.zero_grad()
			data = convert_id_to_state(data_ids[batch_i*batch_size:(batch_i+1)*batch_size], n).to(device)
			state = state_mixture_simulator(n, state_init = data, device = device)
			out = student_QCNN(state)
			loss = criterion(out,y[batch_i*batch_size:(batch_i+1)*batch_size])
			loss.backward()
			optim.step()

			acc = accuracy(out, y[batch_i*batch_size:(batch_i+1)*batch_size])
			# if verbose:
			# 	print('epoch {}, batch {}: loss is {}, accuracy is {}'.format(i, batch_i, loss.item(), acc))
			epoch_loss += loss
			epoch_acc += acc

		epoch_acc = epoch_acc/n_batches
		epoch_loss = epoch_loss/n_batches
		
		if (i%print_every) == 0:
			results['step'].append(i)
			results['loss'].append(epoch_loss.item())
			results['acc'].append(epoch_acc.item())
			if verbose:
				print('epoch {}, all batches: loss is {}, accuracy is {}'.format(i, epoch_loss.item(), epoch_acc))

	df = pd.DataFrame.from_dict(results)
	df.to_csv(save_name)


if __name__ == '__main__':
	main()