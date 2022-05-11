import argparse
import torch
import torch.nn as nn
import numpy as np
from state_simulator import state_simulator, state_mixture_simulator, conjugate_transpose
import pandas as pd 





def get_args():
	parser = argparse.ArgumentParser()

	# optimizer/circuit arguments
	parser.add_argument('--n', default = 11, type=int)
	parser.add_argument('--layers_target', default = 5, type=int)
	parser.add_argument('--layers_learn_max', default = 50, type=int)
	parser.add_argument('--lr', default = 0.002, type=float)
	parser.add_argument('--lr_mult', default = 0.95, type=float)	
	parser.add_argument('--epochs_per', default = 5000, type=int)

	# other arguments
	parser.add_argument('--save-name', default = 'adapt_VQE_qaoa_lradjust_11', type=str)
	parser.add_argument('--device', default = 'cuda:1', type=str)
	parser.add_argument('--print-every', default = 25, type=int)
	parser.add_argument('--verbose', default = True, type=bool)

	return parser.parse_args()

dtype = torch.complex128


Z = torch.tensor(  [[1,0],
					[0,-1]]
					).type(dtype)
I = torch.tensor(  [[1,0],
					[0,1]]
					).type(dtype)

@torch.no_grad()
def random_unitary(n = 2,c = 1., device = 'cpu'):
	out = random_init( torch.zeros((2**n,2**n), dtype = dtype, device = device) )
	out = out - conjugate_transpose(out)
	# out = out/torch.sqrt(torch.sum(out*torch.conj(out)))
	return torch.matrix_exp(c*out)

@torch.no_grad()
def random_global_unitary(n=5, k=2, i_min = 0, i_max = 4, device = 'cpu', c = 0.1):
	out = torch.ones((1,1),dtype = dtype, device = device)
	i = 0
	while i < i_min:
		out = torch.kron(out, I.to(device))
		i += 1

	while (i+k-1) <= i_max:
		out = torch.kron(out, random_unitary(k,c,device))
		i += k

	while i < n:
		out = torch.kron(out, I.to(device))
		i += 1

	# print(torch.isclose(out@conjugate_transpose(out),torch.eye(2**n).to(device).type(dtype)))
	# print(torch.abs(out@conjugate_transpose(out)-torch.eye(2**n).to(device).type(dtype)))
	# print(out.shape)
	return out


@torch.no_grad()
def Z_matrix(z_on = 0, n=5, device = 'cpu'):
	out = torch.ones((1,1),dtype = dtype, device = device)
	i = 0
	while i < z_on:
		out = torch.kron(out, I.to(device))
		i += 1

	out = torch.kron(out, Z.to(device))
	i += 1

	while i < n:
		out = torch.kron(out, I.to(device))
		i += 1

	return out

@torch.no_grad()
def random_init(A, c = 1.):
    rand_nums_1 = torch.randn(A.shape, dtype = A.dtype, device = A.device)
    rand_nums_2 = torch.randn(A.shape, dtype = A.dtype, device = A.device)
    with torch.no_grad():
        # A.copy_(torch.matrix_exp(rand_nums-conjugate_transpose(rand_nums)))
        A.copy_( c*(rand_nums_1+1j*rand_nums_2) )
        return A

@torch.no_grad()
def get_locally_perturbed_QAOA_hamiltonian(n = 5, L = 5, device = 'cpu'):
	U1 = random_global_unitary(n = n, k = 2, i_min = 0, i_max = n-2, device = device, c = 1.0)
	U1 = U1@random_global_unitary(n = n, k = 2, i_min = 1, i_max = n-1, device = device, c = 1.0)
	# U2 = random_global_unitary(n = n, k = 2, i_min = 1, i_max = n-1, device = device, c = 1.0)
	# torch.mm(U1, random_global_unitary(n = n, k = 2, i_min = 1, i_max = n-1, device = device, c = 1.0), out = U1)
	# U1 = U1@U2

	U = torch.eye(2**n).type(dtype).to(device)
	for i in range(L):
		U = U@U1

	del U1

	mat = n*torch.eye(2**n).type(dtype).to(device)
	for i in range(n):
		mat += Z_matrix(z_on = i, n = n, device = device)

	ground_state = torch.zeros((2**n,1)).type(dtype).to(device)
	ground_state[-1,-1] = 1.
	return U@mat@conjugate_transpose(U), U@ground_state


def _2local_U_block(state, U, offset = 0):
	for i in range(state.n//2):
		state.apply_unitary(U[i], [(2*i + offset)%state.n, (2*i + offset+1)%state.n])
	return state


def _1local_U_block(state, U, offset = 0):
	for i in range(state.n):
		state.apply_unitary(U[i], [(i+offset)%state.n])
	return state


class checkerboard(nn.Module):
	def __init__(self,n,L, init_constant = 0.00001):
		super().__init__()
		self.L = L
		self.init_constant = init_constant
		self.n = n

		self.params = nn.ParameterDict()
		for i in range(self.L):
			self.params["2local_1_{}".format(i)] = nn.Parameter(torch.randn(n//2,4,4).type(dtype))
			self.params["2local_2_{}".format(i)] = nn.Parameter(torch.randn(n//2,4,4).type(dtype))
		self.reset_parameters()


	def reset_parameters(self):
		for a in self.params:
			random_init(self.params[a], self.init_constant)


	def forward(self,state):
		for i in range(self.L):
			H1 = self.params["2local_1_{}".format(i)]
			H2 = self.params["2local_2_{}".format(i)]
			state = _2local_U_block(state,torch.matrix_exp(H1-conjugate_transpose(H1)), offset = 0)
			state = _2local_U_block(state,torch.matrix_exp(H2-conjugate_transpose(H2)), offset = 1)
		return state.state

	def add_layer(self):
		self.params["2local_1_{}".format(self.L)] = nn.Parameter(torch.randn(self.n//2,4,4).type(dtype))
		random_init(self.params["2local_1_{}".format(self.L)], self.init_constant)
		self.params["2local_2_{}".format(self.L)] = nn.Parameter(torch.randn(self.n//2,4,4).type(dtype))
		random_init(self.params["2local_2_{}".format(self.L)], self.init_constant)
		self.L += 1




def H_expectation(x,H,normalize = True):
	# norms can sometimes be slightly off due to tiny computational error, normalize will fix this
	x = x.reshape(-1).unsqueeze(-1)
	if normalize:
		x = x / torch.sqrt(torch.abs(torch.sum(x*x.conj())))
	return torch.real(conjugate_transpose(x)@H@x)

def fidelity_loss(x,y, normalize = True):
	# norms can sometimes be slightly off due to tiny computational error, normalize will fix this
	x = x.reshape(-1)
	y = y.reshape(-1)
	if normalize:
		x = x / torch.sqrt(torch.abs(torch.sum(x*x.conj(),-1, keepdim = True)))
		y = y / torch.sqrt(torch.abs(torch.sum(y*y.conj(),-1, keepdim = True)))
	inner_product = torch.sum(x.reshape(-1)*y.reshape(-1).conj(), -1)
	return 1 - torch.mean(torch.abs(inner_product)**2)

def main():
	args = get_args()
	n = args.n
	layers_learn_max = args.layers_learn_max
	layers_target = args.layers_target
	epochs_per = args.epochs_per
	print_every = args.print_every
	save_name = './data/' + args.save_name + '.csv'
	device = args.device
	lr = args.lr
	lr_mult = args.lr_mult
	verbose = args.verbose

	net = checkerboard(n,1).to(device)
	H, ground_state = get_locally_perturbed_QAOA_hamiltonian(n=n, L=layers_target,device=device)

	print(H_expectation(ground_state.reshape(1,-1), H))
	from torch.optim import Adam, SGD, RMSprop, lr_scheduler
	optim = Adam(net.parameters(), lr = lr)

	results = {	'name': save_name,
				'n_qubits': n,
				'layers_target': layers_target,
				'layers_at_step': [], 
				'layers_learn_max': layers_learn_max,
				'lr': lr,
				'step': [],
				'loss': [],
				'ground_state_loss':[]}

	df = pd.DataFrame.from_dict(results)
	df.to_csv(save_name)


	for i in range(layers_learn_max):
		for epoch_i in range(epochs_per):
			step = i*epochs_per + epoch_i

			optim.zero_grad()
			state = state_mixture_simulator(n, device = device, dtype = dtype)
			out = net(state)
			loss = H_expectation(out,H)
			loss.backward()
			optim.step()

			if (epoch_i%print_every) == 0:
				state = state_mixture_simulator(n, device = device, dtype = dtype)
				out = net(state)
				ground_loss = fidelity_loss(out, ground_state)
				results['step'].append(step)
				results['layers_at_step'].append(i)
				results['loss'].append(loss.item())
				results['ground_state_loss'].append(ground_loss.item())
				# results['acc'].append(epoch_acc.item())
				print('layer {}, step {}, all batches: loss is {}, overlap distance is {}'.format(i, epoch_i, loss.item(), ground_loss.item()))

				df = pd.DataFrame.from_dict(results)
				df.to_csv(save_name)

		net.add_layer()
		net.to(device)
		lr *= lr_mult
		optim = Adam(net.parameters(), lr = lr)

		# if loss < 0.0001:
		# 	break


if __name__ == '__main__':
	main()