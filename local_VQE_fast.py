import argparse
import torch
import torch.nn as nn
import numpy as np
from state_simulator import state_simulator, state_mixture_simulator, conjugate_transpose
import pandas as pd 





def get_args():
	parser = argparse.ArgumentParser()

	# optimizer/circuit arguments
	parser.add_argument('--n', default = 9, type=int)
	parser.add_argument('--layers_target', default = 5, type=int)
	parser.add_argument('--layers_learn', default = 5, type=int)
	parser.add_argument('--lr', default = 0.01, type=float)
	parser.add_argument('--epochs', default = 30000, type=int)
	parser.add_argument('--optim_choice', default = 'Adam', type=str)

	# other arguments
	parser.add_argument('--save-name', default = 'fast_local_VQE_qaoa_test', type=str)
	parser.add_argument('--device', default = 'cpu', type=str)
	parser.add_argument('--print-every', default = 25, type=int)
	parser.add_argument('--verbose', default = True, type=bool)

	return parser.parse_args()



Z = torch.tensor(  [[1,0],
					[0,-1]]
					).type(torch.complex64)
I = torch.tensor(  [[1,0],
					[0,1]]
					).type(torch.complex64)

@torch.no_grad()
def random_unitary(n = 2,c = 1., device = 'cpu'):
	out = random_init( torch.zeros((2**n,2**n), dtype = torch.complex64, device = device) )
	out = out - conjugate_transpose(out)
	# out = out/torch.sqrt(torch.sum(out*torch.conj(out)))
	return torch.matrix_exp(c*out)

# @torch.no_grad()
# def random_global_unitary(n=5, k=2, i_min = 0, i_max = 4, device = 'cpu', c = 0.1):
# 	out = torch.ones((1,1),dtype = torch.complex64, device = device)
# 	i = 0
# 	while i < i_min:
# 		out = torch.kron(out, I.to(device))
# 		i += 1

# 	while (i+k-1) <= i_max:
# 		out = torch.kron(out, random_unitary(k,c,device))
# 		i += k

# 	while i < n:
# 		out = torch.kron(out, I.to(device))
# 		i += 1

# 	# print(torch.isclose(out@conjugate_transpose(out),torch.eye(2**n).to(device).type(torch.complex64)))
# 	# print(torch.abs(out@conjugate_transpose(out)-torch.eye(2**n).to(device).type(torch.complex64)))
# 	# print(out.shape)
# 	return out


# @torch.no_grad()
# def Z_matrix(z_on = 0, n=5, device = 'cpu'):
# 	out = torch.ones((1,1),dtype = torch.complex64, device = device)
# 	i = 0
# 	while i < z_on:
# 		out = torch.kron(out, I.to(device))
# 		i += 1

# 	out = torch.kron(out, Z.to(device))
# 	i += 1

# 	while i < n:
# 		out = torch.kron(out, I.to(device))
# 		i += 1

# 	return out

@torch.no_grad()
def Z_vector(z_on = 0, n=5, device = 'cpu'):
	out = torch.ones((1),dtype = torch.complex64, device = device)
	I = torch.ones((2),dtype = torch.complex64, device = device)
	Z = torch.ones((2),dtype = torch.complex64, device = device)
	Z[-1] = -1.
	i = 0
	while i < z_on:
		out = torch.kron(out, I.to(device))
		i += 1

	out = torch.kron(out, Z.to(device))
	i += 1

	while i < n:
		out = torch.kron(out, I.to(device))
		i += 1

	new_shape = [1,1]+[2]*n
	return out.reshape(*new_shape)



@torch.no_grad()
def random_init(A):
    rand_nums_1 = torch.randn(A.shape, dtype = A.dtype, device = A.device)
    rand_nums_2 = torch.randn(A.shape, dtype = A.dtype, device = A.device)
    with torch.no_grad():
        # A.copy_(torch.matrix_exp(rand_nums-conjugate_transpose(rand_nums)))
        A.copy_(rand_nums_1+1j*rand_nums_2)
        return A

# @torch.no_grad()
# def get_locally_perturbed_QAOA_hamiltonian(n = 5, L = 5, device = 'cpu'):
# 	U1 = random_global_unitary(n = n, k = 2, i_min = 0, i_max = n-2, device = device, c = 1.0)
# 	U1 = U1@random_global_unitary(n = n, k = 2, i_min = 1, i_max = n-1, device = device, c = 1.0)
# 	# U2 = random_global_unitary(n = n, k = 2, i_min = 1, i_max = n-1, device = device, c = 1.0)
# 	# torch.mm(U1, random_global_unitary(n = n, k = 2, i_min = 1, i_max = n-1, device = device, c = 1.0), out = U1)
# 	# U1 = U1@U2

# 	U = torch.eye(2**n).type(torch.complex64).to(device)
# 	for i in range(L):
# 		U = U@U1

# 	del U1

# 	mat = n*torch.eye(2**n).type(torch.complex64).to(device)
# 	for i in range(n):
# 		mat += Z_matrix(z_on = i, n = n, device = device)

# 	ground_state = torch.zeros((2**n,1)).type(torch.complex64).to(device)
# 	ground_state[-1,-1] = 1.
# 	return U@mat@conjugate_transpose(U), U@ground_state


class Hamiltonian:
	def __init__(self, n = 5, L = 5, c = 1.0, device = 'cpu'):
		self.n = n
		self.L = L

		with torch.no_grad():
			self.U1 = [random_unitary(2,c,device) for i in range(n//2)]
			self.U2 = [random_unitary(2,c,device) for i in range(n//2)]
			self.U1_inv = [conjugate_transpose(m) for m in self.U1]
			self.U2_inv = [conjugate_transpose(m) for m in self.U2]

			state_shape = [1,1]+[2]*n
			self.Z = torch.ones(state_shape, device = device, dtype = torch.complex64)*n
			for i in range(n):
				self.Z += Z_vector(i, n, device = device)

			ground_state = torch.zeros((2**n)).type(torch.complex64).to(device)
			ground_state[-1] = 1.
			ground_state = state_mixture_simulator(n = n, state_init = ground_state.reshape(*state_shape))
			ground_state = self.apply_unitary(ground_state, False)
			self.ground_state = ground_state.state

	def measure(self, state):
		state_in = state.state.clone()
		state = self.apply_unitary(state, True)
		state.state = state.state*self.Z
		state = self.apply_unitary(state, False)
		return torch.real(torch.sum(state_in.conj()*state.state))

	def apply_unitary(self, state, forward = True):
		for i in range(self.L):
			if forward:
				state = _2local_U_block(state, self.U1, 0)
				state = _2local_U_block(state, self.U2, 1)
			else:
				state = _2local_U_block(state, self.U2_inv, 1)
				state = _2local_U_block(state, self.U1_inv, 0)
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
		return state


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
	layers_learn= args.layers_learn
	layers_target = args.layers_target
	epochs = args.epochs
	print_every = args.print_every
	save_name = './data/' + args.save_name + '.csv'
	device = args.device
	lr = args.lr
	verbose = args.verbose
	optim_choice = args.optim_choice

	from torch.optim import Adam, SGD
	if optim_choice == 'Adam':
		torch_optimizer = Adam
	elif optim_choice == 'SGD':
		torch_optimizer = SGD
	else:
		print('not a valid optimizer choice: resorting to Adam')
		torch_optimizer = Adam

	net = checkerboard(n,layers_learn).to(device)
	H = Hamiltonian(n=n, L=layers_target,device=device)

	ground_state = state_mixture_simulator(n = n, state_init = H.ground_state)
	print(H.measure(ground_state))
	optim = Adam(net.parameters(), lr = lr)

	results = {	'name': save_name,
				'n_qubits': n,
				'layers_target': layers_target,
				'layers_learn': layers_learn, 
				'epochs': epochs,
				'lr': lr,
				'step': [],
				'loss': [],
				'ground_state_loss':[]}

	df = pd.DataFrame.from_dict(results)
	df.to_csv(save_name)


	for i in range(epochs):

		optim.zero_grad()
		state = state_mixture_simulator(n, device = device)
		out = net(state)
		loss = H.measure(out)
		loss.backward()
		optim.step()

		if (i%print_every) == 0:
			state = state_mixture_simulator(n, device = device)
			out = net(state)
			ground_loss = fidelity_loss(out.state, H.ground_state)
			results['step'].append(i)
			results['loss'].append(loss.item())
			results['ground_state_loss'].append(ground_loss.item())
			# results['acc'].append(epoch_acc.item())
			print('epoch {}, all batches: loss is {}, overlap distance is {}'.format(i, loss.item(), ground_loss.item()))

			df = pd.DataFrame.from_dict(results)
			df.to_csv(save_name)

		# if loss < 0.001:
		# 	break


if __name__ == '__main__':
	main()