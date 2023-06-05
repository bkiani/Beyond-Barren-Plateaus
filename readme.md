# Beyond Barren Plateaus

In this repo, we are sharing code for replicating experiments in our paper **Beyond Barren Plateaus: Quantum Variational Algorithms are Swamped with Traps** ([published version](https://www.nature.com/articles/s41467-022-35364-5) or [arXiv link](https://arxiv.org/abs/2205.05786)).


# Files

The main python files are listed below:
- `state_simulator.py`: this file contains classes that can implement quantum circuits either as pure states or density matrices.
- `checkerboard_ansatz.py`: used to run a circuit implementing a checkerboard ansatz and run optimization in a teacher-student setup
- `QCNN.py`: used to implement a QCNN ansatz and run optimization in a teacher-student setup 
- `local_VQE_fast.py`: implements VQE experiments that runs relatively fast even up to 25 qubits (assuming high memory GPU is available)
- `adapt_VQE.py`: perform adaptive layer-wise optimization in a VQE setting

Also included are files to create plots and bash scripts to automate running the code. First run scripts to output data as csv files into the `\.data\` folder before making plots. Please also update the bash scripts if you decide to use them to account for the proper number of GPUs that you have.

## Dependencies

All code is written in Python. The following packages are needed to run the code (version we used in parentheses):
- *Pytorch* (1.11.0)
- *Numpy* (1.22.3)
- *Pandas* (1.1.3)- for loading and saving data
- *Matplotlib* (3.5.1) - for making plots
- *Seaborn* (0.11.2) - for making plots

We ran all simulations on an NVIDIA RTX A6000 GPU which has 48 GB of memory. For GPUs with less memory, consider reducing the number of qubits when out of memory errors arise.

## Authors

* [Bobak Kiani](https://github.com/bkiani) (MIT) 
* [Eric Anschuetz](https://github.com/eanschuetz) (MIT)
