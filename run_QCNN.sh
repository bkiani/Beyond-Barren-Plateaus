#!/bin/bash

for n in 16
	do
		for i in {0..100..1}
			do
				python QCNN.py --n $n --save-name "nvary_qcnn_""$n""_$i" --device cuda:0 --verbose False 
			done
	done
