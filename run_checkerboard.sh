#!/bin/bash

for n in 8 
	do
		for l in 4 16 40 
			do
				for i in {0..8..2}
					do
						python checkerboard_ansatz.py --n $n --layers_teacher 4 --layers_student $l --save-name "checkerboard_""$n""_$l""_$i" --device cuda:0 --verbose 0 --lr 0.001
					done
			done
	done

for n in 8 #{6..10..2}
	do
		for l in 1600
			do
				for i in {0..9..1}
					do
						python checkerboard_ansatz.py --n $n --layers_teacher 4 --layers_student $l --save-name "checkerboard_""$n""_$l""_$i" --device cuda:0 --verbose 0 --lr 0.0001 --print-every 1 --epochs 200
					done
			done
	done

