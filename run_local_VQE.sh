#!/bin/bash

for n in 4 8 12 16 20 24
	do
		for l in 4
			do
				for i in {0..21..3}
					do
						let ii=$i+1
						let iii=$i+2
						python local_VQE_fast.py --n $n --layers_target 4 --layers_learn $l --save-name "SGD_updated_fast_local_VQE_4layers_""$n""_$l""_$i" --device "cuda:1" --verbose 0 --optim_choice 'SGD' --lr 0.01 & pid1=$! 
						python local_VQE_fast.py --n $n --layers_target 4 --layers_learn $l --save-name "SGD_updated_fast_local_VQE_4layers_""$n""_$l""_$ii" --device "cuda:2" --verbose 0 --optim_choice 'SGD' --lr 0.01 & pid2=$! 
						python local_VQE_fast.py --n $n --layers_target 4 --layers_learn $l --save-name "SGD_updated_fast_local_VQE_4layers_""$n""_$l""_$iii" --device "cuda:3" --verbose 0 --optim_choice 'SGD' --lr 0.01 & pid3=$!
						wait $pid1 $pid2 $pid3
					done
			done
	done

