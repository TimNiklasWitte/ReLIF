#!/bin/bash

for dataset_name in NMMNIST SHD; do

	for freq_max in 1 2 8; do	
      		sbatch RunTraining.sh $dataset_name ReLIF $freq_max
	done
	
	#echo RunTraining.sh $dataset_name LIF None 

done

