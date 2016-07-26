#!/bin/bash

mapfile -t MODELS < "$1"
mapfile -t PARAMS < "$2"
OUTPUT_DIR="$3"

echo "Starting many jobs"
for ((i=0; i < ${#MODELS[@]}; i++)); do
	M=( ${MODELS[$i]} )
	m_name=${M[0]}
	m=${M[@]:1}
	for ((j=0; j < ${#PARAMS[@]}; j++)); do
		P=( ${PARAMS[$j]} )
		p_name=${P[0]}
		p=${P[@]:1}
		
		echo python3 src/train_cls.py --gpu '$GPU' $m $p &> ${OUTPUT_DIR}${p_name}_${m_name}
		wait_for_gpu python3 src/train_cls.py --gpu '$GPU' $m $p &>> ${OUTPUT_DIR}${p_name}_${m_name} &
		sleep 0.1
	done
done
echo "Started $(jobs -p | wc -w) jobs. Waiting for them to finish"
wait
