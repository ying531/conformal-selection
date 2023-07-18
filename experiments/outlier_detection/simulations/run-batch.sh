#!/bin/bash
for sig_id in {1..9}
	do
	for out_prop in {1..5}
		do 
		for seed in {1..100} 
			do
  Python3 ./simu.py $sig_id $out_prop $seed
done
done
done
