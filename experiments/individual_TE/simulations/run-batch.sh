#!/bin/bash
for rho_id in {1..2}
	do
	for setting in {1..9}
		do 
		for seed in {1..100} 
			do
  Rscript ./simu.R $rho_id $setting $seed
done
done
done
