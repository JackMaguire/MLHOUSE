#!/bin/bash

cd /home/jackmag/mlhouse_training_data/training_data

while true; do

    for x in TODO; do
	#TODO list of dirs to cycle through

	#Delete Prvious
	if [ -f dir_to_delete.txt ]; then
	    echo Deleting `cat dir_to_delete.txt` >> /home/jackmag/froome_log.txt
	    rm -r `cat dir_to_delete.txt`
	    rm dir_to_delete.txt
	fi

	#create next
	find `pwd/$x` | grep '.csv.gz$' | head -n 1 | xargs -n 10 -P 30 python3 ./slowly_and_carefully_generate_numpy_array_from_csv.py
	next_dir=${x}_npy
	mkdir $next_dir
	mv $x/*.npy $next_dir
	touch waiting_to_install_next_dir.txt

	while [ -f current_dir.txt ] && [ -f waiting_to_install_next_dir.txt ]; do
	    #waiting for python to see waiting_to_install_next_dir.txt and mv current_dir.txt to dir_to_delete.txt
	    #PYTHON DELETES waiting_to_install_next_dir.txt
	    sleep 1
	done

	echo Moving $next_dir to current >> /home/jackmag/froome_log.txt
	echo $next_dir > current_dir.txt
    done

done
