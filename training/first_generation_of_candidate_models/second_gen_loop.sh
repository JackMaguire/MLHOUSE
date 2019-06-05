#!/bin/bash

for prefix in rrrrl.Xx.leaky rrrrl.xX.leaky rrrrl.XX.leaky rrrrl.a.leaky rrrrl.b.leaky rrrrl.c.leaky rrrrl.d.leaky rrrrl.e.leaky; do
    next_model=$prefix.h5
    for x in 0 1 2 3 4 5 6 7; do
	if [ -f $prefix.trainingset3.epoch1.subrange$x.h5 ]; then
	    echo "Skipping $prefix.trainingset3.epoch1.subrange$x.h5"
	    sleep 2
	    next_model=$prefix.trainingset3.epoch1.subrange$x.h5
	else
	    echo "Starting $prefix.trainingset3.epoch1.subrange$x.h5"
	    sleep 2	    
	    for inputfile in /home/jackmag/mlhouse_training_data/training_data/small_numpy_lists/sub_range_${x}*; do

		while [ -f /home/jackmag/sleep_please ]; do
		    sleep 10
		done

		echo $inputfile

		python3 ../train.py --model $next_model --training_data $inputfile --starting_epoch 0 --epoch_checkpoint_frequency_in_hours 100 --num_epochs 1
		next_model=final.h5
	    done
	    cp final.h5 $prefix.trainingset3.epoch1.subrange$x.h5
	fi
    done
done
