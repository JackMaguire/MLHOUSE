#!/bin/bash

for prefix in rrrrl.xx rrrrl.xx.leaky trrrl.xx hrrrl.xx; do

    for x in 0 1 2 3 4 5 6 7; do
	while [ ! -f /home/jackmag/sleep_please ]; do
	    sleep 10
	done

	while [ ! -f $prefix.trainingset3.epoch1.subrange$x.h5 ]; do
	    sleep 10
	done

	for y in 2.onesided.csv 2.random.csv 2.repack.csv 2.twosided.csv 1.onesided.csv 1.random.csv 1.repack.csv 1.twosided.csv; do
	    python3 ../test.custom1.py --model $prefix.trainingset3.epoch1.subrange$x.h5 --testing_data /home/jackmag/mlhouse_training_data/testing_data/npy_training_data.testing_set_$y 2>&1 | grep RESULTS >> $prefix.trainingset3.epoch1.subrange$x.testlog
	done

    done
done

