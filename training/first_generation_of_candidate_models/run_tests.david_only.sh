#!/bin/bash

prefix=$1
subrange=$2
if [ ! -f $prefix.trainingset3.epoch1.subrange$subrange.h5 ]; then
    echo "$prefix.trainingset3.epoch1.subrange$subrange.h5 does not exist"
    exit 1
fi

x=$subrange

for y in 1.onesided.csv 1.random.csv 1.repack.csv 1.twosided.csv; do

    if grep $y $prefix.trainingset3.epoch1.subrange$x.testlog; then
	echo Skipping $y
	continue
    fi

    echo python3 ../test.custom1.py --model $prefix.trainingset3.epoch1.subrange$x.h5 --testing_data /home/jackmag/mlhouse_training_data/testing_data/npy_training_data.testing_set_$y
    echo $y `python3 ../test.custom1.py --model $prefix.trainingset3.epoch1.subrange$x.h5 --testing_data /home/jackmag/mlhouse_training_data/testing_data/npy_training_data.testing_set_$y 2>&1 | grep RESULTS` >> $prefix.trainingset3.epoch1.subrange$x.testlog
done

