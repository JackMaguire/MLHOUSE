#!/bin/bash

while read line; do
    printf "$line "
    for x in training_data.testing_set_1.onesided.csv training_data.testing_set_1.random.csv training_data.testing_set_1.repack.csv training_data.testing_set_1.twosided.csv training_data.testing_set_2.onesided.csv training_data.testing_set_2.random.csv training_data.testing_set_2.repack.csv training_data.testing_set_2.twosided.csv; do
	#printf "$(python3 ../test.custom1.py --model $line --testing_data ~/mlhouse_training_data/testing_data/$x 2>dev/null) "
	printf "$x "
    done
    echo
done
