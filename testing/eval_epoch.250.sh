#!/bin/bash

list=250.csv

if [[ $# -lt 1 ]]; then
    echo which epoch
fi

epoch=$1

for x in */epoch_$epoch.*.h5 ; do
    if [ ! -f $x.${list}.results ]; then
	python3 test.py --data ${list} --model $x 1> $x.${list}.results;
    fi
done
