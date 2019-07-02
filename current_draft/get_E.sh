#!/bin/bash

python3 test.six_bin.py --model $1 --testing_data /home/jackmag/top8000_mousetrap/testing_data/test_10/list.csv 2>/dev/null | awk '{ print $1 " " 4 - ($3 + $4 + $5 + $6 + $7 + $8 + $9 + $10 )}'
