#!/bin/bash

model_tmpl="../testing/multicool_tests/0.005/epoch_XXX.test.0.005.h5"

for x in {0..1}; do
    model=$(echo $model_tmpl | sed "s/XXX/$x/g")
    python3 ../utilities/print_activations_for_each_layer.py --model $model --data ../testing/1.csv --prefix $(printf "%03d" $x)
done
