#!/bin/bash

dirname=base3.hefty
cd $dirname

#model_tmpl="../testing/multicool_tests/0.005/epoch_XXX.test.0.005.h5"
model_tmpl="../testing/base3.hefty/epoch_XXX.base3.hefty.h5"

for x in {0..100}; do
    model=$(echo $model_tmpl | sed "s/XXX/$x/g")
    python3 ../utilities/print_activations_for_each_layer.py --model $model --data ../testing/1.csv --prefix $(printf "%03d" $x)
done

num=$(ls 000.*.pdf | wc -l)
num=$((num-1))
for x in `seq 0 $num`; do
    convert -delay 20 -loop 0 *.$x.pdf layer_$x.gif
done
#for x in *.pdf; do
#    convert -density 300 $x -quality 90 $x.png
#done
