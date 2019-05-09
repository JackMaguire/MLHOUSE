#!/bin/bash

#Just copy-pasting relevant lines here so I don't forget them

./scons.py -j40 mode=release regenerate_training_input_from_interface

for x in *.resfile; do     y=$(echo $x | sed 's/resfile/input2.csv/g');     z=$(echo $x | sed 's/resfile/output.csv/g');     echo $x,$z,$y; done > input_thingy.csv

~/Rosetta/main/source/bin/regenerate_training_input_from_interface.default.linuxgccrelease -resfile_correspondingout_printfile_csv ./input_thingy.csv -s /home/jackmag/mlhouse_training_data/poses/$pdb

for x in {300..339}; do echo $x; done | xargs -n1 -P 40 bash create_new_inputs.9600.sh
