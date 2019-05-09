#!/bin/bash

split=$1

cd split_${split}_dir || exit 1

for x in {1..300}; do
   
    if ls split_${split}.${x}.*.info.txt 1>/dev/null 2>/dev/null; then

	for j in split_${split}.${x}.*.resfile; do
	    y=$(echo $j | sed 's/resfile/input2.csv/g')
	    z=$(echo $j | sed 's/resfile/output.csv.gz/g')
	    echo $j,$z,$y
	done > input_thingy.csv

	#sed -i 's/HH/H/g' *.resfile

	pdb=$(grep pdb split_${split}.${x}.*.info.txt | head -n1 | awk -F/ '{print $(NF-1) "/" $(NF) }')

	#ls /home/jackmag/mlhouse_training_data/poses/$pdb

	/home/jackmag/Rosetta/main/source/bin/regenerate_training_input_from_interface.default.linuxgccrelease -resfile_correspondingout_printfile_csv ./input_thingy.csv -s /home/jackmag/mlhouse_training_data/poses/$pdb

	awk -F, '{print $3}' input_thingy.csv | while read line; do
	    gzip $line
	done

	rm input_thingy.csv
    fi

done
