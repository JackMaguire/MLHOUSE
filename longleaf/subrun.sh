#!/bin/bash

setup_next_dir(){
    next_dir_num=$1
    ln -s /nas/longleaf/home/jackmag/pine/training_data/$next_dir_num.5d6260eb.tar.gz
    tar -xzf $next_dir_num.5d6260eb.tar.gz
    mv $next_dir_num next
    rm $next_dir_num.5d6260eb.tar.gz
}

next_dir=$1
command=$2

if [[ $command -eq 1 ]]; then
    #setup next dir
    next_dir_num=$next_dir
    setup_next_dir $next_dir_num
    echo "DONE SETUP " `date`
else
    #run on current dir
    cd curr
    if false ; then
	head -n10 local_list.csv > local_list.10.csv
	for x in ../sub_trains/*.py; do
	    echo $x
	    ( python3 $x --training_data local_list.10.csv 2>/dev/null | tail -n 3 | head -n 1 ) || ( echo $x went bad && exit 1 )
	done
    else
	for x in ../sub_trains/*.py; do
	    echo $x
	    #( python3 $x --training_data local_list.csv 2>/dev/null | tail -n 3 | head -n 1 ) || ( echo $x went bad && exit 1 )
	    ( python3 $x --training_data local_list.csv ) || ( echo $x went bad && exit 1 )
	done
    fi
    #python3 ../train.py --training_data local_list.csv 2>/dev/null | tail -n 3
    mv final.*.h5 ../
    echo "DONE TRAIN " `date`
fi
