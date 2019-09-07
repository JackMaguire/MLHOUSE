#!/bin/bash

setup_next_dir(){
    next_dir_num=$1
    if [ ! -f /home/jackmag/top8000_mousetrap/contador_overflow/$next_dir_num.5d6260eb.tar.gz ]; then
	scp jackmag@rosettadesign.med.unc.edu:~/mlhouse_training_data/training_data/$next_dir_num.5d6260eb.tar.gz .
    else
	ln -s /home/jackmag/top8000_mousetrap/contador_overflow/$next_dir_num.5d6260eb.tar.gz
    fi

    tar -xzf $next_dir_num.5d6260eb.tar.gz
    mv $next_dir_num next
    rm $next_dir_num.5d6260eb.tar.gz
}

next_dir=$1
e=$2
command=$3

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
	#for y in 0.01 0.005 0.0025 0.001; do
	for y in 0.005; do
	    lr=$(../determine_learning_rate 0.00025 $y 4 $e)
	    echo $e $y $lr
	    ( python3 ../sub_trains/train.$y.py --training_data local_list.10.csv --learning_rate $lr 2>/dev/null | tail -n 3 | head -n 1 ) || ( echo $x went bad && exit 1 ) 
	done
    else
	for y in 0.005; do
	    lr=$(../determine_learning_rate 0.00025 $y 4 $e)
	    echo $e $y $lr
	    ( python3 ../sub_trains/train.$y.py --training_data local_list.csv --learning_rate $lr 2>/dev/null | tail -n 3 | head -n 1 ) || ( echo $x went bad && exit 1 ) 
	done
	#for x in ../sub_trains/*.py; do
	#    echo $x
	#    ( python3 $x --training_data local_list.csv --learning_rate $lr 2>/dev/null | tail -n 3 | head -n 1 ) || ( echo $x went bad && exit 1 )
	#done
    fi
    #python3 ../train.py --training_data local_list.csv 2>/dev/null | tail -n 3
    mv final.*.h5 ../
    echo "DONE TRAIN " `date`
fi
