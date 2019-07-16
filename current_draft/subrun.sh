#!/bin/bash

setup_next_dir(){
    next_dir_num=$1
    if [ ! -f /home/jackmag/top8000_mousetrap/contador_overflow/$next_dir_num.5d6260eb.tar.gz ]; then
	scp jackmag@rosettadesign.med.unc.edu:~/mlhouse_training_data/training_data/$next_dir_num.5d6260eb.tar.gz /home/jackmag/top8000_mousetrap/contador_overflow/$next_dir_num.5d6260eb.tar.gz
    fi
    ln -s /home/jackmag/top8000_mousetrap/contador_overflow/$next_dir_num.5d6260eb.tar.gz

#    if [[ "$next_dir_num" -lt "0" ]]; then
#	#local
#	ln -s /home/jackmag/top8000_mousetrap/contador_overflow/$next_dir_num.5d6260eb.tar.gz
#    else
#	#scp
#	scp jackmag@rosettadesign.med.unc.edu:~/mlhouse_training_data/training_data/$next_dir_num.5d6260eb.tar.gz .
#    fi
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
    if false; then
	head -n10 local_list.csv > local_list.10.csv
	for x in ../sub_trains/*.py; do
	    ( python3 $x --training_data local_list.10.csv 2>/dev/null | tail -n 3 | head -n 1 ) || ( echo $x went bad && exit 1 )
	done
    else
	for x in ../sub_trains/*.py; do
	    ( python3 $x --training_data local_list.csv 2>/dev/null | tail -n 3 | head -n 1 ) || ( echo $x went bad && exit 1 )
	done
    fi
    #python3 ../train.py --training_data local_list.csv 2>/dev/null | tail -n 3
    mv final.*.h5 ../
    echo "DONE TRAIN " `date`
fi
