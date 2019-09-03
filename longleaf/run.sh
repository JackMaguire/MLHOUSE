#!/bin/bash

# We have 2 directories: curr and next

# 0. make first dir (create next, move to curr)
# loop:
#     1. run on curr WHILE create next
#     2. delete curr
#     3. move next to curr

workdir=`pwd`

epoch=1
for y in start.*.h5; do
    cp $y `echo $y | sed 's/start/current/g'`
done

get_next_dir_num(){
    #ssh jackmag@rosettadesign.med.unc.edu bash get_a_dir.sh
    shuf temp_list | head -n 1
}

setup_next_dir(){
    next_dir_num=$1
    bash subrun.sh $next_dir_num 1
}

# Clean up from previous attempt
rm -r curr next

# 0
next_dir=`get_next_dir_num`
setup_next_dir $next_dir
if [ ! -d next ]; then
    echo "next does not exist"
    exit 1
fi
mv next curr

for epoch in {0..1000}; do

    cd $workdir
    if ls epoch_${epoch}.*.h5 1>/dev/null 2>/dev/null; then
	echo "Skipping epoch $epoch"
	for y in epoch_$epoch.*.h5; do
	    cp $y `echo $y | sed "s/epoch_$epoch/current/g"`
	done
	continue;
    fi

    echo "Starting epoch $epoch"

    for subepoch in {1..10}; do
	next_dir=`get_next_dir_num`

	#At this moment, curr is ready to run and next does not exist

	# 1
	echo "START " `date`
	echo 1 2 | xargs -n 1 -P 2 bash subrun.sh $next_dir
	if [[ `ls final.*.h5 | wc -l` -ne `ls current.*.h5 | wc -l` ]]; then
	    echo "finals where not created"
	    ls
	    exit 1
	fi
	for y in final.*.h5; do
	    mv $y `echo $y | sed 's/final/current/g'`
	done

	# 2
	rm -r curr
	if [ -d curr ]; then
	    echo "curr was not properly deleted"
	    exit 1
	fi

	# 3
	if [ ! -d next ]; then
	    echo "next does not exist"
	    exit 1
	fi
	mv next curr

    done

    for y in current.*.h5; do
	cp $y `echo $y | sed "s/current/epoch_$epoch/g"`
    done
done
