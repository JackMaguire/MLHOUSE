#!/bin/bash

#First, let's carve out a working directory
mkdir demo
cd demo

# 1. Data Organization
# We're just using the sample data for now
head -n 100 ../sample_data/sample.repack.input.csv  > training_input.csv || exit 1
head -n 100 ../sample_data/sample.repack.output.csv > training_output.csv || exit 1

tail -n 50 ../sample_data/sample.repack.input.csv  > testing_input.csv || exit 1
tail -n 50 ../sample_data/sample.repack.output.csv > testing_output.csv || exit 1

echo "training_input.csv,training_output.csv" > training_data_files.csv || exit 1
echo "testing_input.csv,testing_output.csv"   > testing_data_files.csv || exit 1

# 2. Create a model
python3 ../training/create_blank_model.py --model starting_model || exit 1

# 2.5. Visualize the model (optional)
#python3 ../utilities/visualize_model.py starting_model.h5 || exit 1

# 3. Train the model
#for this demo, let's just run 10 epochs
python3 ../training/train.py --model ./starting_model.h5 --training_data ./training_data_files.csv --starting_epoch 0 --epoch_checkpoint_frequency 2 --num_epochs 10 || exit 1

# 4. Test the final model
python3 ../training/test.py --testing_data testing_data_files.csv --model final.h5 || exit 1
# OR
python3 ../training/test.py --testing_data testing_data_files.csv --model final.h5 --evaluate_individually true || exit 1
