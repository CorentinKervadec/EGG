#!/bin/sh
source activate egg36
cd /home/ckervadec/EGG/
#python -m egg.zoo.basic_games.play --mode 'rf' --train_data "egg/zoo/basic_games/data_generation_scripts/example_reconstruction_input.txt" --validation_data "egg/zoo/basic_games/data_generation_scripts/example_reconstruction_input.txt" --n_attributes 5 --n_values 3 --n_epochs 200 --batch_size 512 --validation_batch_size 1000 --max_len 2 --vocab_size 200 --sender_hidden 256 --receiver_hidden 512 --sender_embedding 5 --receiver_embedding 30 --receiver_cell "gru" --sender_cell "gru" --lr 0.01 --print_validation_events --tensorboard --tensorboard_dir=./runs/basic_game_reco_m2_voc200/
#python -m egg.zoo.basic_games.play --game_type 'discri' --mode 'rf' --train_data "egg/zoo/basic_games/data_generation_scripts/example_discriminative_input.txt" --validation_data "egg/zoo/basic_games/data_generation_scripts/example_discriminative_input.txt" --n_values 3 --n_epochs 50 --batch_size 512 --validation_batch_size 10 --max_len 4 --vocab_size 100 --sender_hidden 256 --receiver_hidden 512 --sender_embedding 5 --receiver_embedding 30 --lr 0.01 --receiver_cell "gru" --sender_cell "gru" --random_seed 111 --print_validation_events --tensorboard --tensorboard_dir=./runs/basic_game_discri/
python -m egg.zoo.sum_game.launcher
#python -m egg.zoo.sum_game.play --mode 'rf' --train_data egg/zoo/sum_game/data_generation_scripts/sum_dataset_standard_n20_train.txt --validation_data egg/zoo/sum_game/data_generation_scripts/sum_dataset_standard_n20_test.txt --n_range 20 --n_epochs 250 --batch_size 512 --validation_batch_size 1000 --max_len 1 --vocab_size 40 --sender_hidden 256 --receiver_hidden 512 --sender_embedding 5 --receiver_embedding 30 --receiver_cell "gru" --sender_cell "gru" --lr 0.01 --tensorboard --tensorboard_dir=./runs/n_20_rf_m1_v40_4/ --print_validation_events --loss xent --sender_entropy_coeff 0.1
#python -m egg.nest.nest_local --game egg.zoo.sum_game.play --sweep egg/nest/sum_hp.json --n_workers=1
