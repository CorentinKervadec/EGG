import subprocess

"""
Default value
"""
n_range = 20
# train_data = f'egg/zoo/sum_game/data_generation_scripts/sum_dataset_dumb_n{n_range}_train.txt'
# validation_data = f'egg/zoo/sum_game/data_generation_scripts/sum_dataset_dumb_n{n_range}_train.txt'
train_data = f'egg/zoo/sum_game/data_generation_scripts/sum_dataset_standard_n{n_range}_train.txt'
validation_data = f'egg/zoo/sum_game/data_generation_scripts/sum_dataset_standard_n{n_range}_test.txt'
mode = 'rf'
n_epochs = 250
validation_batch_size = 1000
vocab_size = 40
sender_hidden = 256
receiver_hidden = 512
batch_size=512
lr=0.01
send_ent=1e-1
temperature=1.0
seed=0
loss='xent'
max_len = 1
balance_ce = -1
# if max_len > 1
sender_embedding = 5
receiver_embedding = 30
receiver_cell = 'gru'
sender_cell = 'gru'
"""
HP search
"""
for w in [0.5, 0.75, 1]:
    balance_ce = w
    name = f'n{n_range}_{mode}_msg{max_len}_vocab{vocab_size}_balance{balance_ce}'
    bashCommand = f"python -m egg.zoo.sum_game.play --mode {mode} --train_data {train_data} --validation_data {validation_data} --n_range {n_range} --n_epochs {n_epochs} --batch_size {batch_size} --validation_batch_size {validation_batch_size} --max_len {max_len} --vocab_size {vocab_size} --sender_hidden {sender_hidden} --receiver_hidden {receiver_hidden} --sender_embedding {sender_embedding} --receiver_embedding {receiver_embedding} --receiver_cell {receiver_cell} --sender_cell {sender_cell} --lr {lr} --sender_entropy_coeff {send_ent} --temperature {temperature} --loss {loss} --balanced_ce {balance_ce} --random_seed {seed} --tensorboard --tensorboard_dir=./runs/{name}/"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()