import subprocess

"""
Default value
"""
n_range = 50
# train_data = f'egg/zoo/sum_game/data_generation_scripts/sum_dataset_dumb_n{n_range}_train.txt'
# validation_data = f'egg/zoo/sum_game/data_generation_scripts/sum_dataset_dumb_n{n_range}_train.txt'
train_data = f'egg/zoo/sum_game/data_generation_scripts/sum_dataset_standard_n{n_range}_train.txt'
validation_data = f'egg/zoo/sum_game/data_generation_scripts/sum_dataset_standard_n{n_range}_test.txt'
mode = 'rf'
n_epochs = 5000
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
# n_theo = n_range * 2 - 1
# for s in [0, 1]:
#     for v_r in [0.5, 1, 2, 4, 8, 32]:
#         for l in [1,2,3,4]:
#             for w in [0.6]:
#                 if l == 1 and v_r in [0.5, 1, 2, 4]:
#                     continue
#                 balance_ce = w
#                 max_len = l
#                 vocab_size = round(int(n_theo*v_r)**(1/l))
#                 seed = s
#                 name = f'n{n_range}_{mode}_msg{max_len}_r{v_r}_vocab{vocab_size}_balance{balance_ce}_seed{seed}'
#                 bashCommand = f"python -m egg.zoo.sum_game.play --mode {mode} --train_data {train_data} --validation_data {validation_data} --n_range {n_range} --n_epochs {n_epochs} --batch_size {batch_size} --validation_batch_size {validation_batch_size} --max_len {max_len} --vocab_size {vocab_size} --sender_hidden {sender_hidden} --receiver_hidden {receiver_hidden} --sender_embedding {sender_embedding} --receiver_embedding {receiver_embedding} --receiver_cell {receiver_cell} --sender_cell {sender_cell} --lr {lr} --sender_entropy_coeff {send_ent} --temperature {temperature} --loss {loss} --balanced_ce {balance_ce} --random_seed {seed} --tensorboard --tensorboard_dir=./runs/{name}/"
#                 process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#                 output, error = process.communicate()
"""
"""
# n_theo = n_range * 2 - 1
# n_epochs = 2000
# for s in [1]:
#     for l in [5]:
#         for w in [0.6]:
#             for e in [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22]:
#                 for v in [2]:
#                     balance_ce = w
#                     max_len = l
#                     vocab_size = v#round(int(n_theo*v_r)**(1/l))
#                     send_ent = e
#                     seed = s
#                     name = f'n{n_range}_{mode}_msg{max_len}_vocab{vocab_size}_balance{balance_ce}_e{send_ent}_seed{seed}'
#                     bashCommand = f"python -m egg.zoo.sum_game.play --mode {mode} --train_data {train_data} --validation_data {validation_data} --n_range {n_range} --n_epochs {n_epochs} --batch_size {batch_size} --validation_batch_size {validation_batch_size} --max_len {max_len} --vocab_size {vocab_size} --sender_hidden {sender_hidden} --receiver_hidden {receiver_hidden} --sender_embedding {sender_embedding} --receiver_embedding {receiver_embedding} --receiver_cell {receiver_cell} --sender_cell {sender_cell} --lr {lr} --sender_entropy_coeff {send_ent} --temperature {temperature} --loss {loss} --balanced_ce {balance_ce} --random_seed {seed} --tensorboard --tensorboard_dir=./runs/{name}/"
#                     process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#                     output, error = process.communicate()
"""
"""
# n_theo = n_range * 2 - 1
# n_epochs = 3000
# send_ent = 0.18
# balance_ce = 0.6 if n_range==20 else 0.3
# vocab_size_list = {
#     20:{1:[19, 39, 78, 156, 312, 1248], 2:[4, 6, 9, 12, 18, 36], 3:[2,3,4,5,7,11], 4:[2, 3, 4, 5, 6, 7]},
#     50:{1:[19, 39, 78, 156, 312, 1248], 2:[4, 6, 9, 12, 18, 36], 3:[2,3,4,5,7,11], 4:[2, 3, 4, 5, 6, 7]},
#     100:{1:[99, 199, 398, 796, 1592, 6368, 12736, 25472], 2:[10, 14, 20, 28, 40, 79, 113, 160], 3:[4, 5, 6, 9, 12, 19, 23, 29], 4:[3,4,5,6,7,9, 10, 12]}}
# for s in [2]:
#     for l in [1, 2, 3, 4]:
#         for v in vocab_size_list[n_range][l]:
#             # Heuristic to adapt send_ent to the vocab size
#             if n_range == 20:
#                 if v < 40:
#                     send_ent = 0.18
#                 elif v < 80:
#                     send_ent = 0.15
#                 elif v < 1000:
#                     send_ent = 0.1
#                 else:
#                     send_ent = 0.06
#             elif n_range == 100:
#                 if v < 40:
#                     send_ent = 0.2
#                 elif v < 200:
#                     send_ent = 0.14
#                 else:
#                     send_ent = 0.12
#             max_len = l
#             vocab_size = v
#             n_combi = vocab_size ** max_len
#             seed = s
#             name = f'n{n_range}_{mode}_msg{max_len}_combi{n_combi}_vocab{vocab_size}_balance{balance_ce}_e{send_ent}_seed{seed}'
#             bashCommand = f"python -m egg.zoo.sum_game.play --mode {mode} --train_data {train_data} --validation_data {validation_data} --n_range {n_range} --n_epochs {n_epochs} --batch_size {batch_size} --validation_batch_size {validation_batch_size} --max_len {max_len} --vocab_size {vocab_size} --sender_hidden {sender_hidden} --receiver_hidden {receiver_hidden} --sender_embedding {sender_embedding} --receiver_embedding {receiver_embedding} --receiver_cell {receiver_cell} --sender_cell {sender_cell} --lr {lr} --sender_entropy_coeff {send_ent} --temperature {temperature} --loss {loss} --balanced_ce {balance_ce} --random_seed {seed} --tensorboard --tensorboard_dir=./runs/{name}/"
#             process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#             output, error = process.communicate()
"""
"""
n_theo = n_range * 2 - 1
n_epochs = 3000
send_ent = 0.18
balance_ce = 0.6 if n_range==20 else 0.3
vocab_size_list = {
    20:{1:[19, 39, 78, 156, 312, 1248], 2:[4, 6, 9, 12, 18, 36], 3:[2,3,4,5,7,11], 4:[2, 3, 4, 5, 6, 7], 5:[2,3,4], 6:[2,3,4], 7:[2,3], 8:[2,3]},
    50:{1:[99, 199],},
    100:{1:[99, 199, 398, 796, 1592, 6368, 12736, 25472], 2:[10, 14, 20, 28, 40, 79, 113, 160], 3:[4, 5, 6, 9, 12, 19, 23, 29], 4:[3,4,5,6,7,9, 10, 12]}}
for s in [0]:
    for l in [1]:
        for v in vocab_size_list[n_range][l]:
            for e in [0.02, 0.06, 0.1, 0.14, 0.18, 0.22]:    
                send_ent = e
                max_len = l
                vocab_size = v
                n_combi = vocab_size ** max_len
                seed = s
                name = f'n{n_range}_{mode}_msg{max_len}_combi{n_combi}_vocab{vocab_size}_balance{balance_ce}_e{send_ent}_seed{seed}'
                bashCommand = f"python -m egg.zoo.sum_game.play --mode {mode} --train_data {train_data} --validation_data {validation_data} --n_range {n_range} --n_epochs {n_epochs} --batch_size {batch_size} --validation_batch_size {validation_batch_size} --max_len {max_len} --vocab_size {vocab_size} --sender_hidden {sender_hidden} --receiver_hidden {receiver_hidden} --sender_embedding {sender_embedding} --receiver_embedding {receiver_embedding} --receiver_cell {receiver_cell} --sender_cell {sender_cell} --lr {lr} --sender_entropy_coeff {send_ent} --temperature {temperature} --loss {loss} --balanced_ce {balance_ce} --random_seed {seed} --tensorboard --tensorboard_dir=./runs/{name}/"
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()