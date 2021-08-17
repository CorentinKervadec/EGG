# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core
from egg.core import Callback, Interaction, PrintValidationEvents
from egg.zoo.sum_game.architectures import RecoReceiver, Sender, SenderOracle
from egg.zoo.sum_game.data_readers import SumDataset


# the following section specifies parameters that are specific to our games: we will also inherit the
# standard EGG parameters from https://github.com/facebookresearch/EGG/blob/master/egg/core/util.py
def get_params(params):
    parser = argparse.ArgumentParser()
    # arguments controlling the game type
    parser.add_argument(
        "--game_type",
        type=str,
        default="reco",
        help="Selects whether to play a reco(nstruction) or discri(mination) game (default: reco)",
    )
    # arguments concerning the input data and how they are processed
    parser.add_argument(
        "--train_data", type=str, default=None, help="Path to the train data"
    )
    parser.add_argument(
        "--validation_data", type=str, default=None, help="Path to the validation data"
    )
    
    parser.add_argument(
        "--n_range",
        type=int,
        default=None,
        help="Range of the input integer x in [0, N]",
    )

    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=0,
        help="Batch size when processing validation data, whereas training data batch_size is controlled by batch_size (default: same as training data batch size)",
    )
    # arguments concerning the training method
    parser.add_argument(
        "--mode",
        type=str,
        default="rf",
        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: rf)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        help="Supervision loss: xent or mse",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: 1.0)",
    )
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-1,
        help="Reinforce entropy regularization coefficient for Sender, only relevant in Reinforce (rf) mode (default: 1e-1)",
    )
    # arguments concerning the agent architectures
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
    )
    # arguments controlling the script output
    parser.add_argument(
        "--print_validation_events",
        default=False,
        action="store_true",
        help="If this flag is passed, at the end of training the script prints the input validation data, the corresponding messages produced by the Sender, and the output probabilities produced by the Receiver (default: do not print)",
    )
    parser.add_argument(
        "--analyse",
        default=False,
        action="store_true",
        help="If this flag is passed, analyse output at the end",
    )
    parser.add_argument(
        "--balanced_ce",
        type=float,
        default=-1,
        help="Weight the CE to balance the training between frequent and unfrequent labels. -1 to disable",
    )
    args = core.init(parser, params)
    return args


def main(params):
    opts = get_params(params)
    if opts.validation_batch_size == 0:
        opts.validation_batch_size = opts.batch_size
    print(opts, flush=True)

    print('Design loss...')
    if opts.balanced_ce > 0:
        sample_all = [[i+j for j in range(opts.n_range)] for i in range(opts.n_range)]
        sample_all = [item for sublist in sample_all for item in sublist]
        h, b = np.histogram(sample_all, list(range(opts.n_range*2)))
        balance_weights = torch.from_numpy(1/h**opts.balanced_ce).float()
        print('Balance weights: ', balance_weights)

    def loss(
        sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
    ):
        if opts.loss == 'xent':  
            # in the sum game case, accuracy is computed by comparing the index with highest score in Receiver output (a distribution of unnormalized
            # probabilities over target poisitions) and the corresponding label read from input, indicating the ground-truth position of the target
            acc = (receiver_output.argmax(dim=1) == labels).detach().float()
            # We also compute the absolute difference between predition and target, as an additional metric
            dist = (receiver_output.argmax(dim=1) - labels).detach().float().abs()
            # similarly, the loss computes cross-entropy between the Receiver-produced target-position probability distribution and the labels
            if opts.balanced_ce > 0:
                loss = F.cross_entropy(receiver_output, labels, weight=balance_weights.to(labels.device), reduction="none")
            else:
                loss = F.cross_entropy(receiver_output, labels, reduction="none")

        return loss, {"acc": acc, "dist": dist}

    # again, see data_readers.py in this directory for the AttValRecoDataset data reading class
    print('Building dataset...')
    train_loader = DataLoader(
        SumDataset(
            path=opts.train_data,
            n_range=opts.n_range,
        ),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=1,
    )
    test_loader = DataLoader(
        SumDataset(
            path=opts.validation_data,
            n_range=opts.n_range,
        ),
        batch_size=opts.validation_batch_size,
        shuffle=False,
        num_workers=1,
    )
    print('Initialising game...')
    # the number of features for the Receiver (input) is given by 2*n_range because
    # they are fed concat 1-hot representations of the input vectors
    # It is similar for the sender as max sum is N+N=2N
    n_features = 2 * opts.n_range
    if opts.loss == 'xent':
        n_features_rec = n_features - 1
    elif opts.loss == 'mse':
        n_features_rec = 1
    # we define here the core of the receiver for the discriminative game, see the architectures.py file for details
    # this will be embedded in a wrapper below to define the full architecture
    receiver = RecoReceiver(n_features=n_features_rec, n_hidden=opts.receiver_hidden)

    # we are now outside the block that defined game-type-specific aspects of the games: note that the core Sender architecture
    # (see architectures.py for details) is shared by the two games (it maps an input vector to a hidden layer that will be use to initialize
    # the message-producing RNN): this will also be embedded in a wrapper below to define the full architecture
    
    sender = Sender(n_hidden=opts.sender_hidden, n_features=n_features)

    # now, we instantiate the full sender and receiver architectures, and connect them and the loss into a game object
    # the implementation differs slightly depending on whether communication is optimized via Gumbel-Softmax ('gs') or Reinforce ('rf', default)
    if opts.mode.lower() == "gs":
        # in the following lines, we embed the Sender and Receiver architectures into standard EGG wrappers that are appropriate for Gumbel-Softmax optimization
        # the Sender wrapper takes the hidden layer produced by the core agent architecture we defined above when processing input, and uses it to initialize
        # the RNN that generates the message
        
        sender = core.RnnSenderGS(
            sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
            temperature=opts.temperature,
        )
        # the Receiver wrapper takes the symbol produced by the Sender at each step (more precisely, in Gumbel-Softmax mode, a function of the overall probability
        # of non-eos symbols upt to the step is used), maps it to a hidden layer through a RNN, and feeds this hidden layer to the
        # core Receiver architecture we defined above (possibly with other Receiver input, as determined by the core architecture) to generate the output
        receiver = core.RnnReceiverGS(
            receiver,
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.receiver_cell,
        )
        game = core.SenderReceiverRnnGS(sender, receiver, loss)
        # callback functions can be passed to the trainer object (see below) to operate at certain steps of training and validation
        # for example, the TemperatureUpdater (defined in callbacks.py in the core directory) will update the Gumbel-Softmax temperature hyperparameter
        # after each epoch
        callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
    else:  # NB: any other string than gs will lead to rf training!
        # here, the interesting thing to note is that we use the same core architectures we defined above, but now we embed them in wrappers that are suited to
        # Reinforce-based optmization
        # if opts.max_len>1:
        sender = core.RnnSenderReinforce(
            sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
        )
        receiver = core.RnnReceiverDeterministic(
            receiver,
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.receiver_cell,
        )
        game = core.SenderReceiverRnnReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=0,
        )
        callbacks = []


    # we are almost ready to train: we define here an optimizer calling standard pytorch functionality
    optimizer = core.build_optimizer(game.parameters())
    # in the following statement, we finally instantiate the trainer object with all the components we defined (the game, the optimizer, the data
    # and the callbacks)
    if opts.print_validation_events == True:
        # we add a callback that will print loss and accuracy after each training and validation pass (see ConsoleLogger in callbacks.py in core directory)
        # if requested by the user, we will also print a detailed log of the validation pass after full training: look at PrintValidationEvents in
        # language_analysis.py (core directory)
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks
            + [
                core.ConsoleLogger(print_train_loss=True, as_json=True),
                core.PrintValidationEvents(n_epochs=opts.n_epochs),
                core.MessageEntropy(print_train = True, is_gumbel = opts.mode.lower() == "gs")
            ],
        )
    else:
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks
            + [core.ConsoleLogger(print_train_loss=True, as_json=True)],
        )

    # and finally we train!
    print('Start training!')
    trainer.train(n_epochs=opts.n_epochs)

    if opts.analyse:

        import pandas as pd
        import seaborn as sns
        import matplotlib
        import matplotlib.pyplot as plt

        # Customize matplotlib
        matplotlib.rcParams.update(
            {
                'text.usetex': False,
                'font.family': 'stixgeneral',
                'mathtext.fontset': 'stix',
            }
        )

        print('Start Evaluating')
        mean_loss, full_interaction = trainer.eval()
        acc = full_interaction.aux['acc'].mean()
        dist = full_interaction.aux['dist'].mean()
        print('Accuracy: %.2f'%(acc*100))
        print('Distance: %.2f'%(dist))

        
        # Visualisation : (length=1 only)
        if opts.max_len==1:
            sender_input = full_interaction.sender_input
            datalength = sender_input.size(0)
            labels = full_interaction.labels
            ar = torch.arange(opts.n_range).to(sender_input.device)
            ar = torch.stack([ar]*sender_input.size(0), dim=0)
            input_pairs = torch.stack(((ar*sender_input[:, :opts.n_range]).sum(-1), (ar*sender_input[:, opts.n_range:]).sum(-1)), dim=-1).long()
            message = full_interaction.message[:,0].squeeze()
            receiver_prediction = full_interaction.receiver_output.argmax(dim=-1)

            m = torch.full((opts.n_range, opts.n_range), -1)
            s = torch.full((opts.n_range, opts.n_range), -1)
            msg2out = {}
            input2msg = {}
            output2msg = {}
            for i, p in enumerate(input_pairs):
                m[p[0], p[1]]=message[i]
                s[p[0], p[1]]=receiver_prediction[i]
                if message[i].item() not in msg2out:
                    msg2out[message[i].item()] = receiver_prediction[i].item()
                input_pair = (p[0].item(), p[1].item())
                if input_pair not in input2msg:
                    input2msg[input_pair] = message[i].item()
                output = receiver_prediction[i].item()
                if output not in output2msg:
                    output2msg[output] = message[i].item()

            
            # # Plot the communication
            fig = plt.figure()
            cmap = sns.color_palette("hsv", opts.vocab_size)
            if -1 in m:
                cmap[0] = (1., 1., 1., 1)
            # newcmp = ListedColormap(newcolors)
            # cmap = (0., 0., 0.) + list(cmap)
            ax = sns.heatmap(m, cmap=cmap, annot=False)
            # We change the fontsize of minor ticks label 
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=6)
            plt.savefig('analyse_message.pdf', dpi=800)
            # Plot prediction
            fig = plt.figure()
            ax = sns.heatmap(s, annot=False)
            # We change the fontsize of minor ticks label 
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=6)
            plt.savefig('analyse_pred.pdf', dpi=800)

            # How many symbols are used?
            used_symbols, counts_in = np.unique(list(input2msg.values()), return_counts=True)
            print('%d/%d symbols are used!'%(len(used_symbols), opts.vocab_size))
            # Is there any synonymy? (multiple symbols referring to the same input or output)
            out_val, counts_out = np.unique(list(msg2out.values()), return_counts=True)
            cpt_syn_out = sum([c>1 for c in counts_out])
            # uncomment to print the synonyms
            # print(''.join(['%d symbols -> %d\n'%(c, out_val[idx]) for idx, c in enumerate(counts_out) if c>1]))
            print('There is %d groups of synonyms (1 output is reffered by multiple different symbols)'%cpt_syn_out)
            # Is there any polysemy? (different inputs/outputs denoted by the same symbol)
            cpt_pol_in = sum([c>1 for c in counts_in])
            print('There is %d polysems (1 symbol refers to multiple different input'%cpt_pol_in)
            # agents are deterministics, then no synonyms relative to outputs and no polysems relative to input

if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
