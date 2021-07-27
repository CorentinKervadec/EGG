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
from egg.zoo.sum_game.architectures import DiscriReceiver, RecoReceiver, Sender, SenderOracle
from egg.zoo.sum_game.data_readers import SumDataset, BinarySumDataset


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
        h, b = np.histogram(sample_all, list(range(opts.n_range)))
        balance_weights = torch.from_numpy(1/h**opts.balanced_ce).long()
        ptint('Balance weights: ', balance_weights)
    else:
        balance_weights = None
    def loss(
        sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
    ):
        ## Debug: directly compute the sum of the input
        # n = sender_input.size(-1)/2
        # ar = torch.arange(n).to(sender_input.device)
        # ar = torch.cat([ar, ar])
        # ar = torch.stack([ar]*sender_input.size(0), dim=0)
        # decoded = (sender_input*ar).sum(-1).long()
        # print('dec', decoded)
        # print('label', labels)

        if opts.loss == 'xent':  
            # in the sum game case, accuracy is computed by comparing the index with highest score in Receiver output (a distribution of unnormalized
            # probabilities over target poisitions) and the corresponding label read from input, indicating the ground-truth position of the target
            acc = (receiver_output.argmax(dim=1) == labels).detach().float()
            # We also compute the absolute difference between predition and target, as an additional metric
            dist = (receiver_output.argmax(dim=1) - labels).detach().float().abs()
            # print('pred', receiver_output.argmax(dim=1))
            # print('labels', labels)
            # similarly, the loss computes cross-entropy between the Receiver-produced target-position probability distribution and the labels
            loss = F.cross_entropy(receiver_output, labels, weight=balance_weights, reduction="none")
        # elif opts.loss == 'bce':
        #     def vec_bin_array(arr, m):
        #         """
        #         Arguments: 
        #         arr: Numpy array of positive integers
        #         m: Number of bits of each integer to retain

        #         Returns a copy of arr with every element replaced with a bit vector.
        #         Bits encoded as int8's.
        #         """
        #         to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
        #         strs = to_str_func(arr)
        #         ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
        #         for bit_ix in range(0, m):
        #             fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        #             ret[...,bit_ix] = fetch_bit_func(strs).astype("int8")
        #         return ret 
        #     def bin_2_int(bin_arr, batch_size):
        #         ar = np.stack([2**np.arange(4)[ : :-1]]*batch_size, axis=0)
        #         int_val = (ar*bin_array).sum(axis=-1)
        #         return torch.from_numpy(int_val)
        #     m = int(np.log2(2*opts.n_range)+1)
        #     bin_label = vec_bin_array(labels.numpy())

        # elif opts.loss == 'mse':
        #     q_receiver_output = receiver_output.squeeze().round().float()
        #     acc = (q_receiver_output == labels).detach().float()
        #     dist = (q_receiver_output - labels).detach().float().abs()
        #     loss = F.mse_loss(receiver_output.squeeze(), labels.float(), reduction='mean')
        # TODO passer la target en binaire et superviser ave BCE
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
            path=opts.train_data,
            n_range=opts.n_range,
        ),
        batch_size=opts.validation_batch_size,
        shuffle=False,
        num_workers=1,
    )
    print('Initialising game...')
    if opts.max_len==0: #symbolic
        
        n_features = 2 * opts.n_range
        # n_features = 2*int(np.log2(opts.n_range)+1)
        # n_features_rec = int(np.log2(2*opts.n_range)+1)
        if opts.loss == 'xent':
            n_features_rec = n_features
        elif opts.loss == 'mse':
            n_features_rec = 1
        
        receiver = RecoReceiver(n_features=n_features_rec, n_hidden=opts.receiver_hidden)
        sender = Sender(n_hidden=opts.vocab_size, n_features=n_features, log_sftmx=True)

        if opts.mode.lower() == "gs":
            sender = core.GumbelSoftmaxWrapper(sender, temperature=opts.temperature)
            receiver = core.SymbolReceiverWrapper(receiver, vocab_size=opts.vocab_size, agent_input_size=opts.receiver_hidden)
            game = core.SymbolGameGS(sender, receiver, loss)
            callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
        else:  # NB: any other string than gs will lead to rf training!
            sender = core.ReinforceWrapper(sender)
            receiver = core.ReinforceDeterministicWrapper(receiver)
            receiver = core.SymbolReceiverWrapper(receiver, vocab_size=opts.vocab_size, agent_input_size=opts.receiver_hidden)
            game = core.SymbolGameReinforce(sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff, receiver_entropy_coeff=0.0)
            callbacks = []
    else:
        # the number of features for the Receiver (input) is given by 2*n_range because
        # they are fed concat 1-hot representations of the input vectors
        # It is similar for the sender as max sum is N+N=2N
        n_features = 2 * opts.n_range
        if opts.loss == 'xent':
            n_features_rec = n_features
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

    # Visu
    print('VISUALISATION')
    game.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for z in range(opts.vocab_size):
        t = torch.zeros(opts.vocab_size).to(device)
        t[z] = 1
        with torch.no_grad():
            sample, _1, _2 = game.receiver(t, None, None)
            sample = sample.float().cpu()
        sample = sample.argmax(dim=-1).item()
        print('msg[%d]->out[%d]'%(z, sample))
    print('end')

if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
