import argparse
import random

"""
Generate train and test dataset for the sum game.
"""

def main():
    
    """
    Arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_file",
        type=str,
        default="sum_dataset",
        help="Output name",
    )
    parser.add_argument(
        "--n_range",
        type=int,
        default=None,
        help="Range of the integer [0, N[",
    )
    parser.add_argument(
        "--dumb",
        default=False,
        action="store_true",
        help="If this flag is passed, generate a dumb dataset x+0=x",
    )
    args = parser.parse_args()

    
    """
    Generate all samples of the dataset:
    x, y, x+y
    We add each addition in one direction only
    """
    sample_half = []
    sample_all = []
    for i in range(args.n_range):
        if args.dumb:
            sample_half.append([i, 0, i])
        else:
            sample_all.append([i, i, i+i])
            sample_half.append([i, i, i+i])
            for j in range(i):# we dont want to iterate on all the pairs: only in one direction
                sample_all.append([i, j, i+j])
                sample_all.append([j, i, j+i])
                p = random.random()
                if p>0.5: # we randomly assign x+y or y+x to avoid a potential confounder
                    sample_half.append([i, j, i+j])
                else:
                    sample_half.append([j, i, i+j])

    random.shuffle(sample_half) # shuffle
    
    """
    train/test split (70/30 if N is large enough)
    We make sure that any integer in the test has been observed in the train
    """
    train_dataset = []
    test_dataset = []
    check_integer_train = set() # to make sure that train contains each value
    check_sum_train = set()
    for sample in sample_half:
        """
        sample[0]: integer 1
        sample[1]: integer 2
        sample[2]: sum
        """
        if sample[2] not in check_sum_train\
        or sample[0] not in check_integer_train\
        or sample[1] not in check_integer_train: 
            # we make sure that all integer/sum appears at least one time in the train
            train_dataset.append(sample)
            check_integer_train.add(sample[0])
            check_integer_train.add(sample[1])
            check_sum_train.add(sample[2])
        else:
            p = random.random()
            if p > 0.3:#train
                train_dataset.append(sample)
            else:#test
                test_dataset.append(sample)
    print('Length train', len(train_dataset))
    print('Length test', len(test_dataset))

    """
    Save into text file
    """
    with open('./%s_n%d_train.txt'%(args.out_file, args.n_range), 'w') as f:
        for d in train_dataset:
            f.write('%d %d . %d\n'%(d[0], d[1], d[2]))
    with open('./%s_n%d_all.txt'%(args.out_file, args.n_range), 'w') as f:
        for d in sample_all:
            f.write('%d %d . %d\n'%(d[0], d[1], d[2]))
    with open('./%s_n%d_test.txt'%(args.out_file, args.n_range), 'w') as f:
        for d in test_dataset:
            f.write('%d %d . %d\n'%(d[0], d[1], d[2]))
    """
    inv_train is the same as train but xy are inverted (y, x, x+y)
    it will be used as sanity check during evaluation.
    """
    with open('./%s_n%d_inv_train.txt'%(args.out_file, args.n_range), 'w') as f:
        for d in train_dataset:
            f.write('%d %d . %d\n'%(d[1], d[0], d[2]))
    print('Done!')

if __name__ == "__main__":
    main()