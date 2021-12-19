import os
from tqdm import tqdm
from dataloader import Stardata
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from networks import *
import argparse
import time
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import copy

def print_square(dictionary):
    for key in dictionary.keys():
        if "float" in str(type(dictionary[key])):
            newval = round(float(dictionary[key]), 4)
            dictionary[key] = newval

    front_lens = []
    back_lens = []
    for key in dictionary.keys():
        front_lens.append(len(key))
        back_lens.append(len(str(dictionary[key])))
    front_len = max(front_lens)
    back_len = max(back_lens)

    strings = []
    for key in dictionary.keys():
        string = "| {0:<{2}} | {1:<{3}} |".format(key, dictionary[key], front_len, back_len)
        strings.append(string)

    max_len = max([len(i) for i in strings])
    print("-"*max_len)
    for string in strings:
        print(string)
    print("-" * max_len)


def run_epoch(model, loader, optimizer, cuda, test=False):
    if test:
        model.eval()
    else:
        model.train()

    history = {
        "accuracy": [],
        "loss": []
    }

    for mini_batch in tqdm(loader):
        if cuda:
            mini_batch = [batch.cuda() for batch in mini_batch]

        label = mini_batch[-1]
        inputs = mini_batch[:-1]

        pred = model(*inputs)
        loss = F.cross_entropy(pred, label)
        if not test:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        history['accuracy'].append((torch.argmax(pred, dim=1) == label).float().mean().item())
        history['loss'].append(loss.item())
    history['accuracy'] = sum(history['accuracy']) / len(history['accuracy'])
    history['loss'] = sum(history['loss']) / len(history['loss'])
    return history

def train(args, dataset):
    # set arguments
    split_ratio = args.split_ratio
    cuda = args.cuda
    max_epoch = args.max_epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    shuffle = args.shuffle
    id_string = "{}_{}_lr{}_wsize{}_{}_augment".format(args.preprocess_name, args.network_name, str(learning_rate), str(args.window_size), str(int(time.time()))[4:])

    args.id_string = id_string
    args.log_dir = os.path.join(args.log_dir, id_string)

    # log arguments
    logger = SummaryWriter(args.log_dir)
    args_print = {}
    for arg in vars(args):
        logger.add_text(arg, str(getattr(args, arg)), 0)
        args_print[arg] = getattr(args, arg)
    print_square(args_print)

    # set dataloaders
    num_samples = len(dataset)

    #train_idx, val_idx = train_test_split(list(range(num_samples)), train_size=split_ratio)
    point = int(num_samples * split_ratio)
    train_idx, val_idx = range(0, point), range(point, num_samples)

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, val_idx)

    train_weight = dataset.data['label_weight'][train_idx][:, 0].tolist()
    valid_weight = dataset.data['label_weight'][val_idx][:, 0].tolist()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        sampler=WeightedRandomSampler(train_weight, len(train_dataset))
    )

    '''
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=shuffle,
        sampler=WeightedRandomSampler(valid_weight, len(valid_dataset))
    )
    '''
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=shuffle
    )



    # make model
    model = eval(f"{args.network_name}('{args.races}')")
    if cuda:
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if cuda:
        model.cuda()

    # run
    for epoch in range(max_epoch):
        start_time = time.time()
        train_history = run_epoch(model, train_loader, optimizer, cuda)
        with torch.no_grad():
            valid_history = run_epoch(model, valid_loader, optimizer, cuda, test=True)

        history = {}
        for key in train_history.keys():
            history["train/"+key] = train_history[key]
        for key in valid_history.keys():
            history["valid/"+key] = valid_history[key]
        for key in history.keys():
            logger.add_scalar(key, history[key], epoch)

        history["epoch"] = f"{epoch+1}/{max_epoch}"
        history["time"] = round(time.time() - start_time, 2)
        print_square(history)

        torch.save(model.state_dict(), os.path.join(args.log_dir, "model.pth"))

def main(args):
    train(args, Stardata(args.load_dir, window_size=args.window_size, preprocess_func=args.preprocess_name))

if __name__ == "__main__":
    import config
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', default="./data/simple_preprocess/TZ", type=str)
    parser.add_argument('--log-dir', default="./trained_models/TZ", type=str)
    parser.add_argument('--races', default="TZ", type=str)
    parser.add_argument('--network-name', default="sequential_cnn", type=str)
    parser.add_argument('--preprocess-name', default="simple_preprocess", type=str)
    parser.add_argument('--split-ratio', default=0.8, type=float)
    parser.add_argument('--cuda', default=True, type=bool)
    parser.add_argument('--max-epoch', default=30, type=int)
    parser.add_argument('--learning-rate', default=0.0001, type=float)
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--window-size', default=5, type=int)
    parser.add_argument('--shuffle', default=False, type=bool)
    args = parser.parse_args()
    '''

    args = config.ManyinfoPreproc_VanillaResnet()
    main(args)

    '''
    args = config.SimplePreproc_VanillaCnn()
    learning_rates = [1e-4, 5e-5, 1e-5, 5e-6]
    dataset = Stardata(args.load_dir, window_size=args.window_size, preprocess_func=args.preprocess_name)
    for lr in learning_rates:
        temp_args = copy.deepcopy(args)
        temp_args.learning_rate = lr
        train(temp_args, dataset)
    '''


