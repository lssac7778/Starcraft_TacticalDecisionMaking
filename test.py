
from networks import *
from dataloader import Stardata

dataset = Stardata(args.load_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--load-dir', default="./data/simple_preprocess/TZ", type=str)
parser.add_argument('--log-dir', default="./trained_models/TZ", type=str)
parser.add_argument('--split-ratio', default=0.8, type=float)
parser.add_argument('--cuda', default=True, type=bool)
parser.add_argument('--max_epoch', default=50, type=int)
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--shuffle', default=True, type=bool)
args = parser.parse_args()