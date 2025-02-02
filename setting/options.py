import argparse

parser = argparse.ArgumentParser('The training and evaluation script', add_help=False)
# training set
parser.add_argument('--epoch', type=int, default=40, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=128, help='training batch size')

parser.add_argument('--gpu_id', type=str, default='0', help='the gpu id')
parser.add_argument('--num_work',type=int, default=0)
parser.add_argument('--start_epoch',type=int, default=1)

# training dataset
parser.add_argument('--dataset', type=str, default='Berlin',help='Berlin')
parser.add_argument('--useval', type=int, default=0)
parser.add_argument('--save_path', type=str, default='./checkpoints/', help='the path to save models and logs')

parser.add_argument('--best_acc', type=float, default=0, help='save best accuracy')
parser.add_argument('--best_epoch', type=int, default=1, help='save best epoch')

opt = parser.parse_args()