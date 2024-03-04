import argparse
import os.path

from sdfl_api.standalone.AdaptiveFL.my_model_trainer import MyModelTrainer
from sdfl_api.standalone.AdaptiveFL.adaptivefl_api import AdaptiveflAPI
from sdfl_api.data_preprocessing.data import get_datasets, distribute_data_dirichlet, distribute_data, DatasetSplit
import torch
import logging
from torch.utils.data import DataLoader
from datetime import datetime


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument('--model', type=str, default='resnet9', metavar='N',
                        help="network architecture, supporting, 'resnet9', 'vgg16', 'resnet18, vgg11'")
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training, cifar10, cifar100, fmnist, tinyimagenet')
    parser.add_argument('--setting', type=str, default='dynamic-random-width', metavar='N',
                        help='dynamic-random-width, dynamic-fix-width, static-random-width, static-fix-width, no-communication')
    parser.add_argument('--frac_1', type=float, default=0.1, metavar='N',
                        help='if setting is dynamic-xxx, please set the proportion of clients whose width is 1 ')
    parser.add_argument('--n_classes', type=int, default=10, metavar='N',
                        help='local batch size for training')
    parser.add_argument('--width_list', type=float, default=[1, 0.75, 0.5, 0.25], metavar='N',
                        help='dataset used for training')
    parser.add_argument('--num_clients', type=int, default=40, metavar='N',
                        help='dataset used for training')
    parser.add_argument('--client_frac', type=int, default=0.2,
                        help="num of workers for multithreading")
    parser.add_argument('--comm_round', type=int, default=200,
                        help='total communication rounds')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='local training epochs for each client')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')
    parser.add_argument('--client_optimizer', type=str, default='SFW',
                        help='SGD with momentum, adam, sgd')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='local batch size for training')
    parser.add_argument('--non_iid', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=4,
                        help="num of workers for multithreading")
    # SWF
    parser.add_argument('--lr_decay', type=float, default=0.998, metavar='LR_decay',
                        help='learning rate decay')
    parser.add_argument('--wd', help='weight decay parameter', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='NN',
                        help='momentum')
    parser.add_argument('--lmo', type=str, default='KSupportNormBall',
                        help='SpectralKSupportNormBall, KSupportNormBall, GroupKSupportNormBall')
    parser.add_argument('--lmo_nuc_method', type=str, default='qrpartial',
                        help='qrpartial')
    parser.add_argument('--lmo_global', action='store_true', default=False)
    parser.add_argument('--lmo_k', type=float, default=0.4)
    parser.add_argument('--lmo_value', type=int, default=50)
    parser.add_argument('--lmo_mode', type=str, default='initialization',
                        help='initialization')
    parser.add_argument('--lmo_rescale', type=str, default='fast_gradient',
                        help='fast_gradient')
    parser.add_argument('--extensive_metrics', action='store_true', default=False)
    parser.add_argument('--lmo_adjust_diameter', action='store_true', default=False)
    parser.add_argument('--use_amp', action='store_true', default=True)
    return parser


def load_data(args):
    train_dataset, test_dataset = get_datasets(args.dataset)

    if args.dataset == "cifar100":
        num_target = 100
    elif args.dataset == "tinyimagenet":
        num_target = 200
    else:
        num_target = 10
    if args.non_iid:
        user_groups = distribute_data_dirichlet(train_dataset, args)
    else:
        user_groups = distribute_data(train_dataset, args, n_classes=num_target)
    train_loaders = []
    for idx in range(args.num_clients):
        train_dataset_idx = DatasetSplit(train_dataset, user_groups[idx], args=args)
        train_loader = DataLoader(train_dataset_idx, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=False, drop_last=True)
        train_loaders.append(train_loader)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=False)

    return train_loaders, test_loader


def custom_model_trainer(args, model, logger):
    return MyModelTrainer(model, args, logger)


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode='w', encoding='UTF-8')
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser(description='AdaptiveFL-standalone'))
    args = parser.parse_args()

    print("torch version{}".format(torch.__version__))
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    args.identity = "AdaptiveFL" + "-" + args.dataset + "-" + args.model
    log_path = './LOG/' + args.dataset + '/' + args.identity + '.log'
    if not os.path.exists('./LOG/' + args.dataset):
        os.makedirs('./LOG/' + args.dataset)
    logger = logger_config(log_path=log_path, logging_name=args.identity)
    logger.info(args)
    logger.info(device)
    args.client_num_per_round = int(args.num_clients * args.client_frac)

    dataset = load_data(args)
    print("start-time: ", datetime.now())
    adap_flAPI = AdaptiveflAPI(dataset, device, args, logger)
    adap_flAPI.train()
    adap_flAPI.test(global_model=None, round='Test', test=True,
                    width_lists=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13,
                                 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                 0.9, 1])
    print("end-time: ", datetime.now())
