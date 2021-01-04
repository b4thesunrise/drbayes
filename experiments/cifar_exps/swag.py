import argparse
import os, sys
import time
import tabulate

import torch
import torch.nn.functional as F
import torchvision
import numpy as np

from subspace_inference.dataset.mini_imagenet import ImageNet, MetaImageNet
from subspace_inference.dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from subspace_inference.dataset.cifar import CIFAR100, MetaCIFAR100
from subspace_inference.dataset.transform_cfg import transforms_options, transforms_list
from subspace_inference import data, models, utils, losses
from subspace_inference.posteriors import SWAG
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=8, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.01, metavar='LR', help='initial learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

parser.add_argument('--swag', action='store_true')
parser.add_argument('--swag_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swag_lr', type=float, default=0.02, metavar='LR', help='SWA LR (default: 0.02)')
parser.add_argument('--swag_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--cov_mat', action='store_true', help='save sample covariance')
parser.add_argument('--max_num_models', type=int, default=20, help='maximum number of SWAG models to save')

parser.add_argument('--swag_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to restor SWA from (default: None)')
parser.add_argument('--loss', type=str, default='CE', help='loss to use for training model (default: Cross-entropy)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--no_schedule', action='store_true', help='store schedule')

parser.add_argument('--save_iterates', action='store_true', help='save all iterates in the SWA(G) stage (default: off)')
parser.add_argument('--inference', choices=['low_rank_gaussian', 'projected_sgd'], default='low_rank_gaussian')
parser.add_argument('--subspace', choices=['covariance', 'pca', 'freq_dir'], default='covariance')

#from rfs

parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
parser.add_argument('--use_trainval', action='store_true', help='use trainval set')
parser.add_argument('--lr_step', action='store_true', help='use trainval set')
parser.add_argument('--swa_lr_step', action='store_true', help='use trainval set')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)',choices=['miniImageNet', 'tieredImageNet',
                                                                                                              'CIFAR-FS', 'FC100', 'CIFAR10', 'CIFAR100'])
parser.add_argument('--data_root', type=str, default='', help='path to data root')
# meta setting
parser.add_argument('--adam', action='store_true', help='use adam optimizer')
parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
# parser.add_argument('--lr_decay_epochs', type=str, default='70,90,100', help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                    help='Number of test runs')
parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                    help='Number of classes for doing each classification run')
parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                    help='Number of shots in test')
parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                    help='Number of query in test')
parser.add_argument('--n_aug_support_samples', default=5, type=int,
                    help='The number of augmented samples for each meta test sample')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                    help='Size of test batch)')
args = parser.parse_args()

if args.use_trainval:
    args.trial = args.trial + '_trainval'
if args.dataset == 'CIFAR-FS' or args.dataset == 'FC100':
    args.transform = 'D'
args.data_aug = True

train_partition = 'trainval' if args.use_trainval else 'train'
if args.dataset == 'miniImageNet':
    train_trans, test_trans = transforms_options[args.transform]
    train_loader = DataLoader(ImageNet(args=args, partition=train_partition, transform=train_trans),
                              batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(ImageNet(args=args, partition='val', transform=test_trans),
                            batch_size=args.batch_size // 2, shuffle=False, drop_last=False,
                            num_workers=args.num_workers // 2)
    meta_testloader = DataLoader(MetaImageNet(args=args, partition='test',
                                              train_transform=train_trans,
                                              test_transform=test_trans),
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.num_workers)
    meta_valloader = DataLoader(MetaImageNet(args=args, partition='val',
                                             train_transform=train_trans,
                                             test_transform=test_trans),
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers)
    if args.use_trainval:
        num_classes = 80
    else:
        num_classes = 64
elif args.dataset == 'tieredImageNet':
    train_trans, test_trans = transforms_options[args.transform]
    train_loader = DataLoader(TieredImageNet(args=args, partition=train_partition, transform=train_trans),
                              batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(TieredImageNet(args=args, partition='train_phase_val', transform=test_trans),
                            batch_size=args.batch_size // 2, shuffle=False, drop_last=False,
                            num_workers=args.num_workers // 2)
    meta_testloader = DataLoader(MetaTieredImageNet(args=args, partition='test',
                                                    train_transform=train_trans,
                                                    test_transform=test_trans),
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.num_workers)
    meta_valloader = DataLoader(MetaTieredImageNet(args=args, partition='val',
                                                   train_transform=train_trans,
                                                   test_transform=test_trans),
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers)
    if args.use_trainval:
        num_classes = 448
    else:
        num_classes = 351
elif args.dataset == 'CIFAR-FS' or args.dataset == 'FC100':
    train_trans, test_trans = transforms_options['D']

    train_loader = DataLoader(CIFAR100(args=args, partition=train_partition, transform=train_trans),
                              batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(CIFAR100(args=args, partition='train', transform=test_trans),
                            batch_size=args.batch_size // 2, shuffle=False, drop_last=False,
                            num_workers=args.num_workers // 2)
    meta_testloader = DataLoader(MetaCIFAR100(args=args, partition='test',
                                              train_transform=train_trans,
                                              test_transform=test_trans),
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.num_workers)
    meta_valloader = DataLoader(MetaCIFAR100(args=args, partition='val',
                                             train_transform=train_trans,
                                             test_transform=test_trans),
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers)
    if args.use_trainval:
        num_classes = 80
    else:
        if args.dataset == 'CIFAR-FS':
            num_classes = 64
        elif args.dataset == 'FC100':
            num_classes = 60
        else:
            raise NotImplementedError('dataset not supported: {}'.format(args.dataset))
else:
    raise NotImplementedError(args.dataset)


# assert

args.device = None

if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

# print('Loading dataset %s from %s' % (args.dataset, args.data_path))
# loaders, num_classes = data.loaders(
#     args.dataset,
#     args.data_path,
#     args.batch_size,
#     args.num_workers,
#     model_cfg.transform_train,
#     model_cfg.transform_test,
#     use_validation=not args.use_test,
#     split_classes=args.split_classes
# )

print('Preparing model')
print(*model_cfg.args)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)


if args.cov_mat:
    args.no_cov_mat = False
else:
    args.no_cov_mat = True
if args.swag:
    print('SWAG training')
    swag_model = SWAG(model_cfg.base,
                    args.subspace, subspace_kwargs={'max_rank': args.max_num_models}, num_classes=num_classes,
             args=model_cfg.args,  kwargs=model_cfg.kwargs)
    swag_model.to(args.device)
else:
    print('SGD training')


def schedule(epoch, args):
    steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
    t = (epoch) / (args.swag_start if args.swag else args.epochs)
    lr_ratio = args.swag_lr / args.lr_init if args.swag else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    factor = factor if args.swa_lr_step else 1
    lr_factor = args.lr_decay_rate ** steps if args.lr_step else 1
    return args.lr_init * factor * lr_factor

# use a slightly modified loss function that allows input of model 
if args.loss == 'CE':
    criterion = losses.cross_entropy
    #criterion = F.cross_entropy
elif args.loss == 'adv_CE':
    criterion = losses.adversarial_cross_entropy
    
# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=args.lr_init,
#     momentum=args.momentum,
#     weight_decay=args.wd
# )

# optimizer
if args.adam:
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr_init,
                                 weight_decay=0.0005)
else:
    optimizer = torch.optim.SGD(model.parameters(),
                          lr=args.lr_init,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

# criterion = torch.nn.CrossEntropyLoss()
iterations = args.lr_decay_epochs.split(',')
args.lr_decay_epochs = list([])
for it in iterations:
    args.lr_decay_epochs.append(int(it))
start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if args.swag and args.swag_resume is not None:
    checkpoint = torch.load(args.swag_resume)
    swag_model.subspace.rank = torch.tensor(0)
    swag_model.load_state_dict(checkpoint['state_dict'])

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time', 'mem_usage']
if args.swag:
    columns = columns[:-2] + ['swa_te_loss', 'swa_te_acc'] + columns[-2:]
    swag_res = {'loss': None, 'accuracy': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict()
)

sgd_ens_preds = None
sgd_targets = None
n_ensembled = 0.

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    if not args.no_schedule:
        lr = schedule(epoch, args)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init
    
    if (args.swag and (epoch + 1) > args.swag_start) and args.cov_mat:
        train_res = utils.train_epoch(train_loader, model, criterion, optimizer, epoch)
    else:
        train_res = utils.train_epoch(train_loader, model, criterion, optimizer, epoch)

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(val_loader, model, criterion)
        utils.mata_eval(model, meta_valloader, meta_testloader, 'BASELINE')

    else:
        test_res = {'loss': None, 'accuracy': None}

    if args.swag and (epoch + 1) > args.swag_start and (epoch + 1 - args.swag_start) % args.swag_c_epochs == 0:
        sgd_preds, sgd_targets = utils.predictions(val_loader, model)
        # sgd_res = utils.predict(val_loader, model)
        # sgd_preds = sgd_res["predictions"]
        # sgd_targets = sgd_res["targets"]
        # print("updating sgd_ens")
        if sgd_ens_preds is None:
            sgd_ens_preds = sgd_preds.copy()
        else:
            #TODO: rewrite in a numerically stable way
            sgd_ens_preds = sgd_ens_preds * n_ensembled / (n_ensembled + 1) + sgd_preds/ (n_ensembled + 1)
        n_ensembled += 1
        swag_model.collect_model(model)
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            # swag_model.set_swa()
            swag_model.sample(0.0)
            utils.bn_update(train_loader, swag_model)
            swag_res = utils.eval(val_loader, swag_model, criterion)
            utils.mata_eval(swag_model, meta_valloader, meta_testloader, 'SWAG', classifier='LR')
        else:
            swag_res = {'loss': None, 'accuracy': None}

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        if args.swag:
            utils.save_checkpoint(
                args.dir,
                epoch + 1,
                name='swag',
                state_dict=swag_model.state_dict(),
            )
            
    elif args.save_iterates:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )

    time_ep = time.time() - time_ep
    memory_usage = torch.cuda.memory_allocated()/(1024.0 ** 3)
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'], time_ep, memory_usage]
    if args.swag:
        values = values[:-2] + [swag_res['loss'], swag_res['accuracy']] + values[-2:]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )
    if args.swag and args.epochs > args.swag_start:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            name='swag',
            state_dict=swag_model.state_dict(),
        )

        utils.set_weights(model, swag_model.mean)
        utils.bn_update(train_loader, model)
        print("SWA solution")
        print(utils.eval(val_loader, model, losses.cross_entropy))

        utils.save_checkpoint(
            args.dir,
            name='swa',
            state_dict=model.state_dict(),
        )

if args.swag:
    np.savez(os.path.join(args.dir, "sgd_ens_preds.npz"), predictions=sgd_ens_preds, targets=sgd_targets)
