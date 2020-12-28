import argparse
import os, sys
import math
import time
import tabulate

import torch

import numpy as np

from subspace_inference import data, models, utils, losses
from subspace_inference.posteriors import SWAG
from subspace_inference.posteriors.vi_model import VIModel, ELBO
from subspace_inference.posteriors.proj_model import SubspaceModel

from subspace_inference.dataset.mini_imagenet import ImageNet, MetaImageNet
from subspace_inference.dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from subspace_inference.dataset.cifar import CIFAR100, MetaCIFAR100
from subspace_inference.dataset.transform_cfg import transforms_options, transforms_list
from torch.utils.data import DataLoader
import sklearn.decomposition

def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.mean(np.log(ps + 1e-12))
    return nll


parser = argparse.ArgumentParser(description='Subspace VI')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
#parser.add_argument('--log_fname', type=str, default=None, required=True, help='file name for logging')

# parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')

parser.add_argument('--use_test', dest='use_test', action='store_true',
                    help='use test dataset instead of validation (default: False)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs (default: 50')
parser.add_argument('--num_samples', type=int, default=30, metavar='N', help='number of epochs (default: 30')

parser.add_argument('--temperature', type=float, default=1., required=True, 
                    metavar='N', help='Temperature (default: 1.')
parser.add_argument('--no_mu', action='store_true', help='Do not learn the mean of posterior')

parser.add_argument('--rank', type=int, default=2, metavar='N', help='approximation rank (default: 2')
parser.add_argument('--random', action='store_true')

parser.add_argument('--prior_std', type=float, default=1.0, help='std of the prior distribution')
parser.add_argument('--init_std', type=float, default=1.0, help='initial std of the variational distribution')

parser.add_argument('--checkpoint', type=str, default=None, required=True, help='path to SWAG checkpoint')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--bn_subset', type=float, default=1.0, help='BN subset for evaluation (default 1.0)')
parser.add_argument('--max_rank', type=int, default=20, help='maximum rank')

#from rfs
parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
parser.add_argument('--use_trainval', action='store_true', help='use trainval set')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)',choices=['miniImageNet', 'tieredImageNet',
                                                                                                              'CIFAR-FS', 'FC100', 'CIFAR10', 'CIFAR100'])
parser.add_argument('--data_root', type=str, default='', help='path to data root')
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
    bn_loader = DataLoader(CIFAR100(args=args, partition='train', transform=test_trans),
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

def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.mean(np.log(ps + 1e-12))
    return nll

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

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
# loaders, num_classes = data.loaders(
#     args.dataset,
#     args.data_path,
#     args.batch_size,
#     args.num_workers,
#     transform_train=model_cfg.transform_test,
#     transform_test=model_cfg.transform_test,
#     shuffle_train=False,
#     use_validation=not args.use_test,
#     split_classes=args.split_classes
# )

# loaders_bn, _ = data.loaders(
#     args.dataset,
#     args.data_path,
#     args.batch_size,
#     args.num_workers,
#     transform_train=model_cfg.transform_train,
#     transform_test=model_cfg.transform_test,
#     shuffle_train=True,
#     use_validation=not args.use_test,
#     split_classes=args.split_classes
# )

print('Preparing model')
print(*model_cfg.args)

swag_model = SWAG(
    model_cfg.base,
    subspace_type='pca',
    subspace_kwargs={
        'max_rank': args.max_rank,
        'pca_rank': args.rank,
    },
    num_classes=num_classes,
    args=model_cfg.args, kwargs=model_cfg.kwargs
)

swag_model.to(args.device)

print('Loading: %s' % args.checkpoint)
ckpt = torch.load(args.checkpoint)
swag_model.load_state_dict(ckpt['state_dict'], strict=False)

# # first take as input SWA
# swag_model.set_swa()
# utils.bn_update(train_loader, swag_model)
# # print(utils.eval(meta_testloader, swag_model, losses.cross_entropy))
# utils.mata_eval(swag_model, meta_valloader, meta_testloader, 'SWA:')
# swag_model.sample(0.0)
# utils.bn_update(train_loader, swag_model)
# # print(utils.eval(meta_testloader, swag_model, losses.cross_entropy))
# utils.mata_eval(swag_model, meta_valloader, meta_testloader, 'SWAG-0.0:')
# swag_model.sample(0.5)
# utils.bn_update(train_loader, swag_model)
# # print(utils.eval(meta_testloader, swag_model, losses.cross_entropy))
# utils.mata_eval(swag_model, meta_valloader, meta_testloader, 'SWAG-0.5:')
# swag_model.sample(1.0)
# utils.bn_update(train_loader, swag_model)
# # print(utils.eval(meta_testloader, swag_model, losses.cross_entropy))
# utils.mata_eval(swag_model, meta_valloader, meta_testloader, 'SWAG-1.0:')

mean, var, cov_factor = swag_model.get_space()

print(torch.norm(cov_factor, dim=1))
#print(var)

if args.random:
    scale = 0.5 * (np.linalg.norm(cov_factor[1, :]) + np.linalg.norm(cov_factor[0, :]))
    print(scale)
    np.random.seed(args.seed)
    cov_factor = np.random.randn(*cov_factor.shape)


    tsvd = sklearn.decomposition.TruncatedSVD(n_components=args.rank, n_iter=7, random_state=args.seed)
    tsvd.fit(cov_factor)

    cov_factor = tsvd.components_
    cov_factor /= np.linalg.norm(cov_factor, axis=1, keepdims=True)
    cov_factor *= scale

    print(cov_factor[:, 0])

    cov_factor = torch.FloatTensor(cov_factor, device=mean.device)

vi_model = VIModel(
    subspace=SubspaceModel(mean.cuda(), cov_factor.cuda()),
    init_inv_softplus_sigma=math.log(math.exp(args.init_std) - 1.0),
    prior_log_sigma=math.log(args.prior_std),
    num_classes=num_classes,
    base=model_cfg.base,
    with_mu=not args.no_mu,
    args=model_cfg.args, kwargs=model_cfg.kwargs
)

vi_model = vi_model.cuda()
# print(utils.eval(train_loader, vi_model, criterion=losses.cross_entropy))
utils.mata_eval(vi_model, meta_valloader, meta_testloader, 'SWAG-vi-init:')

elbo = ELBO(losses.cross_entropy, len(train_loader.dataset), args.temperature)

#optimizer = torch.optim.Adam([param for param in vi_model.parameters()], lr=0.01)
optimizer = torch.optim.SGD([param for param in vi_model.parameters()], lr=args.lr, momentum=0.9)
# optimizer
# if args.adam:
#     optimizer = torch.optim.Adam(vi_model.parameters(),
#                                  lr=args.learning_rate,
#                                  weight_decay=0.0005)
# else:
#     optimizer = torch.optim.SGD(vi_model.parameters(),
#                           lr=args.learning_rate,
#                           momentum=args.momentum,
#                           weight_decay=args.weight_decay)
#printf, logfile = utils.get_logging_print(os.path.join(args.dir, args.log_fname + '-%s.txt'))
#print('Saving logs to: %s' % logfile)
columns = ['ep', 'acc', 'loss', 'kl', 'nll', 'sigma_1', 'time']

epoch = 0
for epoch in range(args.epochs):
    time_ep = time.time()
    train_res = utils.train_epoch(train_loader, vi_model, elbo, optimizer, epoch)
    time_ep = time.time() - time_ep
    sigma_1 = torch.nn.functional.softplus(vi_model.inv_softplus_sigma.detach().cpu())[0].item()
    values = ['%d/%d' % (epoch + 1, args.epochs), train_res['accuracy'], train_res['loss'],
              train_res['stats']['kl'], train_res['stats']['nll'], sigma_1, time_ep]
    if epoch == 0:
        print(tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f'))
    else:
        print(tabulate.tabulate([values], columns, tablefmt='plain', floatfmt='8.4f').split('\n')[1])



print("sigma:", torch.nn.functional.softplus(vi_model.inv_softplus_sigma.detach().cpu()))
if not args.no_mu:
    print("mu:", vi_model.mu.detach().cpu().data)

utils.save_checkpoint(
    args.dir,
    epoch,
    name='vi',
    state_dict=vi_model.state_dict()
)


eval_model = model_cfg.base(num_classes=num_classes, *model_cfg.args, **model_cfg.kwargs)
eval_model.to(args.device)

num_samples = args.num_samples

query_length = len(meta_testloader.dataset) * len(meta_testloader.dataset[0][-1])
ens_predictions = np.zeros((query_length, args.n_ways))
ens_predictions_feats = np.zeros((query_length, args.n_ways))
targets = np.zeros((query_length , args.n_ways))
targets_feats = np.zeros((query_length , args.n_ways))



#printf, logfile = utils.get_logging_print(os.path.join(args.dir, args.log_fname + '-%s.txt'))
#print('Saving logs to: %s' % logfile)
columns = ['iter ens', 'acc', 'nll']

for i in range(num_samples):
    with torch.no_grad():
        w = vi_model.sample()
        offset = 0
        for param in eval_model.parameters():
            param.data.copy_(w[offset:offset+param.numel()].view(param.size()).to(args.device))
            offset += param.numel()

    utils.bn_update(train_loader, eval_model, subset=args.bn_subset)


    pred_res = utils.predict(meta_testloader, eval_model)
    pred_res_feats = utils.predict(meta_testloader, eval_model, use_logit=False)
    utils.mata_eval(eval_model, meta_valloader, meta_testloader, 'SWAG-vi')
    ens_predictions += pred_res['predictions']
    ens_predictions_feats += pred_res_feats['predictions']
    targets = pred_res['targets']
    targets_feats = pred_res_feats['targets']

    values = ['%3d/%3d' % (i + 1, num_samples),
              np.mean(np.argmax(ens_predictions, axis=1) == targets),
              nll(ens_predictions / (i + 1), targets)]
    values_feats = ['%3d/%3d' % (i + 1, num_samples),
              np.mean(np.argmax(ens_predictions_feats, axis=1) == targets_feats),
              nll(ens_predictions_feats / (i + 1), targets_feats)]
    table = tabulate.tabulate([values] + [values_feats], columns, tablefmt='simple', floatfmt='8.4f')
    if i == 0:
        print(table)
    else:
        print(table.split('\n')[2])

ens_predictions /= num_samples
ens_acc = np.mean(np.argmax(ens_predictions, axis=1) == targets)
ens_nll = nll(ens_predictions, targets)
print("Ensemble NLL:", ens_nll)
print("Ensemble Accuracy:", ens_acc)

print("Ensemble Acc:", ens_acc)
print("Ensemble NLL:", ens_nll)

ens_predictions_feats /= num_samples
ens_acc_feats = np.mean(np.argmax(ens_predictions_feats, axis=1) == targets_feats)
ens_nll_feats = nll(ens_predictions_feats, targets_feats)
print("Ensemble NLL Feats:", ens_nll_feats)
print("Ensemble Accuracy Feats:", ens_acc_feats)

print("Ensemble Acc Feats:", ens_acc_feats)
print("Ensemble NLL Feats:", ens_nll_feats)
np.savez(
    os.path.join(args.dir, 'ens.npz'),
    seed=args.seed,
    ens_predictions=ens_predictions_feats,
    targets=targets,
    ens_acc=ens_acc,
    ens_nll=ens_nll
)
