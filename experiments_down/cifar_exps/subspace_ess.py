import argparse
import os, sys
import math
import time
import tabulate

import torch

import numpy as np

from subspace_inference import data, models, utils, losses
from subspace_inference.posteriors import SWAG
from subspace_inference.posteriors.swag_pca import SWAG as SWAGPCA
from subspace_inference.posteriors.proj_model import SubspaceModel, MNPCA_SubspaceModel, MNPCA_SubspaceModel_spec
from subspace_inference.posteriors.elliptical_slice import elliptical_slice

from subspace_inference.dataset.mini_imagenet import ImageNet, MetaImageNet
from subspace_inference.dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from subspace_inference.dataset.cifar import CIFAR100, MetaCIFAR100
from subspace_inference.dataset.transform_cfg import transforms_options, transforms_list
from torch.utils.data import DataLoader
import sklearn.decomposition

parser = argparse.ArgumentParser(description='Subspace ESS')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
#parser.add_argument('--log_fname', type=str, default=None, required=True, help='file name for logging')

parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')

parser.add_argument('--use_test', dest='use_test', action='store_true',
                    help='use test dataset instead of validation (default: False)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--num_samples', type=int, default=30, metavar='N', help='number of epochs (default: 30')

parser.add_argument('--curve', action='store_true')
parser.add_argument('--random', action='store_true')
parser.add_argument('--rank', type=int, default=2, metavar='N', help='approximation rank (default: 2')
parser.add_argument('--checkpoint', type=str, default=None, required=True, nargs='+', help='path to SWAG checkpoint')


parser.add_argument('--prior_std', type=float, default=1.0, help='std of the prior distribution')
parser.add_argument('--temperature', type=float, default=1., help='temperature')

parser.add_argument('--bn_subset', type=float, default=1.0, help='BN subset for evaluation (default 1.0)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--no_schedule', action='store_true', help='store schedule')

#from rfs
parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
parser.add_argument('--use_trainval', action='store_true', help='use trainval set')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)',choices=['miniImageNet', 'tieredImageNet',
                                                                                                              'CIFAR-FS', 'FC100', 'CIFAR10', 'CIFAR100'])
parser.add_argument('--data_root', type=str, default='', help='path to data root')

parser.add_argument('--adam', action='store_true', help='use adam optimizer')
parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
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

parser.add_argument('--subspace', choices=['covariance', 'pca', 'freq_dir', 'MNPCA_MANY'], default='MNPCA_MANY')

parser.add_argument('--sample_collect', type=int, default=20)

parser.add_argument('--special', type=bool, default=False)


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
with open(os.path.join(args.dir, 'ess_command.sh'), 'w') as f:
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

model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)

if args.curve:

    assert args.rank == 2

    checkpoint = torch.load(args.checkpoint)
    num_parameters = sum([p.numel() for p in model.parameters()])
    w = np.zeros((3, num_parameters))

    for i in range(3):
        offset = 0
        for name, param in model.named_parameters():

            size = param.numel()

            if 'net.%s_1' % name in checkpoint['model_state']:
                w[i, offset:offset+size] = checkpoint['model_state']['net.%s_%d' % (name,i)].cpu().numpy().ravel()
            else:
                tokens = name.split('.')
                name_fixed = '.'.join(tokens[:3] + tokens[4:])
                w[i, offset:offset+size] = checkpoint['model_state']['net.%s_%d' % (name_fixed,i)].cpu().numpy().ravel()
            offset += size


    w[1] = 0.25 * (w[0] + w[2]) + 0.5 * w[1]

    mean = np.mean(w, axis=0)
    u = w[2] - w[0]
    du = np.linalg.norm(u)

    v = w[1] - w[0]
    v -= u / du * np.sum(u / du * v)
    dv = np.linalg.norm(v)

    u /= math.sqrt(3.0)
    v /= 1.5

    cov_factor = np.vstack((u[None, :], v[None, :]))
    subspace = SubspaceModel(torch.FloatTensor(mean), torch.FloatTensor(cov_factor))
    coords = np.dot(cov_factor / np.sum(np.square(cov_factor), axis=1, keepdims=True), (w - mean[None, :]).T).T
    theta = torch.FloatTensor(coords[2, :])

    for i in range(3):
        v = subspace(torch.FloatTensor(coords[i]))
        offset = 0
        for param in model.parameters():
            param.data.copy_(v[offset:offset + param.numel()].view(param.size()).to(args.device))
            offset += param.numel()
        utils.bn_update(train_loader, model)
        print("Performance of model", i, "on the curve", end=":")
        utils.mata_eval(model, meta_valloader, meta_testloader, 'curve')

else:
    assert len(args.checkpoint) == 1
    if args.subspace == 'MNPCA_MANY':
        swag_model = SWAG(
            model_cfg.base,
            args.subspace,
            {
                'max_rank': 20,
                'cov_mat': args.sample_collect
                #'pca_rank': args.rank,
            },
            1e-6,
            *model_cfg.args,
            num_classes = num_classes,
            **model_cfg.kwargs)
        swag_model_pca = SWAGPCA(
        model_cfg.base,
        'pca',
        {
            'max_rank': 20,
            'pca_rank': args.rank,
        },
        1e-6,
        *model_cfg.args,
        num_classes=num_classes,
        **model_cfg.kwargs
        )
        swag_model.to(args.device)
        swag_model_pca.to(args.device)
    else:
        swag_model = SWAG(
            model_cfg.base,
            num_classes=num_classes,
            subspace_type=args.subspace,
            subspace_kwargs={
                'max_rank': 20,
                'pca_rank': args.rank,
            },
            *model_cfg.args,
            **model_cfg.kwargs)
        swag_model.to(args.device)

    print('Loading: %s' % args.checkpoint[0])
    ckpt = torch.load(args.checkpoint[0])
    swag_model.load_state_dict(ckpt['state_dict'], strict=False)
    swag_model_pca.load_state_dict(ckpt['state_dict'], strict=False)
    #print(swag_model.cov_factor)

    # first take as input SWA
    swag_model.set_swa()
    swag_model_pca.set_swa()
    utils.bn_update(train_loader, swag_model)
    utils.bn_update(train_loader, swag_model_pca)
    #print(utils.eval(meta_testloader, swag_model, losses.cross_entropy))
    utils.mata_eval(swag_model, meta_valloader, meta_testloader, 'SWAG:')

    mean, variance, cov_factor = swag_model.get_space()
    mean_pca, variance_pca, cov_factor_pca = swag_model_pca.get_space()
    
    #for sp in swag_model.subspace.subspaces.values():
        #print(sp.dimensions, sp.pca_ranks)

    if args.random:
        cov_factor = cov_factor.cpu().numpy()
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
#change
        cov_factor = torch.FloatTensor(cov_factor, device=mean.device)
    if args.subspace == 'pca':
        subspace = SubspaceModel(mean, cov_factor)
        #print(cov_factor.shape, mean.shape)
        theta = torch.zeros(args.rank)
    elif args.special:
        subspace = MNPCA_SubspaceModel_spec(mean, swag_model.subspace.subspaces, cov_factor)
        theta = torch.zeros(4)
        subspace_pca = SubspaceModel(mean_pca, cov_factor_pca)
        theta_pca = torch.zeros(args.rank)
    else:
        subspace = MNPCA_SubspaceModel(mean, swag_model.subspace.subspaces, cov_factor)
        lens = 0
        for sp in swag_model.subspace.subspaces.values():
            lens += np.prod(sp.pca_ranks)
        print(lens)
        theta = torch.zeros(lens)


def log_pdf(theta, subspace, model, loader, criterion, temperature, device, temp_pca = False):
    if args.subspace != 'MNPCA_MANY' or temp_pca:
        w = subspace(torch.FloatTensor(theta))
        offset = 0
        for param in model.parameters():
            param.data.copy_(w[offset:offset + param.numel()].view(param.size()).to(device))
            offset += param.numel()
    else:
        w = subspace(theta)
        for name, param in model.named_parameters():
            param.data.copy_(torch.from_numpy(w[name]).view(param.size()).to(device))
    model.train()
    with torch.no_grad():
        loss = 0
        for data, target, _ in loader:
            data = data.to(device)
            target = target.to(device)
            batch_loss, _, _ = criterion(model, data, target)
            loss += batch_loss * data.size()[0]
    return -loss.item() / temperature


def oracle(theta):
    return log_pdf(
        theta,
        subspace=subspace,
        model=model,
        loader=train_loader,
        criterion=losses.cross_entropy,
        temperature=args.temperature,
        device=args.device
    )

def oracle_pca(theta):
    return log_pdf(
        theta,
        subspace=subspace_pca,
        model=model,
        loader=train_loader,
        criterion=losses.cross_entropy,
        temperature=5000,#args.temperature,
        device=args.device,
        temp_pca = True
    )
'''
query_length = len(meta_testloader.dataset) * len(meta_testloader.dataset[0][-1])
ens_predictions = np.zeros((query_length, args.n_ways))
targets = np.zeros((query_length , args.n_ways))
'''
#change
query_length = len(meta_testloader.dataset) * len(meta_testloader.dataset[0][-1])
ens_predictions = np.zeros((query_length, args.n_ways))
ens_predictions_feats = np.zeros((query_length, args.n_ways))
targets = np.zeros((query_length , args.n_ways))
targets_feats = np.zeros((query_length , args.n_ways))

columns = ['iter', 'log_prob', 'acc', 'nll', 'time']
#change
samples = np.zeros((args.num_samples, 4))
samples_pca = np.zeros((args.num_samples, args.rank))

for i in range(args.num_samples):
#for i in range(0):
    time_sample = time.time()
    prior_sample = np.random.normal(loc=0.0, scale=args.prior_std, size=4)
    theta, log_prob = elliptical_slice(initial_theta=theta.numpy().copy(), prior=prior_sample,
                                                    lnpdf=oracle)
    samples[i, :] = theta
    theta = torch.FloatTensor(theta)
    w = subspace(theta)
    if args.subspace != 'MNPCA_MANY':
        offset = 0
        for param in model.parameters():
            param.data.copy_(w[offset:offset + param.numel()].view(param.size()).to(args.device))
            offset += param.numel()
    else:
        for name, param in model.named_parameters():
            param.data.copy_(torch.from_numpy(w[name]).view(param.size()).to(args.device))
    utils.bn_update(train_loader, model, subset=args.bn_subset)
    pred_res = utils.predict(meta_testloader, model)#use logit, testloader=test and no feat
    #new
    pred_res_feats = utils.predict(meta_testloader, model, use_logit=False)   
    utils.mata_eval(model, meta_valloader, meta_testloader, 'SWAG-ess')
    ens_predictions += pred_res['predictions']
    #new
    ens_predictions_feats += pred_res_feats['predictions']
    targets = pred_res['targets']
    #new
    targets_feats = pred_res_feats['targets']
    time_sample = time.time() - time_sample
    values = ['%3d/%3d' % (i + 1, args.num_samples),
              log_prob,
              np.mean(np.argmax(ens_predictions, axis=1) == targets),
              nll(ens_predictions / (i + 1), targets),
              time_sample]
    #new
    values_feats = ['%3d/%3d' % (i + 1, args.num_samples),
              log_prob,
              np.mean(np.argmax(ens_predictions_feats, axis=1) == targets_feats),
              nll(ens_predictions_feats / (i + 1), targets_feats),
              time_sample]
    #change
    table = tabulate.tabulate([values] + [values_feats], columns, tablefmt='simple', floatfmt='8.4f')
    #table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if i == 0:
        print(table)
    else:
        print(table)
        #print(table.split('\n')[2])
print('in pca mode')
for i in range(args.num_samples):
    time_sample = time.time()
    prior_sample = np.random.normal(loc=0.0, scale=args.prior_std, size=args.rank)
    theta_pca, log_prob = elliptical_slice(initial_theta=theta_pca.numpy().copy(), prior=prior_sample,
                                                    lnpdf=oracle_pca)
    samples_pca[i, :] = theta_pca
    theta_pca = torch.FloatTensor(theta_pca)
    #print(theta)
    w = subspace_pca(theta_pca)
    offset = 0
    for param in model.parameters():
        param.data.copy_(w[offset:offset + param.numel()].view(param.size()).to(args.device))
        offset += param.numel()
    utils.bn_update(train_loader, model, subset=args.bn_subset)
    pred_res = utils.predict(meta_testloader, model)
    #new
    pred_res_feats = utils.predict(meta_testloader, model, use_logit=False) 
    utils.mata_eval(model, meta_valloader, meta_testloader, 'SWAG-ess')
    ens_predictions += pred_res['predictions']
    #new
    ens_predictions_feats += pred_res_feats['predictions']
    targets = pred_res['targets']
    #new
    targets_feats = pred_res_feats['targets']
    time_sample = time.time() - time_sample
    values = ['%3d/%3d' % (i + 1, args.num_samples),
              log_prob,
              np.mean(np.argmax(ens_predictions, axis=1) == targets),
              nll(ens_predictions / (i + 1), targets),
              time_sample]
    #new
    values_feats = ['%3d/%3d' % (i + 1, args.num_samples),
              log_prob,
              np.mean(np.argmax(ens_predictions_feats, axis=1) == targets_feats),
              nll(ens_predictions_feats / (i + 1), targets_feats),
              time_sample]
    #change
    table = tabulate.tabulate([values] + [values_feats], columns, tablefmt='simple', floatfmt='8.4f')
    #table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if i == 0:
        print(table)
    else:
        print(table)
        #print(table.split('\n')[2])

ens_predictions /= args.num_samples
ens_acc = np.mean(np.argmax(ens_predictions, axis=1) == targets)
ens_nll = nll(ens_predictions, targets)
print("Ensemble NLL:", ens_nll)
print("Ensemble Accuracy:", ens_acc)

print("Ensemble Acc:", ens_acc)
print("Ensemble NLL:", ens_nll)
#new
ens_predictions_feats /= args.num_samples
ens_acc_feats = np.mean(np.argmax(ens_predictions_feats, axis=1) == targets_feats)
ens_nll_feats = nll(ens_predictions_feats, targets_feats)
print("Ensemble NLL Feats:", ens_nll_feats)
print("Ensemble Accuracy Feats:", ens_acc_feats)

print("Ensemble Acc Feats:", ens_acc_feats)
print("Ensemble NLL Feats:", ens_nll_feats)

if not os.path.exists(args.dir):
    os.mkdir(args.dir)
np.savez(
    os.path.join(args.dir, 'ens.npz'),
    seed=args.seed,
    samples=samples,
    ens_predictions=ens_predictions_feats,#change
    #ens_predictions=ens_predictions,
    targets=targets,
    ens_acc=ens_acc,
    ens_nll=ens_nll
)
