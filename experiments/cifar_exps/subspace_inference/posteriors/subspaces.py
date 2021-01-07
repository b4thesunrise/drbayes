"""
    subspace classes
    CovarianceSpace: covariance subspace
    PCASpace: PCA subspace 
    FreqDirSpace: Frequent Directions Space
"""

import abc

import torch
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition.pca import _assess_dimension_
from sklearn.utils.extmath import randomized_svd
import copy

device = torch.device('cuda')

##unfold
def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')
##fold
def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order = 'F'), 0, mode)
def getformat(i):
    if i == 1:
        return 'ij'
    elif i == 2:
        return 'ijk'
    elif i == 3:
        return 'ijkl'
    elif i == 4:
        return 'ijklm'
    elif i == 5:
        return 'ijklmx'
    elif i == 6:
        return 'ijklmxy'
    elif i == 7:
        return 'ijklmxyz'
def axisiisj(Xs, i, j):
    if i == 0:
        return Xs[j]
    elif i == 1:
        return Xs[:,j]
    elif i == 2:
        return Xs[:,:,j]
    elif i == 3:
        return Xs[:,:,:,j]
    elif i == 4:
        return Xs[:,:,:,:,j]
    elif i == 5:
        return Xs[:,:,:,:,:,j]
    elif i == 6:
        return Xs[:,:,:,:,:,:,j]
    else:
        return Xs[:,:,:,:,:,:,:,j]

def changereductions(reductions, Xs):
    Xs = np.array(Xs)
    print(Xs.shape)
    #（3，3，3）
    #mode 1
    #(3,0,3),(3,1,3),(3,2,3),(3,3,3)
    for dim,num in enumerate(Xs.shape):
        if dim == 0:
            continue
        print(dim)
        for i in range(num):
            scaler = np.linalg.norm(axisiisj(Xs, dim, i))
            print(i, scaler)
            reductions[dim-1][:,i] = reductions[dim-1][:,i] * scaler
    return reductions


class Subspace(torch.nn.Module, metaclass=abc.ABCMeta):
    subclasses = {}

    @classmethod
    def register_subclass(cls, subspace_type):
        def decorator(subclass):
            cls.subclasses[subspace_type] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, subspace_type, **kwargs):
        if subspace_type not in cls.subclasses:
            raise ValueError('Bad subspaces type {}'.format(subspace_type))
        return cls.subclasses[subspace_type](**kwargs)

    def __init__(self):
        super(Subspace, self).__init__()

    @abc.abstractmethod
    def collect_vector(self, vector):
        pass

    @abc.abstractmethod
    def get_space(self):
        pass

def tensorpro(X):
    print(X.shape)
    tensor_X = torch.from_numpy(X).to(device)
    return torch.mm(tensor_X,tensor_X.t()).cpu().numpy()
    
@Subspace.register_subclass('MNPCA')
class MNPCASpace(Subspace):

    def __init__(self, dimensions, formatstr, pca_ranks, max_rank):
        super(MNPCASpace, self).__init__()
        self.dimensions = dimensions
        self.dimension_len = dimensions.shape[0]
        self.formatstr = formatstr
        self.pca_ranks = pca_ranks
        self.max_rank = max_rank
        self.register_buffer('rank', torch.zeros(1, dtype=torch.long))
        self.register_buffer('cov_mat_sqrt',
                             torch.empty(0, *self.dimensions, dtype=torch.float32))

    def collect_vector(self, tensor):
        if self.rank.item() + 1 > self.max_rank:
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:]
        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, tensor), dim=0)
        #self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, vector.view(1, -1)), dim=0)
        self.rank = torch.min(self.rank + 1, torch.as_tensor(self.max_rank)).view(-1)
    
    def getformatstr(self, formatstr, pos):
        format2 = formatstr[pos] + 'n'
        format1 = formatstr.replace(formatstr[pos], 'n')
        einstr = formatstr + ',' + format2 + '->' + format1
        return einstr

    def get_space(self):      
        formatstr = self.formatstr[1:]
        dimensions = copy.deepcopy(self.dimensions)
        print('in get_space',dimensions)
        reductions = []
        Xs = np.array(copy.deepcopy(self.cov_mat_sqrt.cpu().numpy()))
        for d in range(0, self.dimension_len):
            Xs = [ten2mat(x, d) for x in Xs]
            Matrix = np.sum(np.array([X @ X.T for X in Xs]), axis = 0)/self.rank.cpu().numpy()
            #print(Matrix)
            Q, sigma, Qt = np.linalg.svd(Matrix)
            reductions.append(copy.deepcopy(Q[:,0:self.pca_ranks[d]]))
            #print(dimensions)
            Xs = [mat2ten(x, dimensions, d) for x in Xs]
            einstr = self.getformatstr(formatstr, d)
            #print(einstr, Xs[0].shape, Q[:,0:self.hypers[d]].shape)
            Xs = [np.einsum(einstr, x, Q[:,0:self.pca_ranks[d]]) for x in Xs]
            #print(Xs[0].shape)
            #print('------------------------')
            dimensions[d] = self.pca_ranks[d]
        self.reductions = reductions
        #print(reductions)
        #20 样本--》降低维得到20个低维样本，以及映射矩阵--》在低维空间上采样（因为先验是各向同性），再映射回去求似然函数，用MCMC
        self.reductions = changereductions(self.reductions, Xs)
        return copy.deepcopy(reductions)
    
    def reconstruct(self, ts):
        #print(ts.shape)
        for i,reduce in enumerate(self.reductions):
            format2 = self.formatstr[i+1] + 'n'
            format1 = self.formatstr.replace(self.formatstr[i+1], 'n')
            einstr = self.formatstr[1:] + ',' + format2 + '->' + format1[1:]
            #print(einstr)
            ts = np.einsum(einstr, ts, self.reductions[i].T)
        return ts
'''
@Subspace.register_subclass('MNPCA_MANY')
class MNPCA_MANYSpace(Subspace):

    def __init__(self, model, max_rank, num_parameters, pca_rankss = None, cov_mat = None):
        super(MNPCA_MANYSpace, self).__init__()
        self.num_parameters = num_parameters
        self.subspaces = {}
        self.sizes = {}
        for i, (name, param) in enumerate(model.named_parameters()):
            dimensions = np.array(param.size())
            print(name, dimensions)
            while dimensions[-1] == 1:
                dimensions = dimensions[0:-1]
            if 'classifier.weight' in name:
                pcaranks = [3 for x in dimensions]
            elif 'conv' in name:
                pcaranks = [2 for x in dimensions]
            else:
                pcaranks = [1 for x in dimensions]
            formatstr = getformat(len(dimensions))
            self.subspaces[name] = Subspace.create('MNPCA',dimensions = dimensions, formatstr = formatstr, pca_ranks = pcaranks, max_rank = max_rank)
            self.sizes[name] = (param.size()[0:dimensions.shape[0]], param.numel())
        self.register_buffer('rank', torch.zeros(1, dtype=torch.long))
        self.max_rank = max_rank
        if cov_mat:
            self.register_buffer('cov_mat_sqrt',
                             torch.empty(cov_mat, self.num_parameters, dtype=torch.float32))
        else:
            self.register_buffer('cov_mat_sqrt',
                             torch.empty(0, self.num_parameters, dtype=torch.float32))
            
    def collect_vector(self, vector): 
        if self.rank.item() + 1 > self.max_rank:
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :]
        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, vector.view(1, -1)), dim=0)
        self.rank = torch.min(self.rank + 1, torch.as_tensor(self.max_rank)).view(-1)
    
    def get_space(self):
        for vector in self.cov_mat_sqrt:
            offset = 0
            for name, (dim, size) in self.sizes.items():
                self.subspaces[name].collect_vector(vector[offset:offset + size].view(1,*dim))
                offset += size
        for name in self.sizes.keys():
            print(name)
            self.subspaces[name].get_space()
            #print(reductions_dict)
        return copy.deepcopy(self.sizes)
'''
#函数1，把网络参数拼起来的函数
#函数2，把网络参数加回去的函数
def addparam(model):
    ret1 = torch.zeros(0,64,64,3,3)
    ret2 = torch.zeros(0,160,160,3,3)
    ret3 = torch.zeros(0,320,320,3,3)
    ret4 = torch.zeros(0,640,640,3,3)
    offset = 0
    namedict = {}
    for i, (name, param) in enumerate(model.named_parameters()):
        namedict[name] = [param.size(), param.numel()]
        if 'conv' in name and 'conv1' not in name:
            param = param.reshape(1,*param.size())
            if 'layer1' in name:
                ret1 = torch.cat([ret1, param], dim = 0)
            elif 'layer2' in name:
                ret2 = torch.cat([ret2, param], dim = 0)
            elif 'layer3' in name:
                ret3 = torch.cat([ret3, param], dim = 0)
            elif 'layer4' in name:
                ret4 = torch.cat([ret4, param], dim = 0)
        else:
            continue
    #dimensions = np.array(ret.size())
    #num = ret.numel()
    return ret1, ret2, ret3, ret4, namedict
    
@Subspace.register_subclass('MNPCA_MANY')
class MNPCA_MANYSpace(Subspace):
#change
    def __init__(self, model, max_rank, num_parameters, pca_rankss = None, cov_mat = None):
        super(MNPCA_MANYSpace, self).__init__()
        self.num_parameters = num_parameters
        self.subspaces = {}
        self.sizes = {}
        pcaranks1 = [1,1,1,1,1]
        pcaranks2 = [1,1,1,1,1]
        pcaranks3 = [1,1,1,1,1]
        pcaranks4 = [1,1,1,1,1]
        ret1, ret2, ret3, ret4, namedict = addparam(model)
        self.sizes = namedict
        formatstr1 = getformat(len(np.array(ret1.size())))
        formatstr2 = getformat(len(np.array(ret2.size())))
        formatstr3 = getformat(len(np.array(ret3.size())))
        formatstr4 = getformat(len(np.array(ret4.size())))
        self.subspaces['layer1'] = Subspace.create('MNPCA',dimensions = np.array(ret1.size()), formatstr =  formatstr1, pca_ranks = pcaranks1, max_rank = max_rank)
        self.subspaces['layer2'] = Subspace.create('MNPCA',dimensions = np.array(ret2.size()), formatstr = formatstr2, pca_ranks = pcaranks2, max_rank = max_rank)
        self.subspaces['layer3'] = Subspace.create('MNPCA',dimensions = np.array(ret3.size()), formatstr = formatstr3, pca_ranks = pcaranks3, max_rank = max_rank)
        self.subspaces['layer4'] = Subspace.create('MNPCA',dimensions = np.array(ret4.size()), formatstr = formatstr4, pca_ranks = pcaranks4, max_rank = max_rank)
        #self.sizes['conv'] = (param.size()[0:dimensions.shape[0]], param.numel())
        self.register_buffer('rank', torch.zeros(1, dtype=torch.long))
        self.max_rank = max_rank
        if cov_mat:
            self.register_buffer('cov_mat_sqrt',
                             torch.empty(cov_mat, self.num_parameters, dtype=torch.float32))
        else:
            self.register_buffer('cov_mat_sqrt',
                             torch.empty(0, self.num_parameters, dtype=torch.float32))
        
    def collect_vector(self, vector): 
        if self.rank.item() + 1 > self.max_rank:
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :]
        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, vector.view(1, -1)), dim=0)
        self.rank = torch.min(self.rank + 1, torch.as_tensor(self.max_rank)).view(-1)
    
    def get_space(self):
        for vector in self.cov_mat_sqrt:
            collect1 = torch.zeros(0,64,64,3,3)
            collect2 = torch.zeros(0,160,160,3,3)
            collect3 = torch.zeros(0,320,320,3,3)
            collect4 = torch.zeros(0,640,640,3,3)
            offset = 0
            for name, (dim, size) in self.sizes.items():
                if 'conv' in name and 'conv1' not in name:
                    param = vector[offset:offset + size]
                    if 'layer1' in name:
                        collect1 = torch.cat([collect1, param.view(1,64,64,3,3)], dim = 0)
                    elif 'layer2' in name:
                        collect2 = torch.cat([collect2, param.view(1,160,160,3,3)], dim = 0)
                    elif 'layer3' in name:
                        collect3 = torch.cat([collect3, param.view(1,320,320,3,3)], dim = 0)
                    elif 'layer4' in name:
                        collect4 = torch.cat([collect4, param.view(1,640,640,3,3)], dim = 0)
                offset += size
            self.subspaces['layer1'].collect_vector( collect1.reshape(1,2,64,64,3,3) )
            self.subspaces['layer2'].collect_vector( collect2.reshape(1,2,160,160,3,3) )
            self.subspaces['layer3'].collect_vector( collect3.reshape(1,2,320,320,3,3) )
            self.subspaces['layer4'].collect_vector( collect4.reshape(1,2,640,640,3,3) )
        self.subspaces['layer1'].get_space()
        self.subspaces['layer2'].get_space()
        self.subspaces['layer3'].get_space()
        self.subspaces['layer4'].get_space()
        return copy.deepcopy(self.sizes)

@Subspace.register_subclass('random')
class RandomSpace(Subspace):
    def __init__(self, num_parameters, rank=20, method='dense'):
        assert method in ['dense', 'fastfood']

        super(RandomSpace, self).__init__()

        self.num_parameters = num_parameters
        self.rank = rank
        self.method = method

        if method == 'dense':
            self.subspace = torch.randn(rank, num_parameters)

        if method == 'fastfood':
            raise NotImplementedError("FastFood transform hasn't been implemented yet")

    # random subspace is independent of data
    def collect_vector(self, vector):
        pass
    
    def get_space(self):
        return self.subspace


@Subspace.register_subclass('covariance')
class CovarianceSpace(Subspace):

    def __init__(self, num_parameters, max_rank=20):
        super(CovarianceSpace, self).__init__()

        self.num_parameters = num_parameters

        self.register_buffer('rank', torch.zeros(1, dtype=torch.long))
        self.register_buffer('cov_mat_sqrt',
                             torch.empty(0, self.num_parameters, dtype=torch.float32))

        self.max_rank = max_rank

    def collect_vector(self, vector):
        if self.rank.item() + 1 > self.max_rank:
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :]
        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, vector.view(1, -1)), dim=0)
        self.rank = torch.min(self.rank + 1, torch.as_tensor(self.max_rank)).view(-1)

    def get_space(self):
        return self.cov_mat_sqrt.clone() / (self.cov_mat_sqrt.size(0) - 1) ** 0.5

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        rank = state_dict[prefix + 'rank'].item()
        self.cov_mat_sqrt = self.cov_mat_sqrt.new_empty((rank, self.cov_mat_sqrt.size()[1]))
        super(CovarianceSpace, self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                                           strict, missing_keys, unexpected_keys,
                                                           error_msgs)


@Subspace.register_subclass('pca')
class PCASpace(CovarianceSpace):

    def __init__(self, num_parameters, pca_rank=20, max_rank=20):
        super(PCASpace, self).__init__(num_parameters, max_rank=max_rank)

        # better phrasing for this condition?
        assert(pca_rank == 'mle' or isinstance(pca_rank, int))
        if pca_rank != 'mle':
            assert 1 <= pca_rank <= max_rank

        self.pca_rank = pca_rank

    def get_space(self):

        cov_mat_sqrt_np = self.cov_mat_sqrt.clone().numpy()

        # perform PCA on DD'
        cov_mat_sqrt_np /= (max(1, self.rank.item() - 1))**0.5

        if self.pca_rank == 'mle':
            pca_rank = self.rank.item()
        else:
            pca_rank = self.pca_rank

        pca_rank = max(1, min(pca_rank, self.rank.item()))
        pca_decomp = TruncatedSVD(n_components=pca_rank)
        pca_decomp.fit(cov_mat_sqrt_np)

        _, s, Vt = randomized_svd(cov_mat_sqrt_np, n_components=pca_rank, n_iter=5)

        # perform post-selection fitting
        if self.pca_rank == 'mle':
            eigs = s ** 2.0
            ll = np.zeros(len(eigs))
            correction = np.zeros(len(eigs))

            # compute minka's PCA marginal log likelihood and the correction term
            for rank in range(len(eigs)):
                # secondary correction term based on the rank of the matrix + degrees of freedom
                m = cov_mat_sqrt_np.shape[1] * rank - rank * (rank + 1) / 2.
                correction[rank] = 0.5 * m * np.log(cov_mat_sqrt_np.shape[0])
                ll[rank] = _assess_dimension_(spectrum=eigs,
                                              rank=rank,
                                              n_features=min(cov_mat_sqrt_np.shape),
                                              n_samples=max(cov_mat_sqrt_np.shape))
            
            self.ll = ll
            self.corrected_ll = ll - correction
            self.pca_rank = np.nanargmax(self.corrected_ll)
            print('PCA Rank is: ', self.pca_rank)
            return torch.FloatTensor(s[:self.pca_rank, None] * Vt[:self.pca_rank, :])
        else:
            return torch.FloatTensor(s[:, None] * Vt)


@Subspace.register_subclass('freq_dir')
class FreqDirSpace(CovarianceSpace):
    def __init__(self, num_parameters, max_rank=20):
        super(FreqDirSpace, self).__init__(num_parameters, max_rank=max_rank)
        self.register_buffer('num_models', torch.zeros(1, dtype=torch.long))
        self.delta = 0.0
        self.normalized = False

    def collect_vector(self, vector):
        if self.rank >= 2 * self.max_rank:
            sketch = self.cov_mat_sqrt.numpy()
            [_, s, Vt] = np.linalg.svd(sketch, full_matrices=False)
            if s.size >= self.max_rank:
                current_delta = s[self.max_rank - 1] ** 2
                self.delta += current_delta
                s = np.sqrt(s[:self.max_rank - 1] ** 2 - current_delta)
            self.cov_mat_sqrt = torch.from_numpy(s[:, None] * Vt[:s.size, :])

        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, vector.view(1, -1)), dim=0)
        self.rank = torch.as_tensor(self.cov_mat_sqrt.size(0))
        self.num_models.add_(1)
        self.normalized = False

    def get_space(self):
        if not self.normalized:
            sketch = self.cov_mat_sqrt.numpy()
            [_, s, Vt] = np.linalg.svd(sketch, full_matrices=False)
            self.cov_mat_sqrt = torch.from_numpy(s[:, None] * Vt)
            self.normalized = True
        curr_rank = min(self.rank.item(), self.max_rank)
        return self.cov_mat_sqrt[:curr_rank].clone() / max(1, self.num_models.item() - 1) ** 0.5
