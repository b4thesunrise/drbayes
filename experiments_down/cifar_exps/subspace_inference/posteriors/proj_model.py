import torch
from subspace_inference import unflatten_like
import numpy as np
import copy

class SubspaceModel(torch.nn.Module):
    def __init__(self, mean, cov_factor):
        super(SubspaceModel, self).__init__()
        self.rank = cov_factor.size(0)
        self.register_buffer('mean', mean)
        self.register_buffer('cov_factor', cov_factor)

    def forward(self, t):
        return self.mean + self.cov_factor.t() @ t

class MNPCA_SubspaceModel(torch.nn.Module):
    def __init__(self, mean, subspaces, size):
        super(MNPCA_SubspaceModel, self).__init__()
        self.subspaces = subspaces
        self.mean = mean.cpu().numpy()
        self.sizes = size
        
    def forward(self, t):
        ret = {}
        offset1 = 0
        offset2 = 0
        for name in self.sizes:
            subspace = self.subspaces[name]
            subnum = np.prod(subspace.pca_ranks)
            #print(subspace.pca_ranks)
            tensorinput = t[offset1:offset1 + subnum].reshape(subspace.pca_ranks)
            offset1 = offset1 + subnum
            tensoroutput = subspace.reconstruct(tensorinput)
            meanplus = self.mean[offset2:offset2 + self.sizes[name][1]].reshape(*self.sizes[name][0])
            offset2 = offset2 + self.sizes[name][1]
            tensoroutput = tensoroutput + meanplus
            ret[name] = tensoroutput
        return copy.deepcopy(ret)

class MNPCA_SubspaceModel_spec(torch.nn.Module):
    def __init__(self, mean, subspaces, size):
        super(MNPCA_SubspaceModel_spec, self).__init__()
        self.subspaces = subspaces
        self.mean = mean.cpu().numpy()
        self.sizes = size
   #change     
    def forward(self, t):
        tensoroutput1 = self.subspaces['layer1'].reconstruct(t[0:1].reshape(self.subspaces['layer1'].pca_ranks))
        #print('tensoroutput1', tensoroutput1.shape)
        tensoroutput2 = self.subspaces['layer2'].reconstruct(t[1:2].reshape(self.subspaces['layer2'].pca_ranks))
        #print('tensoroutput2', tensoroutput2.shape)
        tensoroutput3 = self.subspaces['layer3'].reconstruct(t[2:3].reshape(self.subspaces['layer3'].pca_ranks))
        #print('tensoroutput3', tensoroutput3.shape)
        tensoroutput4 = self.subspaces['layer4'].reconstruct(t[3:4].reshape(self.subspaces['layer4'].pca_ranks))
        #print('tensoroutput4', tensoroutput4.shape)
        ret = {}
        offset2 = 0
        for name in self.sizes:
            meanplus = self.mean[offset2:offset2 + self.sizes[name][1]].reshape(*self.sizes[name][0])
            offset2 = offset2 + self.sizes[name][1]
            if 'conv' in name and 'conv1' not in name:
                if 'layer1' in name:
                    if 'conv2' in name:
                        add1 = tensoroutput1[0].reshape(64,64,3,3)
                    elif 'conv3' in name:
                        add1 = tensoroutput1[1].reshape(64,64,3,3)
                elif 'layer2' in name:
                    if 'conv2' in name:
                        add1 = tensoroutput2[0].reshape(160,160,3,3)
                    elif 'conv3' in name:
                        add1 = tensoroutput2[1].reshape(160,160,3,3)
                elif 'layer3' in name:
                    if 'conv2' in name:
                        add1 = tensoroutput3[0].reshape(320,320,3,3)
                    elif 'conv3' in name:
                        add1 = tensoroutput3[1].reshape(320,320,3,3)
                elif 'layer4' in name:
                    if 'conv2' in name:
                        add1 = tensoroutput4[0].reshape(640,640,3,3)
                    elif 'conv3' in name:
                        add1 = tensoroutput4[1].reshape(640,640,3,3)
                    #add1 = tensoroutput[offset1:offset1 + int(self.sizes[name][1] / 9)].reshape(*self.sizes[name][0])
                    #offset1 += int(self.sizes[name][1] / 9)
                ret[name] = add1 + meanplus
            else:
                ret[name] = meanplus
        return copy.deepcopy(ret)

class ProjectedModel(torch.nn.Module):
    def __init__(self, proj_params, model, projection=None, mean=None, subspace=None):
        super(ProjectedModel, self).__init__()
        self.model = model

        if subspace is None:
            self.subspace = SubspaceModel(mean, projection)
        else:
            self.subspace = subspace

        if mean is None and subspace is None:
            raise NotImplementedError('Must enter either subspace or mean')

        self.proj_params = proj_params

    def update_params(self, vec, model):
        vec_list = unflatten_like(likeTensorList=list(model.parameters()), vector=vec.view(1,-1))
        for param, v in zip(model.parameters(), vec_list):
            param.detach_()
            param.mul_(0.0).add_(v)

    def forward(self, *args, **kwargs):
        y = self.subspace(self.proj_params)

        self.update_params(y, self.model)
        return self.model(*args, **kwargs)
