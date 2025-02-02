import os
import random
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
from torch.utils import data
import torch
from scipy.io import loadmat
import yaml
from sklearn.decomposition import PCA
from sklearn import preprocessing


def applyPCA(X, numComponents):
    """
    apply PCA to the image to reduce dimensionality
  """
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def min_max(x):
    min = np.min(x)
    max = np.max(x)
    return (x - min) / (max - min)

class HXDataset(data.Dataset):
    def __init__(self,hsi, Xdata, hsi_pca,index,gt, windowSize,category=None):
        modes = ['symmetric', 'reflect']
        self.pad = windowSize // 2
        self.windowSize = windowSize 
        
        self.hsi = np.pad(hsi, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize%2])
        self.hsi_pca = np.pad(hsi_pca, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        if(len(Xdata.shape)==2):
            self.Xdata = np.pad(Xdata, ((self.pad, self.pad),
                                    (self.pad, self.pad)), mode=modes[windowSize%2])
        else:
            self.Xdata = np.pad(Xdata, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])   #(0,0)表示在通道维
        
        self.category=category

        self.pos = index
        self.gt = gt

    def __getitem__(self, index):
        h, w = self.pos[index, :]
        hsi = self.hsi[h: h + self.windowSize, w: w + self.windowSize].astype(np.float64) 
        hsi_pca=self.hsi_pca[h: h + self.windowSize, w: w + self.windowSize].astype(np.float64) 
        Xdata = self.Xdata[h: h + self.windowSize, w: w + self.windowSize].astype(np.float64) 
        
        hsi = transforms.ToTensor()(hsi).float()
        Xdata = transforms.ToTensor()(Xdata).float()
        hsi_pca = transforms.ToTensor()(hsi_pca).float()
        
        if(self.category=='train'):
            if random.random()<0.5:
                t = random.randint(1, 2) 
                hsi=torch.flip(hsi,dims=[-t])
                Xdata=torch.flip(Xdata,dims=[-t])
                hsi_pca=torch.flip(hsi_pca,dims=[-t])    
        
        gt = torch.tensor(self.gt[h, w] - 1).long()
        return hsi, Xdata, hsi_pca, gt,h,w
    def __len__(self):
        return self.pos.shape[0]


def get_loader(dataset, batchsize, num_workers=0, useval=0,pin_memory=True):
    
    with open('dataset_info.yaml', 'r') as file:
        data_info = yaml.safe_load(file)
    data_info=data_info[dataset]
    windowSize=data_info['window_size']
    
    hsi_path="data/"+dataset+'/'+ data_info['info'][1]
    X_path="data/"+dataset+'/'+ data_info['info'][3]
    gt_path="data/"+dataset+'/'+ data_info['info'][0]
    index_path="data/"+dataset+'/'+ data_info['info'][2]
        
    hsi = loadmat(hsi_path)[data_info['keys'][0]]
    Xdata = loadmat(X_path)[data_info['keys'][1]]
    gt = loadmat(gt_path)[data_info['keys'][2]]
    train_index = loadmat(index_path)[data_info['keys'][3]]
    test_index = loadmat(index_path)[data_info['keys'][4]]
    trntst_index = np.concatenate((train_index, test_index), axis=0)
    all_index=loadmat(index_path)[data_info['keys'][5]]

    random_indices = np.random.choice(test_index.shape[0], size=test_index.shape[0] // 2, replace=False)
    val_index = test_index[random_indices]
    hsi_pca = applyPCA(hsi, data_info['pca_num'])
    
    hsi_pca_all = preprocessing.scale(hsi_pca.reshape(np.prod(hsi_pca.shape[:2]),hsi_pca.shape[2]))
    hsi_pca = hsi_pca_all.reshape(hsi_pca.shape[0],hsi_pca.shape[1],hsi_pca.shape[2])
    if(len(Xdata.shape)==2):
        Xdata_all=preprocessing.scale(Xdata.reshape(np.prod(Xdata.shape[:2])))
        Xdata=Xdata_all.reshape(Xdata.shape[0],Xdata.shape[1])
    else:
        Xdata_all=preprocessing.scale(Xdata.reshape(np.prod(Xdata.shape[:2]),Xdata.shape[2]))
        Xdata=Xdata_all.reshape(Xdata.shape[0],Xdata.shape[1],Xdata.shape[2])
        
    HXtrainset = HXDataset(hsi=hsi, Xdata=Xdata, hsi_pca=hsi_pca,
                    index=train_index,gt=gt, windowSize=windowSize,category='train')
    HXallset = HXDataset(hsi, Xdata,hsi_pca, all_index,gt, windowSize)
    if(useval==0):
        HXtestset = HXDataset(hsi=hsi, Xdata=Xdata, hsi_pca=hsi_pca,
                    index=test_index,gt=gt, windowSize=windowSize,category='test')
    else:
        HXtestset = HXDataset(hsi=hsi, Xdata=Xdata, hsi_pca=hsi_pca,
                    index=val_index,gt=gt, windowSize=windowSize,category='val')
    HXtrntstset = HXDataset(hsi=hsi, Xdata=Xdata, hsi_pca=hsi_pca,
            index=trntst_index,gt=gt, windowSize=windowSize,category='trntst')

    train_loader = data.DataLoader(dataset=HXtrainset,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=True)
    test_loader = data.DataLoader(dataset=HXtestset,
                                batch_size=batchsize,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                drop_last=True)
    trntst_loader = data.DataLoader(dataset=HXtrntstset,
                            batch_size=batchsize,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            drop_last=True)
    all_loader = data.DataLoader(
                HXallset, batch_size=batchsize, shuffle=False,
                pin_memory=pin_memory, num_workers=num_workers)
    
    print('num_workers='+ str(num_workers))

    return train_loader,test_loader,trntst_loader, all_loader,train_index.shape[0],test_index.shape[0],trntst_index.shape[0]