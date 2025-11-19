import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch.nn.functional as F
from . layers import GraphAttentionLayer,SpGraphAttentionLayer
from . utils import mclust, sparse_mx_to_torch_sparse_tensor

# PROST_NN类：用于密集图注意力聚类与嵌入
class PROST_NN(nn.Module):
    def __init__(self, nfeat, embedding_size, cuda=False):
        '''
        初始化PROST_NN模型，设置输入特征维度、嵌入维度及是否使用CUDA
        '''
        super(PROST_NN, self).__init__()
        self.embedding_size = embedding_size
        self.cuda = cuda
        if self.cuda:
            if torch.cuda.is_available(): 
                print("Using cuda acceleration")
            else:
                raise ValueError("Cuda is unavailable, please set 'cuda=False'")

        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.beta=0.5

        # 图注意力层
        self.gal = GraphAttentionLayer(nfeat, embedding_size, 0.05, 0.15).to(self.device)
    
    # 计算软聚类分布q
    def get_q(self, z):
        '''
        根据嵌入z和聚类中心mu计算软聚类分布q
        '''
        q = 1.0 / ((1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.beta) + 1e-8)
        q = q**(self.beta+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q
    
    # 计算目标分布p
    def target_distribution(self, q):
        '''
        根据q计算目标分布p，用于KL散度优化
        '''
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    # 计算KL散度损失
    def KL_div(self, p, q):
        '''
        计算KL散度损失，用于聚类优化
        '''
        loss = torch.mean(torch.sum(p*torch.log(p/(q+1e-6)), dim=1))
        return loss
  
    # 前向传播，返回嵌入和软聚类分布
    def forward(self, x, adj):
        '''
        前向传播，输入特征和邻接矩阵，输出嵌入z和分布q
        '''
        z = self.gal(x, adj)
        q = self.get_q(z)
        return z, q

    # 训练模型，聚类并优化嵌入
    def train_(self, X, adj, init="mclust",n_clusters=7,res=0.1,tol=1e-3,lr=0.1, 
                max_epochs=500, seed = 818, update_interval=3):
        '''
        训练PROST_NN模型，初始化聚类中心并进行迭代优化
        '''
        optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=5e-4)
        
        X = torch.FloatTensor(X).to(self.device)
        adj = torch.FloatTensor(adj).to(self.device) 

        # 预计算特征嵌入
        with torch.no_grad():
            features = self.gal(X, adj)
     
        # 初始化聚类中心
        if init=="kmeans":
            print("\nInitializing cluster centers with kmeans, n_clusters known")
            self.n_clusters=n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(features.detach().cpu().numpy())
            
        elif init=="mclust":
            print("\nInitializing cluster centers with mclust, n_clusters known")
            data = features.detach().cpu().numpy()
            self.n_clusters = n_clusters
            self.seed = seed
            y_pred = mclust(data, num_cluster = self.n_clusters, random_seed = self.seed)
            y_pred = y_pred.astype(int)

        elif init=="louvain":
            print("\nInitializing cluster centers with louvain, resolution = ", res)
            adata = sc.AnnData(features.detach().cpu().numpy())
            sc.pp.neighbors(adata, n_neighbors=10)
            sc.tl.louvain(adata, resolution=res)
            y_pred = adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
            
        elif init=="leiden":
            print("\nInitializing cluster centers with leiden, resolution = ", res)
            adata=sc.AnnData(features.detach().cpu().numpy())
            sc.pp.neighbors(adata, n_neighbors=10)
            sc.tl.leiden(adata, resolution=res)
            y_pred = adata.obs['leiden'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
        #----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.embedding_size))

        features = pd.DataFrame(features.detach().cpu().numpy()).reset_index(drop = True)
        Group = pd.Series(y_pred, index=np.arange(0,features.shape[0]), name="Group")
        Mergefeature = pd.concat([features,Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())       
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.mu.data = self.mu.data.to(self.device)
        
        #---------------------------------------------------------------- 
        with tqdm(total=max_epochs) as t:
            for epoch in range(max_epochs):            
                t.set_description('Epoch')
                self.train()
                
                if epoch%update_interval == 0:
                    _, Q = self.forward(X, adj)
                    q = Q.detach().data.cpu().numpy().argmax(1)              
                    t.update(update_interval)
                    
                z,q = self(X, adj)
                p = self.target_distribution(Q.detach())
                
                loss = self.KL_div(p, q)

                optimizer.zero_grad()              
                loss.backward()
                optimizer.step()
    
                t.set_postfix(loss = loss.data.cpu().numpy())
                
                # 检查停止条件
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
                y_pred_last = y_pred
                if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    print("Total epoch:", epoch)
                    break

    # 预测嵌入和聚类分布
    def predict(self, X, adj):
        '''
        输入特征和邻接矩阵，输出嵌入和聚类分布
        '''
        X = torch.FloatTensor(X).to(self.device)
        adj = torch.FloatTensor(adj).to(self.device)
        z, q = self(X, adj)

        return z, q
    
    
# PROST_NN_sparse类：用于稀疏图注意力聚类与嵌入
class PROST_NN_sparse(nn.Module):
    def __init__(self, nfeat, embedding_size, cuda=False):
        '''
        初始化PROST_NN_sparse模型，设置输入特征维度、嵌入维度及是否使用CUDA
        '''
        super(PROST_NN_sparse, self).__init__()
        self.embedding_size = embedding_size
        self.cuda = cuda
        if self.cuda:
            if torch.cuda.is_available(): 
                print("Using cuda acceleration")
            else:
                raise ValueError("Cuda is unavailable, please set 'cuda=False'")

        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.beta=0.5

        # 稀疏图注意力层
        self.gal = SpGraphAttentionLayer(nfeat, embedding_size, 0.05, 0.15).to(self.device)

    # 计算软聚类分布q
    def get_q(self, z):
        '''
        根据嵌入z和聚类中心mu计算软聚类分布q
        '''
        q = 1.0 / ((1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.beta) + 1e-8)
        q = q**(self.beta+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q
    
    # 计算目标分布p
    def target_distribution(self, q):
        '''
        根据q计算目标分布p，用于KL散度优化
        '''
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    # 计算KL散度损失
    def KL_div(self, p, q):
        '''
        计算KL散度损失，用于聚类优化
        '''
        loss = torch.mean(torch.sum(p*torch.log(p/(q+1e-6)), dim=1))
        return loss
  
    # 前向传播，返回嵌入和软聚类分布
    def forward(self, x, adj):
        '''
        前向传播，输入特征和邻接矩阵，输出嵌入z和分布q
        '''
        z = self.gal(x, adj)
        q = self.get_q(z)
        return z, q

    # 训练模型，聚类并优化嵌入
    def train_(self, X, adj, init="mclust",n_clusters=7,res=0.1,tol=1e-3,lr=0.1, 
                max_epochs=500, seed = 818, update_interval=3):
        '''
        训练PROST_NN_sparse模型，初始化聚类中心并进行迭代优化
        '''
        optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=5e-4)
        
        X = torch.FloatTensor(X).to(self.device)
        adj = sparse_mx_to_torch_sparse_tensor(adj).to(self.device) 

        # 预计算特征嵌入
        with torch.no_grad():
            features = self.gal(X, adj)
     
        # 初始化聚类中心
        if init=="kmeans":
            print("\nInitializing cluster centers with kmeans, n_clusters known")
            self.n_clusters=n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(features.detach().cpu().numpy())
            
        elif init=="mclust":
            print("\nInitializing cluster centers with mclust, n_clusters known")
            data = features.detach().cpu().numpy()
            self.n_clusters = n_clusters
            self.seed = seed
            y_pred = mclust(data, num_cluster = self.n_clusters, random_seed = self.seed)
            y_pred = y_pred.astype(int)

        elif init=="louvain":
            print("\nInitializing cluster centers with louvain, resolution = ", res)
            adata = sc.AnnData(features.detach().cpu().numpy())
            sc.pp.neighbors(adata, n_neighbors=10)
            sc.tl.louvain(adata, resolution=res)
            y_pred = adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
            
        elif init=="leiden":
            print("\nInitializing cluster centers with leiden, resolution = ", res)
            adata=sc.AnnData(features.detach().cpu().numpy())
            sc.pp.neighbors(adata, n_neighbors=10)
            sc.tl.leiden(adata, resolution=res)
            y_pred = adata.obs['leiden'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
        #----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.embedding_size))

        features = pd.DataFrame(features.detach().cpu().numpy()).reset_index(drop = True)
        Group = pd.Series(y_pred, index=np.arange(0,features.shape[0]), name="Group")
        Mergefeature = pd.concat([features,Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())       
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.mu.data = self.mu.data.to(self.device)
        
        #----------------------------------------------------------------
        
        with tqdm(total=max_epochs) as t:
            for epoch in range(max_epochs):            
                t.set_description('Epoch')
                self.train()
                
                if epoch%update_interval == 0:
                    _, Q = self.forward(X, adj)
                    q = Q.detach().data.cpu().numpy().argmax(1)              
                    t.update(update_interval)
                    
                z,q = self(X, adj)
                p = self.target_distribution(Q.detach())
                
                loss = self.KL_div(p, q)
   
                optimizer.zero_grad()              
                loss.backward()
                optimizer.step()
    
                t.set_postfix(loss = loss.data.cpu().numpy())
                
                # 检查停止条件
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
                y_pred_last = y_pred
                if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    print("Total epoch:", epoch)
                    break

    # 预测嵌入和聚类分布
    def predict(self, X, adj):
        '''
        输入特征和邻接矩阵，输出嵌入和聚类分布
        '''
        X = torch.FloatTensor(X).to(self.device)
        adj = sparse_mx_to_torch_sparse_tensor(adj).to(self.device)
        z, q = self(X, adj)

        return z, q
