import torch
import torch.nn as nn
import torch.nn.functional as F
from . utils import preprocess_graph

# 图注意力层（密集版），实现类似GAT的机制
class GraphAttentionLayer(nn.Module):
    """
    简单的图注意力层，类似于 https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout 
        self.in_features = in_features 
        self.out_features = out_features 
        self.alpha = alpha 
        self.concat = concat 
        
        # 权重参数初始化
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) # xavier初始化
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414) # xavier初始化
        
        # LeakyReLU激活
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        前向传播
        h: 输入特征矩阵 [N, in_features]
        adj: 邻接矩阵 [N, N]
        """
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self.prepare_attentional_mechanism_input(Wh) # 计算注意力分数 [N, N]
        
        zero_vec = -9e15 * torch.ones_like(e) # 用极小值填充非邻接部分
        attention = torch.where(adj > 0, e, zero_vec)   # 仅保留邻接部分
        
        attention = F.softmax(attention, dim=1) # 按行归一化
        attention = F.dropout(attention, self.dropout,training=self.training) # dropout
        h_prime = torch.matmul(attention, Wh)  # 聚合邻居特征
        # [N, N].[N, out_features] => [N, out_features]
        
        if self.concat:
            return F.elu(h_prime) # 非最后一层用ELU激活
        else:
            return h_prime # 最后一层直接输出

    def prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[: self.out_features, :]) # 第一部分
        Wh2 = torch.matmul(Wh, self.a[self.out_features :, :]) # 第二部分
        # 广播加法，得到节点间注意力分数
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
   

# 稀疏专用的Spmm反向传播函数，只对稀疏区域做反向
class SpecialSpmmFunction(torch.autograd.Function):
   """仅对稀疏区域做反向传播的特殊函数。"""
   @staticmethod
   def forward(ctx, indices, values, shape, b):
       # indices不需要梯度
       assert indices.requires_grad == False
       a = torch.sparse_coo_tensor(indices, values, shape)
       ctx.save_for_backward(a, b)
       ctx.N = shape[0]
       return torch.matmul(a, b)

   @staticmethod
   def backward(ctx, grad_output):
       a, b = ctx.saved_tensors
       grad_values = grad_b = None
       if ctx.needs_input_grad[1]:
           grad_a_dense = grad_output.matmul(b.t())
           edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
           grad_values = grad_a_dense.view(-1)[edge_idx]
       if ctx.needs_input_grad[3]:
           grad_b = a.t().matmul(grad_output)
       return None, grad_values, None, grad_b

# 稀疏Spmm模块
class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

# 稀疏图注意力层，适用于大规模稀疏图
class SpGraphAttentionLayer(nn.Module):
    """
    稀疏版图注意力层，类似于 https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 权重参数初始化
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, X, adj):
        # 判断设备类型
        dv = 'cuda' if X.is_cuda else 'cpu'

        N = X.size()[0]
        
        # 获取边索引
        if adj.is_sparse:
            edge = adj.coalesce().indices()
        else:
            edge = adj.nonzero().t()

        h = torch.mm(X, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # 节点自注意力机制，拼接边两端特征
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        # 计算边的注意力分数
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        # 计算每个节点的归一化系数
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        # 聚合邻居特征
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # 非最后一层用ELU激活
            return F.elu(h_prime)
        else:
            # 最后一层直接输出
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'