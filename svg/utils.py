import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
import scanpy as sc
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
import scipy.sparse as sp
import scipy.stats as stats
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from statsmodels.stats.multitest import fdrcorrection
import multiprocessing as mp
from tqdm import trange, tqdm
import cv2
from skimage.measure import label

 








# 方差稳定化变换
def var_stabilize(data):
    """
    对数据进行方差稳定化变换，减少均值与方差的相关性。
    """
    varx = np.var(data, 1)
    meanx = np.mean(data, 1)
    fun = lambda phi, varx, meanx : meanx + phi * meanx ** 2 - varx
    target_phi = least_squares(fun, x0 = 1, args = (varx, meanx))
    return np.log(data + 1 / (2 * target_phi.x))

# 最小最大归一化
def minmax_normalize(data):
    """
    对数据进行最小最大归一化，使数据范围在[0,1]之间。
    """
    maxdata = np.max(data)
    mindata = np.min(data)
    return (data - mindata)/(maxdata - mindata)


# 预处理基因表达数据，过滤低表达基因并可选方差稳定化
def pre_process(adata, percentage = 0.1, var_stabilization = True):
    """
    预处理基因表达数据，过滤低表达基因并进行方差稳定化。
    参数说明：
    adata: AnnData对象
    percentage: 过滤低表达基因的阈值
    var_stabilization: 是否进行方差稳定化
    返回：基因索引和处理后的表达矩阵
    """     
    if sp.issparse(adata.X):
        rawcount = adata.X.A.T
    else:
        rawcount = adata.X.T
        
    # 检查是否有全为常数的行，将其置零
    equal_rows = np.all(rawcount[:, 1:] == rawcount[:, :-1], axis=1)
    rawcount[equal_rows,:] = 0

    if percentage > 0:
        count_sum = np.sum(rawcount > 0, 1) 
        threshold = int(np.size(rawcount, 1) * percentage)
        gene_use = np.where(count_sum >= threshold)[0]
        print("\n正在过滤低表达基因 ...")
        rawcount = rawcount[gene_use, :]
    else:
        gene_use = np.array(range(len(rawcount)))
          
    if var_stabilization:
        print("\n对每个基因进行方差稳定化变换 ...")
        rawcount = var_stabilize(rawcount) 

    return gene_use, rawcount


def simple_preprocess(adata, percentage=0.1, var_stabilization=True, copy=True):
    """
    预处理基因表达数据：过滤低表达基因并进行方差稳定化。
    参数说明：
        adata: AnnData对象
        percentage: 过滤低表达基因的阈值（如0.1表示至少在10%点有表达）
        var_stabilization: 是否进行方差稳定化
        copy: 是否返回副本
    返回：过滤后的 AnnData（与原 AnnData 结构一致）
    """
 

    ad = adata.copy() if copy else adata

    # 获取表达矩阵，转置为 [基因, 细胞]
    if sp.issparse(ad.X):
        rawcount = ad.X.A.T
    else:
        rawcount = ad.X.T

    # 检查是否有全为常数的行，将其置零
    equal_rows = np.all(rawcount[:, 1:] == rawcount[:, :-1], axis=1)
    rawcount[equal_rows, :] = 0

    # 过滤低表达基因
    if percentage > 0:
        count_sum = np.sum(rawcount > 0, axis=1)
        threshold = int(rawcount.shape[1] * percentage)
        gene_use = np.where(count_sum >= threshold)[0]
        print(f"\n正在过滤低表达基因 ... 保留 {len(gene_use)} 个基因")
        rawcount = rawcount[gene_use, :]
    else:
        gene_use = np.arange(rawcount.shape[0])

    # 方差稳定化
    if var_stabilization:
        print("\n对每个基因进行方差稳定化变换 ...")
        rawcount = np.log1p(rawcount)

    # 构建新 AnnData
    adata_pi = sc.AnnData(rawcount.T)
    adata_pi.obs = ad.obs.copy()
    adata_pi.obsm = ad.obsm.copy()
    adata_pi.var = ad.var.iloc[gene_use].copy()
    adata_pi.var_names = ad.var_names[gene_use]
    if 'spatial' in ad.uns:
        adata_pi.uns['spatial'] = ad.uns['spatial']

    return adata_pi












