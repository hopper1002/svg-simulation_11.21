import pandas as pd
import numpy as np
import os
import random
import numba
import torch
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

# 设置随机种子，保证实验可复现
def setup_seed(seed):
    """
    设置所有相关库的随机种子，保证结果可复现。
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

# 将scipy稀疏矩阵转换为torch稀疏张量
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy稀疏矩阵转换为torch稀疏张量。
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# 计算空间转录组数据的邻接矩阵
def get_adj(adata, mode = 'neighbour', k_neighbors = 7, min_distance = 150, self_loop = True):
    """
    计算空间转录组数据的邻接矩阵。
    参数说明：
    mode: 邻居定义方式，'neighbour'为最近邻，'distance'为距离阈值
    k_neighbors: 最近邻数量
    min_distance: 距离阈值
    self_loop: 是否包含自环
    返回邻接矩阵
    """
    spatial = adata.obsm["spatial"]   
    if mode == 'distance':
        assert min_distance is not None,"请为get_adj()设置min_distance"
        adj = metrics.pairwise_distances(spatial, metric='euclidean')
        adj[adj > min_distance] = 0
        if self_loop:
            adj += np.eye(adj.shape[0])  
        adj = np.int64(adj>0)
        return adj
    
    elif mode == 'neighbour':
        assert k_neighbors is not None,"请为get_adj()设置k_neighbors"
        adj = kneighbors_graph(spatial, n_neighbors = k_neighbors, include_self = self_loop)
        return adj
        

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



# 将二维图像索引转换为一维索引
@numba.jit
def get_image_idx_1D(image_idx_2d):
    """
    将二维图像索引转换为一维索引，用于后续一维基因计数恢复。
    """
    print("\n正在计算图像索引1D:")
    image_idx_1d = np.ones(np.max(image_idx_2d[:])).astype(int)
    for i in trange(1, np.max(image_idx_2d[:])+1):   
        image_idx_1d[i-1] = np.where(image_idx_2d.T.flatten() == i)[0]+1
    return image_idx_1d



# 一维基因表达数据转为二维空间图像
def make_image(genecount, locates, platform = "visium", get_image_idx = False, 
               grid_size = 20, interpolation_method='linear'):
    """
    将一维基因表达数据转换为二维空间插值图像。
    参数说明：
    genecount: 基因表达计数
    locates: 空间坐标
    platform: 测序平台
    get_image_idx: 是否返回一维索引
    grid_size: 网格大小
    interpolation_method: 插值方法
    返回二维图像和一维索引
    """ 
    if platform=="visium":
        xloc = np.round(locates[:, 0]).astype(int)
        maxx = np.max(xloc)
        minx = np.min(xloc)
        yloc = np.round(locates[:, 1]).astype(int)
        maxy = np.max(yloc)
        miny = np.min(yloc)
        
        image = np.zeros((maxy, maxx))    
        image_idx_2d = np.zeros((maxy, maxx)).astype(int)  
        for i in range(len(xloc)):
            temp_y = yloc[i]
            temp_x = xloc[i]
            temp_value = genecount[i]
            image[temp_y - 1, temp_x - 1] = temp_value
            image_idx_2d[temp_y - 1 , temp_x - 1] = i+1
            
        image = np.delete( image, range(miny - 1), 0)
        image = np.delete( image, range(minx - 1), 1)
        image_idx_2d = np.delete(image_idx_2d, range(miny - 1), 0) 
        image_idx_2d = np.delete(image_idx_2d, range(minx - 1), 1)
        image_idx_1d = np.ones(np.max(image_idx_2d[:])).astype(int)
        if get_image_idx:
            image_idx_1d = get_image_idx_1D(image_idx_2d)
                
        return image, image_idx_1d
    #--------------------------------------------------------------------------
    else:
        xloc = locates[:, 0]
        maxx, minx = np.max(xloc), np.min(xloc)

        yloc = locates[:, 1]
        maxy, miny = np.max(yloc), np.min(yloc)

        xloc_new = np.round(locates[:, 0]).astype(int)
        maxx_new, minx_new = np.max(xloc_new), np.min(xloc_new)
        
        yloc_new = np.round(locates[:, 1]).astype(int)
        maxy_new, miny_new = np.max(yloc_new), np.min(yloc_new)

        # 插值，将不规则空间表达插值到规则网格
        grid_x, grid_y = np.mgrid[minx_new: maxx_new+1: grid_size, miny_new: maxy_new+1: grid_size]       
        image = griddata(locates, genecount, (grid_x,grid_y), method = interpolation_method)

        return image, image.shape
        




# 二维基因图像恢复为一维表达计数
@numba.jit
def gene_img_flatten(I, image_idx_1d):
    """
    将二维插值基因图像恢复为一维基因表达计数。
    参数说明：
    I: 二维基因图像
    image_idx_1d: 一维索引
    返回一维基因表达计数
    """ 
    I_1d = I.T.flatten()
    output = np.zeros(image_idx_1d.shape)
    for ii in range(len(image_idx_1d)):
        idx = image_idx_1d[ii]
        output[ii] = I_1d[idx - 1]
    return output

# 对单个基因空间表达图像进行高斯滤波
def gau_filter_for_single_gene(gene_data, locates, platform = "visium", image_idx_1d = None):
    """
    对单个基因空间表达图像进行高斯滤波，平滑空间表达。
    参数说明：
    gene_data: 基因表达数据
    locates: 空间坐标
    platform: 测序平台
    image_idx_1d: 一维索引
    返回一维基因表达计数
    """ 
    if platform=="visium":
        I,_ = make_image(gene_data, locates, platform)  
        I = gaussian_filter(I, sigma = 1, truncate = 2)
        output = gene_img_flatten(I, image_idx_1d)
    #--------------------------------------------------------------------------
    else:
        I,_ = make_image(gene_data, locates, platform) 
        I = gaussian_filter(I, sigma = 1, truncate = 2)
        output = I.flatten()
    return output

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
      

# 利用空间邻域信息细化聚类标签
def refine_clusters(result, adj, p=0.5):
    """
    利用空间邻域信息细化聚类标签。
    参数说明：
    result: 初始聚类结果
    adj: 邻接矩阵
    p: 邻居标签变更比例阈值
    返回细化后的聚类标签
    """
    if sp.issparse(adj):
        adj = adj.A

    pred_after = []  
    for i in range(result.shape[0]):
        temp = list(adj[i])  
        temp_list = []
        for index, value in enumerate(temp):
            if value > 0:
                temp_list.append(index) 
        self_pred = result[i]
        neighbour_pred = []      
        for j in temp_list:
            neighbour_pred.append(result[j])
        # 判断邻居标签是否需要变更
        if (neighbour_pred.count(self_pred) < (len(neighbour_pred))*p) and (neighbour_pred.count(max(set(neighbour_pred), key=neighbour_pred.count))>(len(neighbour_pred))*p):
            pred_after.append(np.argmax(np.bincount(np.array(neighbour_pred))))
        else:
            pred_after.append(self_pred)
    return np.array(pred_after)
      

# 聚类后处理工具，结合邻域信息细化聚类标签
def cluster_post_process(adata, adj_mode, k_neighbors = None, min_distance = None, 
                         key_added = "pp_clustering", p = 0.5, run_times = 3):
    """
    聚类后处理工具，结合邻域信息细化聚类标签。
    参数说明：
    adata: AnnData对象
    adj_mode: 邻域定义方式
    k_neighbors: 最近邻数量
    min_distance: 距离阈值
    key_added: 结果存储的obs键名
    p: 邻居标签变更比例阈值
    run_times: 最大迭代次数
    返回处理后的AnnData对象
    """
    
    print("\n正在对聚类结果进行后处理 ...")
    clutser_result = adata.obs["clustering"]
    # 计算邻接矩阵
    if adj_mode=="neighbour":
        PP_adj = get_adj(adata, mode = 'neighbour', k_neighbors = k_neighbors)
        PP_adj = PP_adj.toarray()
    elif adj_mode=="distance":
        PP_adj = get_adj(adata, mode = "distance", min_distance = min_distance)
    else:
        raise ValueError("adj_mode 必须为 ['neighbour', 'distance'] 之一")

    result_final = pd.DataFrame(np.zeros(clutser_result.shape[0]))
    i = 1             
    while True:        
        clutser_result = refine_clusters(clutser_result, PP_adj, p)
        print("细化聚类标签, 第{}/{}次迭代".format(i,run_times))
        result_final.loc[:, i] = clutser_result        
        if result_final.loc[:, i].equals(result_final.loc[:, i-1]) or i == run_times:
            adata.obs[key_added] = np.array(result_final.loc[:, i])
            adata.obs[key_added] = adata.obs[key_added].astype('category')
            return adata
        i += 1

# 调用R的mclust算法进行聚类
def mclust(data, num_cluster, modelNames = 'EEE', random_seed = 818):
    """
    使用R的mclust算法进行聚类分析。
    参数说明：
    data: 输入数据
    num_cluster: 聚类数
    modelNames: 模型类型
    random_seed: 随机种子
    返回聚类结果
    """ 
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()  
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(data, num_cluster, modelNames)
    return np.array(res[-2])

# 计算Moran's I空间自相关指标
def calc_I(y, w):
    """
    计算Moran's I空间自相关指标。
    参数说明：
    y: 属性向量
    w: 空间权重矩阵
    返回Moran's I值
    """ 
    y = np.array(y)

    z = y - y.mean()
    z = z.reshape(len(z),1)
    zl = np.multiply(w, z)
    num = np.multiply(zl, z.T).sum()
    z2ss = (z * z).sum()
    return y.shape[0] / w.sum() * num / z2ss

# 批量计算Moran's I
def batch_morans_I(Y, w):
    """
    批量计算多个y向量的Moran's I。
    参数说明：
    Y: 多个y向量组成的矩阵，每一列为一个y
    w: 空间权重矩阵
    返回每个y的Moran's I数组
    """
    Y = np.array(Y)
    w = np.array(w)

    n, m = Y.shape
    
    mean_Y = np.mean(Y, axis=0, keepdims=True)
    Z = Y - mean_Y
    num = np.sum(w @ Z * Z, axis=0)
    denom = np.sum(Z**2, axis=0)
    I = n / np.sum(w) * num / denom
    
    return I

# 计算Geary's C空间自相关指标
def calc_C(w, y):
    """
    计算Geary's C空间自相关指标。
    参数说明：
    w: 空间权重矩阵
    y: 属性向量
    返回Geary's C值
    """   
    n = y.shape[0]
    s0 = w.sum()
    z = y - y.mean()
    z = z.reshape(len(z),1)
    z2ss = (z * z).sum()
    den = z2ss * s0 * 2.0
    a, b = w.nonzero()
    num = (w.data * ((y[a] - y[b]) ** 2)).sum()
    return (n - 1) * num / den

# 计算单个基因的空间统计指标及显著性
def cal_eachGene(arglist):
    """
    计算单个基因的空间统计指标及显著性，包括Moran's I、Geary's C及p值。
    参数说明：
    arglist: 包含所有计算所需参数的列表
    返回：各项指标及p值
    """
    # 解包参数
    gene_i, exp, w, permutations, n, n2, s1, s2, s02, E, V_norm = arglist

    # 计算Moran's I和Geary's C
    _moranI = calc_I(exp, w.todense())
    _gearyC = calc_C(w, exp)
    
    # 显著性检验相关计算
    z = exp - exp.mean()
    z2 = z ** 2
    z4 = z ** 4
    D = (z4.sum() / n) / ((z2.sum() / n) ** 2)
    A = n * ((n2 - 3 * n + 3) * s1 - n * s2 + 3 * s02)
    B = D * ((n2 - n) * s1 - 2 * n * s2 + 6 * s02)
    C = ((n - 1) * (n - 2) * (n - 3) * s02)
    E_2 = (A - B) / C
    V_rand = E_2 - E * E

    # 计算z分数
    z_norm = (_moranI-E) / V_norm**(1 / 2.0)
    z_rand = (_moranI-E) / V_rand**(1 / 2.0)

    # 计算p值
    _p_norm = stats.norm.sf(abs(z_norm))
    _p_rand = stats.norm.sf(abs(z_rand))

    # 如果设置了permutations，计算模拟p值
    if permutations:    
        data_perm = np.array([np.random.permutation(exp) for _ in range(permutations)])
        sim = batch_morans_I(data_perm.T, w.todense())
        sim = np.array(sim)
        larger = np.sum(sim >= _moranI)
        if (permutations - larger) < larger:
            larger = permutations - larger
        _p_sim = (larger+1) / (permutations+1)

        return [gene_i, _moranI, _gearyC, _p_norm, _p_rand, _p_sim]

    return [gene_i, _moranI, _gearyC, _p_norm, _p_rand]

# 统计每个基因的空间自相关性
def spatial_autocorrelation(adata, k = 10, permutations = None, multiprocess = True):
    """
    统计每个基因的空间自相关性，包括Moran's I、Geary's C及显著性检验。
    参数说明：
    adata: AnnData对象
    k: 邻居数量
    permutations: 随机置换次数
    multiprocess: 是否多进程加速
    返回AnnData对象，结果存储在adata.var
    """
    if sp.issparse(adata.X):
        genes_exp = adata.X.A
    else:
        genes_exp = adata.X
    spatial = adata.obsm['spatial'] 
    w = kneighbors_graph(spatial, n_neighbors = k, include_self = False).toarray()

    s0 = w.sum()
    s02 = s0 * s0
    t = w + w.transpose()
    s1 = np.multiply(t, t).sum()/2.0
    s2 = (np.array(w.sum(1) + w.sum(0).transpose()) ** 2).sum()
    n = len(genes_exp)
    n2 = n * n
    E = -1.0 / (n - 1)

    v_num = n2 * s1 - n * s2 + 3 * s02
    v_den = (n - 1) * (n + 1) * s02
    V_norm = v_num / v_den - (1.0 / (n - 1)) ** 2

    w = sp.csr_matrix(w)
    N_gene = genes_exp.shape[1]

    # 数据生成器
    def sel_data():
        for gene_i in range(N_gene):
            yield [gene_i, genes_exp[:, gene_i], w, permutations, n, n2, s1, s2, s02, E, V_norm]

    if multiprocess:
        num_cores = int(mp.cpu_count() / 2)         # 默认使用一半CPU核心
        with mp.Pool(processes=num_cores) as pool:
            results = list(tqdm(pool.imap(cal_eachGene, sel_data()), total=N_gene))
    else:
        results = list(tqdm(map(cal_eachGene, sel_data()), total=N_gene))

    col = ['idx', 'moranI', 'gearyC', 'p_norm', 'p_rand']
    if len(results[0]) == 6:
        col.append('p_sim')

    results = pd.DataFrame(results, columns=col)

    _, fdr_norm = fdrcorrection(results.p_norm, alpha=0.05)
    _, fdr_rand = fdrcorrection(results.p_rand, alpha=0.05)

    adata.var["Moran_I"] = results.moranI.values
    adata.var["Geary_C"] = results.gearyC.values
    adata.var["p_norm"] = results.p_norm.values # 正态假设下的p值
    adata.var["p_rand"] = results.p_rand.values # 随机化假设下的p值
    adata.var["fdr_norm"] = fdr_norm
    adata.var["fdr_rand"] = fdr_rand

    if permutations:
        _, fdr_sim = fdrcorrection(results.p_sim, alpha=0.05)
        adata.var["p_sim"] = results.p_sim.values
        adata.var["fdr_sim"] = fdr_sim
    
    return adata

# 邻接矩阵预处理，归一化并生成拉普拉斯矩阵
def preprocess_graph(adj, layer = 2, norm = 'sym', renorm = True, k = 2/3):
    """
    邻接矩阵预处理，归一化并生成拉普拉斯矩阵。
    参数说明：
    adj: 邻接矩阵
    layer: 层数
    norm: 归一化方式
    renorm: 是否加单位阵
    k: 拉普拉斯系数
    返回处理后的邻接矩阵列表
    """
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj  
    rowsum = np.array(adj_.sum(1)) 
    
    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
        
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized  
        
    reg = [k] * layer
    adjs = []
    for i in range(len(reg)):
        adjs.append(ident-(reg[i] * laplacian))
    return adjs

# 空间转录组特征选择工具
def feature_selection(adata, selected_gene_name = None, by = 'prost', n_top_genes = 3000):
    """
    空间转录组特征选择工具。
    参数说明：
    adata: AnnData对象
    selected_gene_name: 手动指定基因名列表
    by: 特征选择方法
    n_top_genes: 选择的基因数量
    返回仅包含选中基因的AnnData对象
    """
    if selected_gene_name is None:
        if by == "prost":
            try:
                pi_score = adata.var["PI"]
            except:
                raise KeyError("在adata.var中找不到'PI'，请先运行'PROST.cal_prost_index()'！")
            pi_score = adata.var["PI"]
            sorted_score = pi_score.sort_values(ascending = False)
            gene_num = np.sum(sorted_score>0)
            selected_num = np.minimum(gene_num, n_top_genes)
            selected_gene_name = list(sorted_score[:selected_num].index)          
        elif by == "scanpy":
            sc.pp.highly_variable_genes(adata, n_top_genes = n_top_genes)
            adata = adata[:, adata.var.highly_variable]
            return adata
    else:    
        assert isinstance(selected_gene_name, list),"请将'selected_gene_name'输入为list类型！"
    selected_gene_name = [i.upper() for i in selected_gene_name]
    raw_gene_name = [i.upper() for i in list(adata.var_names)]
    
    adata.var['selected'] = False
    for i in range(len(raw_gene_name)):
        name = raw_gene_name[i]
        if name in selected_gene_name:
            adata.var['selected'][i] = True

    adata = adata[:, adata.var.selected]
    return adata
    

# DLPFC数据集聚类评估指标计算
def cal_metrics_for_DLPFC(labels_pred, labels_true_path=None, print_result = True):
    """
    计算DLPFC数据集聚类评估指标，包括ARI、NMI、轮廓系数等。
    参数说明：
    labels_pred: 预测标签
    labels_true_path: 真实标签csv路径
    print_result: 是否打印结果
    返回各项指标
    """
    # 处理真实标签
    labels_true = pd.read_csv(labels_true_path)
    labels_true['ground_truth'] = labels_true['ground_truth'].str[-1]
    labels_true = labels_true.fillna(8)   
    for i in range(labels_true.shape[0]):
        temp = labels_true['ground_truth'].iloc[i]
        if temp == 'M':
            labels_true['ground_truth'].iloc[i] = 7       
    labels_true = pd.DataFrame(labels_true['ground_truth'], dtype=np.int64).values
    labels_true = labels_true[:,0]    
    #
    ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
    AMI = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    v_measure_score = metrics.v_measure_score(labels_true, labels_pred)
    silhouette_score = metrics.silhouette_score(np.array(labels_true).reshape(-1, 1), np.array(labels_pred).reshape(-1, 1).ravel())
    if print_result:
        print('\nARI =', ARI, '\nAMI =', AMI, '\nNMI =', NMI, 
              '\nv_measure_score =', v_measure_score, '\nsilhouette_score =',silhouette_score,
              '\n==================================================================')
    return ARI, NMI, silhouette_score

# 模拟空间转录组数据的dropout和噪声
def simulateH5Data(adata, rr=0.0, mu=0.0, sigma=1.0, alpha=1.0):
    """
    模拟空间转录组数据的dropout和高斯噪声。
    参数说明：
    adata: AnnData对象
    rr: dropout比例
    mu: 高斯噪声均值
    sigma: 高斯噪声标准差
    alpha: 噪声缩放系数
    返回加噪声和dropout后的AnnData对象
    """
    if rr > 1 or rr < 0:
        print("警告！Dropout比例不合法！")
        return 0
    print("\n正在运行simulateH5Data...")

    import numpy as np
    from random import sample
    import scipy.sparse as sp
    import copy
    
    # 获取表达矩阵
    issparse = 0
    if sp.issparse(adata.X):
        data_ori_dense = adata.X.A
        issparse = 1
    else:
        data_ori_dense = adata.X
    
    # 添加高斯噪声
    n_r, n_c = len(data_ori_dense), len(data_ori_dense[0])
    Gnoise = np.random.normal(mu, sigma, (n_r, n_c))
    data_ori_dense = data_ori_dense + alpha * Gnoise
    
    data_ori_dense = np.clip(data_ori_dense, 0, None)
    
    print(f"添加高斯噪声: {alpha} * gauss({mu}, {sigma})")

    # 从非零元素中采样
    flagXY = np.where(data_ori_dense != 0)      
    ncount = len(flagXY[0])

    # 随机采样rr比例的元素置零
    flag = sample(range(ncount), k=int(rr * ncount))
    dropX, dropY = flagXY[0][flag], flagXY[1][flag]

    # 更新AnnData对象
    data_new = data_ori_dense.copy()
    for dx, dy in zip(dropX, dropY):
        data_new[dx, dy] = 0.0
    reCount = (data_new != 0).sum()
    if issparse:
        data_new = sp.csr_matrix(data_new)
    print(f"Dropout比例 = {rr}")

    # 返回新AnnData对象
    newAdata = copy.deepcopy(adata)
    newAdata.X = data_new

    # 注意：不会更新元数据！
    print(f"完成！剩余 {100 * round(reCount/ncount, 2)}% ({reCount}/{ncount})") 
    return newAdata



#-----------------------------------------------------------------------



