import torch
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA

from . utils import setup_seed, get_adj, preprocess_graph, cluster_post_process
from . model import PROST_NN, PROST_NN_sparse
from . calculate_PI import minmax_scaler, gau_filter, get_binary, get_sub, cal_prost_index, prepare_for_ot,cal_ot,cal_ot_gpu
from . plot import plot_gene_ot
from . calculate_PI import prepare_for_wavelet,wavelet_decompose,visualize_wavelet_multiscale, cal_wavelet_pattern_score

def cal_PI(adata, kernel_size=5, del_rate=0.01, platform = "visium", multiprocess=False):
    '''
    使用 PI 指数识别空间变异基因（SVGene），适用于空间转录组（ST）数据。
    
    参数说明
    ----------
    adata : Anndata
        形状为 `n_obs` × `n_vars` 的注释数据矩阵。行对应细胞，列对应基因。
    grid_size : int (默认: 5)
        插值的网格大小，仅在 `platform != visium` 时设置。
    kernel_size : int (默认: 5)
        定义用于形态学闭运算的核大小。核越大，能消除的前景内细小区域越多。
    del_rate : float (默认: 0.01)
        若某基因最大前景区域小于整体空间表达的 `del_rate`，则认为其无显著空间模式。
    platform : str ['visium','Slide-seq','Stereo-seq','osmFISH','SeqFISH' 或其他产生不规则点的平台] (默认: 'visium')
        产生 ST 数据的测序平台。
    multiprocess : bool
        Linux 下可用多线程大幅提升 PI 计算效率。
        
    返回值
    -------
    adata : Anndata
        adata.obs.PI : 每个基因的 PI 分数。
        adata.obs.SEP : 每个基因的可分性分数。
        adata.obs.SIG : 每个基因的显著性分数。
    '''
    
    adata = minmax_scaler(adata)
    adata = gau_filter(adata, platform, multiprocess=multiprocess)
    adata = get_binary(adata, platform, method = "iterative", multiprocess=multiprocess)
    adata = get_sub(adata, kernel_size, platform, del_rate)
    adata = cal_prost_index(adata, platform)
    print("\nPROST Index calculation completed !!")    
    return adata


def run_PNN(adata, SEED, init="leiden", n_clusters=5, res=0.2,
                adj_mode="neighbour", k_neighbors=7, min_distance=50, 
                key_added="PROST", lap_filter=2, 
                lr=0.1, tol=5e-3, max_epochs=500, post_processing=False, 
                pp_run_times=3, cuda=False):
    '''
    使用 PNN 识别空间结构域，适用于空间转录组（ST）数据。
    
    参数说明
    ----------
    adata : Anndata
        形状为 `n_obs` × `n_vars` 的注释数据矩阵。行对应细胞，列对应基因。
    SEED : int
        随机种子。
    init : str ["kmeans","mclust","louvain","leiden"] (默认: leiden)
        初始化聚类中心的方法。
    n_clusters : int (默认: 5)
        若已知空间结构域数量，则设置聚类数（用于 `init='kmeans'` 或 `init='mclust'`）。
    res : float (默认: 0.5)
        若未知空间结构域数量，则设置分辨率参数（用于 `init='kmeans'` 或 `init='mclust'`）。
    adj_mode : str ['neighbour','distance'] (默认: 'neighbour')
        构建细胞图的邻域模式。
    k_neighbors : int (默认: 7)
        若 `mode = 'neighbour'`，设置最近邻数量。
    min_distance : int (默认: 50)
        若 `mode = 'distance'`，设置最近邻距离。
    key_added : str (默认: 'PROST')
        PROST 生成的嵌入表示存储于 `adata.obsm` 的键名。
    lap_filter : int (默认: 2)
        堆叠拉普拉斯滤波器数量。
    lr : float (默认: 0.1)
        学习率。
    tol : float (默认: 5e-3)
        停止准则。若两次迭代聚类分配变化小于 `tol`，则停止。
    max_epochs : int (默认: 500)
        训练轮数。
    post_processing : bool (默认: False)
        是否对原始聚类结果进行后处理。
    pp_run_times : int (默认: 3)
        若 `post_processing=True`，设置后处理运行次数。
    cuda : bool (默认: False)
        是否使用 CUDA 加速。
    
    返回值
    -------
    adata : Anndata
        adata.obs['clustering'] : 每个点（细胞）的原始聚类标签。
        adata.obs['pp_clustering'] : 每个点（细胞）的后处理聚类标签。
        adata.obsm['PROST'] : PROST 生成的嵌入表示。
    '''

    setup_seed(SEED)
 
    #--------------------------------------------------------------------------
    if adj_mode=="neighbour":
        print(f"\nCalculating adjacency matrix, mode={adj_mode}, k_neighbors={k_neighbors}...")
        adj = get_adj(adata, mode = 'neighbour', k_neighbors = k_neighbors)
        adj = adj.toarray()
    elif adj_mode=="distance":
        print(f"\nCalculating adjacency matrix, mode={adj_mode}, min_distance={k_neighbors}...")
        adj = get_adj(adata, mode = 'distance', min_distance = min_distance)
    else:
        raise ValueError("adj_mode must input one of ['neighbour', 'distance']")
    

    #--------------------------------------------------------------------------
    num_pcs = int(min(50, adata.shape[1]))
    pca = PCA(n_components=num_pcs)
    print("\nRunning PCA ...")
    if sp.issparse(adata.X):
        pca.fit(adata.X.A)
        embed=pca.transform(adata.X.A)
    else:
        pca.fit(adata.X)
        embed=pca.transform(adata.X)
        
    #--------------------------------------------------------------------------
    if lap_filter>0 and lap_filter!=False:
        print('Laplacian Smoothing ...') # Graph laplacin smoothing 
        adj_norm = preprocess_graph(adj, lap_filter, norm='sym', renorm=False)
        for a in adj_norm:
            embed = a.dot(embed)

    #--------------------------------------------------------------------------
    PNN = PROST_NN(embed.shape[1], embed.shape[1], cuda)
    
    PNN.train_(embed, adj, init=init, n_clusters=n_clusters, res=res, tol=tol, 
               lr=lr, max_epochs=max_epochs, seed=SEED)  
    
    embed, prop = PNN.predict(embed, adj)
    print("Clustering completed !!")
    
    #--------------------------------------------------------------------------
    embed = embed.detach().cpu().numpy()
    y_pred = torch.argmax(prop, dim=1).data.cpu().numpy()   
    
    adata.obsm[key_added] = embed
    adata.obs["clustering"] = y_pred
    adata.obs["clustering"] = adata.obs["clustering"].astype('category')

    #--------------------------------------------------------------------------
    if post_processing:
        cluster_post_process(adata, adj_mode, 
                             k_neighbors = int(k_neighbors*3/2)+1, 
                             min_distance = min_distance*3/2, 
                             key_added = "pp_clustering", 
                             run_times = pp_run_times) 
    return adata
            

def run_PNN_sparse(adata, SEED, init="leiden", n_clusters=5, res=0.5,
                    k_neighbors=7, key_added="PROST", lap_filter=2, 
                    lr=0.1, tol=5e-3, max_epochs=500, cuda=False):
    '''
    PNN 的稀疏版本。
    
    参数说明
    ----------
    adata : Anndata
        形状为 `n_obs` × `n_vars` 的注释数据矩阵。行对应细胞，列对应基因。
    SEED : int
        随机种子。
    init : str ["kmeans","mclust","louvain","leiden"] (默认: leiden)
        初始化聚类中心的方法。
    n_clusters : int (默认: 5)
        若已知空间结构域数量，则设置聚类数（用于 `init='kmeans'` 或 `init='mclust'`）。
    res : float (默认: 0.5)
        若未知空间结构域数量，则设置分辨率参数（用于 `init='kmeans'` 或 `init='mclust'`）。
    k_neighbors : int (默认: 7)
        计算邻接矩阵时设置最近邻数量。
    key_added : str (默认: 'PROST')
        PROST 生成的嵌入表示存储于 `adata.obsm` 的键名。
    lap_filter : int (默认: 2)
        堆叠拉普拉斯滤波器数量。
    lr : float (默认: 0.1)
        学习率。
    tol : float (默认: 5e-3)
        停止准则。若两次迭代聚类分配变化小于 `tol`，则停止。
    max_epochs : int (默认: 500)
        训练轮数。
    cuda : bool (默认: False)
        是否使用 CUDA 加速。
    
    返回值
    -------
    adata : Anndata
        adata.obs['clustering'] : 每个点（细胞）的原始聚类标签。
        adata.obs['pp_clustering'] : 每个点（细胞）的后处理聚类标签。
        adata.obsm['PROST'] : PROST 生成的嵌入表示。
    '''

    setup_seed(SEED)
 
    #--------------------------------------------------------------------------
    print("\nCalculating adjacency matrix ...")
    adj = get_adj(adata, mode = 'neighbour', k_neighbors = k_neighbors)

    #--------------------------------------------------------------------------
    num_pcs = int(min(50, adata.shape[1]))
    pca = PCA(n_components=num_pcs)
    print("\nRunning PCA ...")
    if sp.issparse(adata.X):
        pca.fit(adata.X.A)
        embed=pca.transform(adata.X.A)
    else:
        pca.fit(adata.X)
        embed=pca.transform(adata.X)
        
    #--------------------------------------------------------------------------
    print('Laplacian Smoothing ...') # Graph laplacin smoothing 
    adj_norm = preprocess_graph(adj, lap_filter, norm='sym', renorm=False) 
    for a in adj_norm:
        embed = a.dot(embed)
           
    #--------------------------------------------------------------------------
    PNN = PROST_NN_sparse(embed.shape[1], embed.shape[1], cuda)
    
    PNN.train_(embed, adj, init=init, n_clusters=n_clusters, res=res, tol=tol, 
               lr=lr, max_epochs=max_epochs, seed=SEED)  
    
    embed, prop = PNN.predict(embed, adj)
    print("Clustering completed !!")
    
    #--------------------------------------------------------------------------
    embed = embed.detach().cpu().numpy()
    y_pred = torch.argmax(prop, dim=1).data.cpu().numpy()   
    
    adata.obsm[key_added] = embed
    adata.obs["clustering"] = y_pred
    adata.obs["clustering"] = adata.obs["clustering"].astype('category')

    return adata


#-----------------------------------------------------------------------
prepare_for_ot=prepare_for_ot
cal_ot=cal_ot
cal_ot_gpu=cal_ot_gpu
plot_gene_ot=plot_gene_ot








prepare_for_wavelet,wavelet_decompose,visualize_wavelet_multiscale=prepare_for_wavelet,wavelet_decompose,visualize_wavelet_multiscale
cal_wavelet_pattern_score=cal_wavelet_pattern_score