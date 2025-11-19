import pandas as pd
import numpy as np
import cv2
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
import scanpy as sc
from tqdm import trange, tqdm
from skimage.measure import label
import multiprocessing as mp
from . utils import pre_process, make_image, gene_img_flatten, minmax_normalize, gau_filter_for_single_gene



def prepare_for_PI(adata, grid_size=20, percentage=0.1, platform="visium"):    
    selected_gene_idxs, postcount = pre_process(adata, percentage, var_stabilization = False)

    if platform=="visium" or platform=="ST":
        try:
            locates = adata.obs[["array_row","array_col"]]
            locates = locates.values if isinstance(locates, pd.DataFrame) else locates
        except:
            locates = adata.obsm["spatial"]
            locates = locates.values if isinstance(locates, pd.DataFrame) else locates
        if np.min(locates) == 0:
            locates += 1
        _, image_idx = make_image(postcount[0], locates, platform, get_image_idx = True, grid_size=grid_size)
        adata = adata[:, selected_gene_idxs]
        sc.pp.filter_genes(adata, min_cells=3)
        adata.obs['image_idx_1d'] = image_idx

    else:
        locates = adata.obsm["spatial"].astype(float)
        locates = locates.values if isinstance(locates, pd.DataFrame) else locates
        _, shape = make_image(postcount[0], locates, platform, grid_size=grid_size)
        assert shape[0]>1 and shape[1]>1, f"Gene image size is {shape[0]} * {shape[1]} after interpolation. Please set a smaller grid size!!"
        print (f"Spatial gene expression is interpolated into images of size [{shape[0]} * {shape[1]}]")
        adata = adata[:, selected_gene_idxs]
        sc.pp.filter_genes(adata, min_cells=50)
        adata.uns['shape'] = shape
    adata.uns['grid_size'] = grid_size
    adata.uns['locates'] = locates
    return adata


def minmax_scaler(adata):
    if sp.issparse(adata.X):
        data = adata.X.A.T
    else:
        data = adata.X.T        

    print('\nNormalize each geneing...')
    nor_counts = data.copy().T
    _nor_maxdata = np.max(nor_counts, 0)
    _nor_mindata = np.min(nor_counts, 0)
    nor_counts = (nor_counts - _nor_mindata) / (_nor_maxdata - _nor_mindata)
    adata.uns['nor_counts'] = nor_counts.T
    return adata


def gau_filter_for_single_gene(arglist):
    gene_data, locates, platform, image_idx_1d = arglist

    I,_ = make_image(gene_data, locates, platform) 
    I = gaussian_filter(I, sigma = 1, truncate = 2)
    if platform=="visium":
        I_1d = I.T.flatten()
        output = I_1d[image_idx_1d-1]
    else:
        output = I.flatten()
    return output


def gau_filter(adata, platform="visium", multiprocess=False):
    gene_data = adata.uns['nor_counts']
    locates = adata.uns['locates']
    N_gene = len(gene_data)

    print('\nGaussian filtering...')
    if platform=="visium":
        image_idx_1dd = adata.obs['image_idx_1d'].astype(int).values

    def sel_data():  # data generater
        for gene_i in range(N_gene):
            if platform=="visium":
                yield [gene_data[gene_i], locates, platform, image_idx_1dd]
            else:
                yield [gene_data[gene_i], locates, platform, '']
                
    if multiprocess:
        num_cores = int(mp.cpu_count() / 2)         # default core is half of total
        with mp.Pool(processes=num_cores) as pool:
            gau_fea = list(tqdm(pool.imap(gau_filter_for_single_gene, sel_data()), total=N_gene))
    else:
        gau_fea = list(tqdm(map(gau_filter_for_single_gene, sel_data()), total=N_gene))
        
    adata.uns['gau_fea'] = np.array(gau_fea, dtype=np.float64) 
    return adata


def _iget_binary(arglists):
    fig1, locates, platform, method, r1 = arglists

    if platform=="visium":
        Im, _ = make_image(fig1, locates, platform)
        if method == "iterative":    
            m, n = Im.shape       
            zd = float(np.max(Im))
            zx = float(np.min(Im))
            Th = float((zd+zx))/2
            while True:
                S0 = 0.0; n0 = 0.0; S1 = 0.0; n1 = 0.0
                flag = Im >= Th
                S1 = Im[flag].sum()
                n1 = flag.sum()
                S0 = Im[~flag].sum()
                n0 = (~flag).sum()
                T0 = S0/n0; T1 = S1/n1
                if abs(Th - ((T0 + T1)/2)) < 0.0001:
                    break
                else:
                    Th = (T0 + T1)/2 
                        
        elif method == "otsu":
            thres_list = np.arange(0.01,0.995,0.025)
            temp_std = np.zeros(thres_list.shape)
            for iii in range(len(thres_list)):
                temp_thres = thres_list[iii]
                q1 = fig1 > temp_thres
                b1 = fig1 <= temp_thres
                qv = r1[q1]
                bv = r1[b1]
                if len(qv) >= len(r1) * 0.15:
                    temp_std[iii] = (len(qv) * np.std(qv) + len(bv) * np.std(bv)) / len(fig1)
                else:
                    temp_std[iii] = 1e4
            Th = thres_list[temp_std == np.min(temp_std)]
    #--------------------------------------------------------------------------        
    else:               
        if method == "iterative":        
            zd = float(np.nanmax(fig1))
            zx = float(np.nanmin(fig1))
            Th = float((zd+zx))/2
            while True:                       
                S1 = np.sum(fig1[fig1>=Th])
                n1 = len(fig1[fig1>=Th])
                S0 = np.sum(fig1[fig1<Th])
                n0 = len(fig1[fig1<Th])
                T0 = S0/n0; T1 = S1/n1
                if abs(Th - ((T0 + T1)/2)) < 0.0001:
                    break
                else:
                    Th = (T0 + T1)/2 

        elif method == "otsu":
            # for ii in trange(len(gene_data)):
            img = fig1.reshape(locates)
            Th2, a_img = cv2.threshold(img.astype(np.uint8), 0, 255, cv2.THRESH_OTSU) 

    return fig1 >= Th


def get_binary(adata, platform="visium", method = "iterative", multiprocess=False):
    gene_data = adata.uns['gau_fea']
    if sp.issparse(adata.X):
        raw_gene_data = adata.X.A.T
    else:
        raw_gene_data = adata.X.T

    if platform=="visium":
        locates = adata.uns['locates']
    else:
        locates = adata.uns['shape']
    
    print('\nBinary segmentation for each gene:')
    
    N_gene = len(gene_data)
    def sel_data():  # data generater
        for gene_i in range(N_gene):
            yield [gene_data[gene_i, :], locates, platform, method, raw_gene_data[gene_i,:]]

    if multiprocess:
        num_cores = int(mp.cpu_count() / 2)         # default core is half of total
        with mp.Pool(processes=num_cores) as pool:
            output = list(tqdm(pool.imap(_iget_binary, sel_data()), total=N_gene))
    else:
        output = list(tqdm(map(_iget_binary, sel_data()), total=N_gene))
    output = np.array(output, dtype=np.float64) + 0.0
    adata.uns['binary_image'] = output
    return adata


def get_sub(adata, kernel_size = 5, platform="visium",del_rate = 0.01): 
    gene_data = adata.uns['binary_image']
    locates = adata.uns['locates']
        
    print('\nSpliting subregions for each gene:')
    #--------------------------------------------------------------------------
    if platform=="visium":
        image_idx_1d = adata.obs['image_idx_1d']
        output = np.zeros(gene_data.shape)
        del_index = np.ones(gene_data.shape[0])
        for i in trange(len(gene_data)):
            temp_data = gene_data[i, :]
            temp_i, _ = make_image(temp_data, locates)      
            kernel = np.ones((kernel_size,kernel_size), np.uint8)
            temp_i = cv2.morphologyEx(temp_i, cv2.MORPH_CLOSE, kernel) # close
            region_label = label(temp_i)
            T = np.zeros(region_label.shape)
            classes = np.max(np.unique(region_label)) + 1      
            len_list = np.zeros(classes)     
            for j in range(classes):
                len_list[j] = len(region_label[region_label == j])
            cond = len_list >= gene_data.shape[1] * 0.01        
            if len(np.where(cond[1:] == True)[0]) == 0:
                del_index[i] = 0
            indexes = np.where(cond == True)[0]       
            for j in range(len(indexes)):
                tar_num = indexes[j]
                tar_locs = region_label == tar_num
                T[tar_locs] = j
            targe_image = T * (temp_i > 0)
            classes_n = np.max(np.unique(targe_image)).astype(int) + 1       
            len_list_n = np.zeros(classes_n)        
            for j in range(classes_n):
                len_list_n[j] = len(targe_image[targe_image == j])            
            if len(len_list_n) > 1:                        
                if np.max(len_list_n[1:]) < gene_data.shape[1] * del_rate:
                    del_index[i] = 0
            else:
                del_index[i] = 0
            output[i, :] = gene_img_flatten(targe_image, image_idx_1d)      
    #--------------------------------------------------------------------------
    else:
        output = np.zeros((gene_data.shape[0], adata.uns['shape'][0]*adata.uns['shape'][1]))
        del_index = np.ones(gene_data.shape[0])
        for i in trange(len(gene_data)):
            temp_data = gene_data[i, :]
            temp_i = temp_data.reshape(adata.uns['shape'])     
            kernel = np.ones((kernel_size,kernel_size), np.uint8)
            temp_i = cv2.morphologyEx(temp_i, cv2.MORPH_CLOSE, kernel)
            region_label = label(temp_i)
            T = np.zeros(region_label.shape)
            classes = np.max(region_label) + 1      
            len_list = np.zeros(classes)
            for j in range(classes):
                len_list[j] = len(region_label[region_label == j])
            cond = len_list >= gene_data.shape[1] * 0.002        
            if len(np.where(cond[1:] == True)[0]) == 0:
                del_index[i] = 0
            indexes = np.where(cond == True)[0]       
            for j in range(len(indexes)):
                tar_num = indexes[j]
                tar_locs = region_label == tar_num
                T[tar_locs] = j
            targe_image = T * (temp_i > 0)
            classes_n = np.max(np.unique(targe_image)).astype(int) + 1       
            len_list_n = np.zeros(classes_n)        
            for j in range(classes_n):
                len_list_n[j] = len(targe_image[targe_image == j])            
            if len(len_list_n) > 1:                        
                if np.max(len_list_n[1:]) < gene_data.shape[1] * del_rate:
                    del_index[i] = 0
            else:
                del_index[i] = 0
            output[i, :] = targe_image.flatten()
    #--------------------------------------------------------------------------
    adata.uns['subregions'] = output
    adata.uns['del_index'] = del_index.astype(int)  
    return adata


def cal_prost_index(adata, platform="visium"):
    data = adata.uns['nor_counts']
    subregions = adata.uns['subregions']
    del_idx = adata.uns['del_index']
    
    print('\nComputing PROST Index for each gene:')
    #--------------------------------------------------------------------------
    if platform=="visium":
        SEP = np.zeros(len(data))
        SIG = np.zeros(len(data))
        region_number = np.zeros(len(data))
        
        for i in trange(len(data)): 
            temp_raw = data[i, :]
            temp_label = subregions[i, :]
            back_value = temp_raw[temp_label == 0]
            back_value = back_value[back_value > 0]
            if back_value.size == 0:
                back_value = 0  
            class_mean = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
            class_var = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
            class_std = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
            class_len = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
         
            for ii in range(max(np.unique(temp_label)).astype(int) + 1):
                Temp = temp_raw[temp_label == ii]
                if Temp.size == 0:
                    class_value = 0
                else:
                    class_value = Temp
                class_mean[ii] = np.mean(class_value)
                class_var[ii] = np.var(class_value)
                class_std[ii] = np.std(class_value)
                if isinstance(class_value, int):
                    if class_value == 0:
                        class_len[ii] = 0
                    else:              
                        class_len[ii] = len(class_value) - 1               
                else:
                    if class_value.size == 0:
                        class_len[ii] = 0
                    else:              
                        class_len[ii] = len(class_value) - 1
                        
            target_class = np.where(class_mean > 0)[0]
            class_mean = class_mean[target_class]
            class_std = class_std[target_class]
            class_var = class_var[target_class]
            class_len = class_len[target_class]
            
            # Calculate Separability and Significance
            SEP[i] = 1 - sum((class_len * class_var)) / ((len(temp_raw)-1) * np.var(temp_raw))
            SIG[i] = (np.mean(class_mean) - np.mean(back_value)) / sum(class_std / class_mean) 
            region_number[i] = len(class_len)
            del class_mean, class_var, class_len, class_std
                     
        # Pattern Index    
        PI = minmax_normalize(SEP) * minmax_normalize(SIG)
        PI = PI * del_idx   
        adata.var["SEP"] = SEP
        adata.var["SIG"] = SIG
        adata.var["PI"] = PI
    #--------------------------------------------------------------------------
    else:
        locates = adata.uns['locates']
        SEP = np.zeros(len(data))
        SIG = np.zeros(len(data))
        for i in trange(len(data)): 
            temp_raw = data[i, :]
            temp_img,_ = make_image(temp_raw, locates, platform)
            temp_raw = temp_img.flatten()
            temp_label = subregions[i, :]
            back_value = temp_raw[temp_label == 0]
            back_value = back_value[back_value > 0]
            
            if back_value.size == 0:
                back_value = 0  
            class_mean = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
            class_var = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
            class_std = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
            class_len = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
         
            for ii in range(max(np.unique(temp_label)).astype(int) + 1):
                Temp = temp_raw[temp_label == ii]
                if Temp.size == 0:
                    class_value = 0
                else:
                    class_value = Temp
                class_mean[ii] = np.nanmean(class_value)
                class_var[ii] = np.nanvar(class_value)
                class_std[ii] = np.nanstd(class_value)
                if isinstance(class_value, int):
                    if class_value == 0:
                        class_len[ii] = 0
                    else:              
                        class_len[ii] = len(class_value) - 1               
                else:
                    if class_value.size == 0:
                        class_len[ii] = 0
                    else:              
                        class_len[ii] = len(class_value) - 1
                        
            target_class = np.where(class_mean > 0)[0]
            class_mean = class_mean[target_class]
            class_std = class_std[target_class]
            class_var = class_var[target_class]
            class_len = class_len[target_class]
            
            # Calculate Separability and Significance
            SEP[i] = 1 - sum((class_len * class_var)) / ((len(temp_raw)-1) * np.nanvar(temp_raw))
            SIG[i] = (np.mean(class_mean) - np.mean(back_value)) / sum(class_std / class_mean)      
            del class_mean, class_var, class_len, class_std
            
        # Pattern Index    
        PI = minmax_normalize(SEP) * minmax_normalize(SIG)
        PI = PI * del_idx
        adata.var["SEP"] = SEP
        adata.var["SIG"] = SIG
        adata.var["PI"] = PI
        
        adata.uns['shape'] = []
    #--------------------------------------------------------------------------
    adata.uns['nor_counts'] = []
    adata.uns['binary_image'] = []
    adata.uns['subregions'] = []
    adata.uns['del_index'] = []
    
    return adata




#ot-----------------------------------------------------------------------

def prepare_for_ot(adata, platform="visium", min_cells=3, coord_key=None, metric="euclidean", copy=False, percentage=0.1):
    """
    为基于最优传输(OT)的 SVG 度量做准备：
    - 用 pre_process 剔除低表达基因
    - 将每个基因表达归一化为在各 spot 上的概率分布（总质量=1）
    - 目标为相同支持上的均匀分布

    参数:
      platform: "visium" | "ST" | 其它
      min_cells: 过滤低表达基因的阈值
      coord_key: None(自动) | (col1,col2) 从 obs 取 | "key" 从 obsm 取
      metric: 代价度量名称（仅作为元信息保存）
      copy: 是否返回副本
      percentage: 剔除低表达基因的比例（同 prepare_for_PI）
    """
    ad = adata.copy() if copy else adata

    # 用 pre_process 剔除低表达基因
    selected_gene_idxs, postcount = pre_process(ad, percentage, var_stabilization=False)
    ad = ad[:, selected_gene_idxs]
    sc.pp.filter_genes(ad, min_cells=min_cells)

    # 取坐标
    if coord_key is not None:
        if isinstance(coord_key, (list, tuple)) and len(coord_key) == 2:
            locates = ad.obs[list(coord_key)]
            locates = locates.values if isinstance(locates, pd.DataFrame) else locates
        elif isinstance(coord_key, str):
            locates = ad.obsm[coord_key]
            locates = locates.values if isinstance(locates, pd.DataFrame) else locates
        else:
            raise ValueError("coord_key 必须为 None、长度为2的 obs 列名序列，或一个 obsm 键名(str)。")
    else:
        if platform in ["visium", "ST"]:
            try:
                locates = ad.obs[["array_row", "array_col"]]
                locates = locates.values if isinstance(locates, pd.DataFrame) else locates
            except Exception:
                locates = ad.obsm["spatial"]
                locates = locates.values if isinstance(locates, pd.DataFrame) else locates
        else:
            locates = ad.obsm["spatial"]
            locates = locates.values if isinstance(locates, pd.DataFrame) else locates
    locates = np.asarray(locates, dtype=float)

    # 构建每个基因在 spots 上的概率分布
    if sp.issparse(ad.X):
        counts = ad.X.A.T  # 形状: genes x spots
    else:
        counts = ad.X.T
    counts = counts.astype(np.float64, copy=False)
    counts[counts < 0] = 0.0

    gene_mass = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, gene_mass, out=np.zeros_like(counts), where=gene_mass > 0)

    # 同一支持上的均匀目标分布
    n_spots = counts.shape[1]
    uniform = np.full(n_spots, 1.0 / n_spots, dtype=np.float64)

    # 保存到 adata.uns
    ad.uns["ot_prob"] = probs                # [n_genes, n_spots]
    ad.uns["ot_uniform"] = uniform           # [n_spots]
    ad.uns["ot_locates"] = locates           # [n_spots, 2]
    ad.uns["ot_metric"] = metric
    ad.uns["ot_platform"] = platform

    return ad




def cal_ot(adata, method="emd", reg=0.05, coord_scale="minmax", cost_norm="max",
           return_plan=False, verbose=False):
    """
    基于 POT 计算每个基因的 OT(Wasserstein) 距离，度量其由当前空间分布搬运到均匀分布的代价。

    依赖: 请先运行 prepare_for_ot，使 adata.uns 含 ot_prob/ot_uniform/ot_locates。

    参数:
      method: "emd" | "sinkhorn"（EMD精确、慢；Sinkhorn近似、快，需 reg）
      reg: sinkhorn 正则系数
      coord_scale: "minmax" | "zscore" | "none"  对坐标做尺度归一化
      cost_norm: "max" | "median" | "none"       对成本矩阵做尺度归一化，便于跨样本可比
      return_plan: 是否返回最后一个基因的运输计划 T（调试用）
      verbose: 显示进度条

    结果:
      写入 adata.var["OT"] 作为每个基因的 Wasserstein 距离
      记录归一化与方法信息到 adata.uns
    """
    import warnings
    try:
        import ot
    except ImportError as e:
        raise ImportError("需要安装 POT 库: pip install POT") from e

    assert "ot_prob" in adata.uns and "ot_uniform" in adata.uns and "ot_locates" in adata.uns, \
        "请先运行 prepare_for_ot。"

    P = adata.uns["ot_prob"]     # [n_genes, n_spots]
    u = adata.uns["ot_uniform"]  # [n_spots]
    X = np.asarray(adata.uns["ot_locates"], dtype=float)  # [n_spots, 2]
    n_genes, n_spots = P.shape
    assert u.shape[0] == n_spots and X.shape[0] == n_spots, "分布与坐标尺寸不一致。"

    # 坐标尺度归一化
    if coord_scale == "minmax":
        rng = X.ptp(axis=0)
        rng[rng == 0] = 1.0
        Xn = (X - X.min(axis=0)) / rng
    elif coord_scale == "zscore":
        std = X.std(axis=0)
        std[std == 0] = 1.0
        Xn = (X - X.mean(axis=0)) / std
    else:
        Xn = X

    # 成本矩阵 C（欧氏距离）
    from scipy.spatial.distance import cdist
    if n_spots > 6000:
        warnings.warn(f"spot 数为 {n_spots}，成本矩阵将较大（~{(n_spots**2)/1e6:.1f}M 元素），请考虑用 sinkhorn 或降采样。")
    C = cdist(Xn, Xn, metric=adata.uns.get("ot_metric", "euclidean")).astype(np.float64)

    # 成本矩阵归一化
    if cost_norm == "max":
        m = C.max()
        if m > 0:
            C = C / m
    elif cost_norm == "median":
        med = np.median(C[C > 0])
        if med > 0:
            C = C / med

    # 确保概率归一
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    u = u / (u.sum() + 1e-12)

    dists = np.zeros(n_genes, dtype=np.float64)
    last_T = None

    iterator = range(n_genes)
    if verbose:
        from tqdm import tqdm as _tqdm
        iterator = _tqdm(iterator, total=n_genes, desc="OT")

    if method == "emd":
        for i in iterator:
            a = P[i]
            dists[i] = ot.emd2(a, u, C)
            if return_plan and i == n_genes - 1:
                last_T = ot.emd(a, u, C)
    elif method == "sinkhorn":
        for i in iterator:
            a = P[i]
            # sinkhorn2 返回正则化的运输成本
            val = ot.sinkhorn2(a, u, C, reg)
            # 兼容不同返回形式
            if isinstance(val, (tuple, list)):
                val = val[0]
            dists[i] = float(np.asarray(val).squeeze())
            if return_plan and i == n_genes - 1:
                last_T = ot.sinkhorn(a, u, C, reg)
    else:
        raise ValueError("method 只能是 'emd' 或 'sinkhorn'。")

    adata.var["OT"] = dists
    adata.uns["ot_coord_scale"] = coord_scale
    adata.uns["ot_cost_norm"] = cost_norm
    adata.uns["ot_method"] = method
    adata.uns["ot_reg"] = reg

    if return_plan:
        return adata, last_T
    return adata


def cal_ot_gpu(adata, reg=0.05, coord_scale="minmax", cost_norm="max",
               verbose=False, debias=False, p=2, batch_size=256, device=None):
    """
    使用 cupy/torch 实现 Sinkhorn 迭代计算每个基因到均匀分布的正则化 OT 成本。
    - 优先使用 cupy+CUDA；若不可用则退回 torch（GPU/CPU）。
    - debias=True 时计算 Sinkhorn divergence: OT(a,u) - 0.5*OT(a,a) - 0.5*OT(u,u)
    """
    import numpy as np
    import scipy.sparse as sp

    assert "ot_prob" in adata.uns and "ot_uniform" in adata.uns and "ot_locates" in adata.uns, \
        "请先运行 prepare_for_ot。"

    P = adata.uns["ot_prob"]     # [n_genes, n_spots]
    u = adata.uns["ot_uniform"]  # [n_spots]
    X = np.asarray(adata.uns["ot_locates"], dtype=float)  # [n_spots, 2]
    n_genes, n_spots = P.shape

    # 坐标归一化
    if coord_scale == "minmax":
        rng = X.ptp(axis=0)
        rng[rng == 0] = 1.0
        Xn = (X - X.min(axis=0)) / rng
    elif coord_scale == "zscore":
        std = X.std(axis=0)
        std[std == 0] = 1.0
        Xn = (X - X.mean(axis=0)) / std
    else:
        Xn = X

    # 概率归一
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    u = u / (u.sum() + 1e-12)

    # 设备与引擎选择
    engine = "torch"
    dev = "cpu"
    try:
        import cupy as cp
        import cupy.cuda.runtime as cur
        if (device or "cuda").startswith("cuda") and cur.getDeviceCount() > 0:
            engine = "cupy"
    except Exception:
        engine = "torch"

    # torch 后备
    import torch
    if engine == "cupy":
        dev = "cuda"
    else:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 成本矩阵构建
    def build_cost_and_kernel_cupy(Xg, eps):
        x = cp.asarray(Xg, dtype=cp.float32)  # [N,D]
        diff = x[:, None, :] - x[None, :, :]
        d2 = cp.sum(diff * diff, axis=2)  # 欧氏距离平方
        if p == 2:
            C = d2
        elif p == 1:
            C = cp.sqrt(d2)
        else:
            C = cp.power(cp.sqrt(d2), p)
        # 成本归一化
        if cost_norm == "max":
            m = C.max()
            if m > 0:
                C = C / m
        elif cost_norm == "median":
            med = cp.median(C[C > 0])
            if med > 0:
                C = C / med
        K = cp.exp(-C / max(eps, 1e-8))
        return C.astype(cp.float32), K.astype(cp.float32)

    def build_cost_and_kernel_torch(Xg, eps, device):
        xt = torch.tensor(Xg, dtype=torch.float32, device=device)  # [N,D]
        d = torch.cdist(xt, xt, p=2)  # 欧氏距离
        if p == 2:
            C = d.pow(2)
        elif p == 1:
            C = d
        else:
            C = d.pow(p)
        # 成本归一化
        if cost_norm == "max":
            m = torch.max(C)
            if m.item() > 0:
                C = C / m
        elif cost_norm == "median":
            med = torch.median(C[C > 0])
            if med.item() > 0:
                C = C / med
        K = torch.exp(-C / max(eps, 1e-8))
        return C.to(torch.float32), K.to(torch.float32)

    # Sinkhorn 批量迭代
    def sinkhorn_batch_cupy(A, b, C, K, eps, n_iter=200, tol=1e-6):
        # A: [B,N], b:[N], C:[N,N], K:[N,N]
        B, N = A.shape
        v = cp.ones((B, N), dtype=cp.float32) / N
        for _ in range(n_iter):
            Kv = K.dot(v.T).T + 1e-12               # [B,N]
            u_scal = A / Kv
            KT_u = K.T.dot(u_scal.T).T + 1e-12      # [B,N]
            v_new = b[None, :] / KT_u
            # 收敛判据
            if cp.max(cp.abs(v_new - v)).item() < tol:
                v = v_new
                break
            v = v_new
        # 成本: sum_{ij} u_i K_ij v_j C_ij
        G = K * C                                   # [N,N]
        w = G.dot(v.T)                              # [N,B]
        cost = cp.sum(u_scal.T * w, axis=0)         # [B]
        return cost

    def sinkhorn_batch_torch(A, b, C, K, eps, n_iter=200, tol=1e-6, device="cpu"):
        B, N = A.shape
        v = torch.full((B, N), 1.0 / N, dtype=torch.float32, device=device)
        for _ in range(n_iter):
            Kv = (K @ v.transpose(0,1)).transpose(0,1) + 1e-12   # [B,N]
            u_scal = A / Kv
            KT_u = (K.transpose(0,1) @ u_scal.transpose(0,1)).transpose(0,1) + 1e-12
            v_new = b.unsqueeze(0) / KT_u
            if torch.max(torch.abs(v_new - v)).item() < tol:
                v = v_new
                break
            v = v_new
        G = K * C                                                # [N,N]
        w = G @ v.transpose(0,1)                                 # [N,B]
        cost = torch.sum(u_scal.transpose(0,1) * w, dim=0)       # [B]
        return cost

    # 预备公共变量
    eps = float(reg)
    dists = np.zeros(n_genes, dtype=np.float64)

    # 将均匀分布与坐标搬到对应设备并构造 C、K
    if engine == "cupy":
        cp = __import__("cupy")
        Xg = Xn.astype(np.float32)
        C_cp, K_cp = build_cost_and_kernel_cupy(Xg, eps)
        b_cp = cp.asarray(u.astype(np.float32))
        # 预计算 OT(u,u) 供 debias
        if debias:
            cost_uu = float(sinkhorn_batch_cupy(b_cp[None, :], b_cp, C_cp, K_cp, eps)[0].get())
        # 批处理
        rng_iter = range(0, n_genes, batch_size)
        if verbose:
            from tqdm import tqdm as _tqdm
            rng_iter = _tqdm(rng_iter, total=(n_genes + batch_size - 1)//batch_size, desc="OT[cupy]")
        for s in rng_iter:
            e = min(s + batch_size, n_genes)
            a_np = P[s:e]
            valid_idx = np.where(a_np.sum(axis=1) > 1e-8)[0]
            if len(valid_idx) == 0:
                continue
            A_cp = cp.asarray(a_np[valid_idx].astype(np.float32))  # [B,N]
            cost_ab = sinkhorn_batch_cupy(A_cp, b_cp, C_cp, K_cp, eps)  # [B]
            if debias:
                # cost(a,a)
                cost_aa = sinkhorn_batch_cupy(A_cp, A_cp, C_cp, K_cp, eps)
                vals = cost_ab - 0.5 * cost_aa - 0.5 * cost_uu
                vals = vals.get()
            else:
                vals = cost_ab.get()
            dists[s:e][valid_idx] = vals.astype(np.float64)
        adata.uns["ot_engine"] = "cupy"

    else:
        # torch 引擎（GPU/CPU 自适应）
        device_t = torch.device(dev)
        Xg = Xn.astype(np.float32)
        C_t, K_t = build_cost_and_kernel_torch(Xg, eps, device_t)
        b_t = torch.tensor(u.astype(np.float32), device=device_t)
        if debias:
            cost_uu = float(sinkhorn_batch_torch(b_t[None, :], b_t, C_t, K_t, eps, device=device_t)[0].item())
        rng_iter = range(0, n_genes, batch_size)
        if verbose:
            from tqdm import tqdm as _tqdm
            rng_iter = _tqdm(rng_iter, total=(n_genes + batch_size - 1)//batch_size, desc=f"OT[torch-{dev}]")
        with torch.no_grad():
            for s in rng_iter:
                e = min(s + batch_size, n_genes)
                a_np = P[s:e]
                valid_idx = np.where(a_np.sum(axis=1) > 1e-8)[0]
                if len(valid_idx) == 0:
                    continue
                A_t = torch.tensor(a_np[valid_idx].astype(np.float32), device=device_t)  # [B,N]
                cost_ab = sinkhorn_batch_torch(A_t, b_t, C_t, K_t, eps, device=device_t)  # [B]
                if debias:
                    cost_aa = sinkhorn_batch_torch(A_t, A_t, C_t, K_t, eps, device=device_t)
                    vals = cost_ab - 0.5 * cost_aa - 0.5 * cost_uu
                else:
                    vals = cost_ab
                dists[s:e][valid_idx] = vals.detach().cpu().numpy().astype(np.float64)
        adata.uns["ot_engine"] = f"torch-{dev}"

    # 写回结果与元信息
    adata.var["OT"] = dists
    adata.uns["ot_coord_scale"] = coord_scale
    adata.uns["ot_cost_norm"] = cost_norm
    adata.uns["ot_method"] = "sinkhorn"
    adata.uns["ot_reg"] = reg
    adata.uns["ot_p"] = p
    adata.uns["ot_debias"] = bool(debias)
    return adata


prepare_for_ot = prepare_for_ot
cal_ot = cal_ot
cal_ot_gpu = cal_ot_gpu



#xb-----------------------------------------------------------------------
# ...existing code...


def prepare_for_wavelet(adata, grid_size=20, percentage=0.1, platform="visium", min_cells=3, copy=False):
    """
    第一步：为小波分析做准备（参考 prepare_for_PI 的写法）。
    - 剔除低表达基因
    - 对坐标做规整（并在需要时计算插值网格形状或 image 索引）
    - 不做归一化与滤波，仅保存进行小波分解所需的元信息

    返回:
      ad 或 adata（依据 copy），其中:
        - ad.uns['grid_size'] = grid_size
        - ad.uns['locates'] = 坐标数组
        - 若 platform 非 visium/ST: ad.uns['shape'] = (H, W)
        - 若 platform 为 visium/ST: ad.obs['image_idx_1d'] = 网格位置索引（从1开始）
        - ad.uns['wavelet_platform'] = platform
    """
    ad = adata.copy() if copy else adata

    # 1) 预处理：剔除低表达基因
    selected_gene_idxs, postcount = pre_process(ad, percentage, var_stabilization=False)

    if platform in ["visium", "ST"]:
        # 2) 坐标与网格索引
        try:
            locates = ad.obs[["array_row", "array_col"]]
            locates = locates.values if isinstance(locates, pd.DataFrame) else locates
        except Exception:
            locates = ad.obsm["spatial"]
            locates = locates.values if isinstance(locates, pd.DataFrame) else locates
        locates = locates.astype(float)
        if np.min(locates) == 0:
            locates += 1  # 转为从1开始，兼容 make_image 的用法
        _, image_idx = make_image(postcount[0], locates, platform, get_image_idx=True, grid_size=grid_size)

        # 3) 保留基因、按 min_cells 过筛
        ad = ad[:, selected_gene_idxs]
        sc.pp.filter_genes(ad, min_cells=min_cells)
        ad.obs["image_idx_1d"] = image_idx
    else:
        # 其它平台：从 obsm["spatial"] 获取坐标，并推断插值后的图像形状
        locates = ad.obsm["spatial"]
        locates = locates.values if isinstance(locates, pd.DataFrame) else locates
        locates = locates.astype(float)
        _, shape = make_image(postcount[0], locates, platform, grid_size=grid_size)
        assert shape[0] > 1 and shape[1] > 1, (
            f"Gene image size is {shape[0]} * {shape[1]} after interpolation. Please set a smaller grid size!!"
        )
        ad = ad[:, selected_gene_idxs]
        sc.pp.filter_genes(ad, min_cells=max(3, min_cells))
        ad.uns["shape"] = shape

    ad.uns["grid_size"] = grid_size
    ad.uns["locates"] = locates
    ad.uns["wavelet_platform"] = platform
    return ad


def wavelet_decompose(adata, wavelet="db2", level=None, mode="periodization",
                      platform=None, store_key="wavelet", fillna=0.0, return_arrays=False):
    """
    第二步：对每个基因的插值图像做二维离散小波分解，输出小波系数。
    - 使用 pywt.wavedec2 进行多尺度分解
    - 用 pywt.coeffs_to_array 将多尺度系数打包为单个二维数组，便于统一存储
    - 不进行自定义评分

    参数:
      wavelet: 小波基名称（如 "db2", "haar", "sym4"...）
      level: 分解层数；None 则使用最大合理层数
      mode: 边界处理模式（推荐 "periodization" 或 "symmetric"）
      platform: 若为 None，则取 adata.uns['wavelet_platform'] 或 'visium'
      store_key: 在 adata.uns 中的前缀（默认 "wavelet"）
      fillna: 将插值图中的 NaN 替换为该值（默认 0.0）
      return_arrays: 若为 True，返回 (adata, arrays, slices)

    写入:
      adata.uns[f"{store_key}_array"]  -> np.ndarray，形如 [n_genes, Hc, Wc] 的打包系数数组
      adata.uns[f"{store_key}_slices"] -> pywt.coeffs_to_array 返回的切片信息（用于还原各级子带）
      adata.uns[f"{store_key}_meta"]   -> 字典，含 wavelet/level/mode/shape 等元信息
    """
    import pywt

    plat = platform or adata.uns.get("wavelet_platform", "visium")
    locates = adata.uns["locates"]

    # 提取基因 x spot 的表达矩阵
    if sp.issparse(adata.X):
        GXS = adata.X.A.T  # [n_genes, n_spots]
    else:
        GXS = adata.X.T

    n_genes, n_spots = GXS.shape

    # 构建第一个基因的图像，以确定层数与系数打包形状
    img0, _ = make_image(GXS[0], locates, plat)
    img0 = np.nan_to_num(img0, nan=fillna)

    w = pywt.Wavelet(wavelet)
    if level is None:
        # 自动选择最大分解层
        max_level = pywt.dwt_max_level(min(img0.shape), w.dec_len)
        lev = max(1, max_level)
    else:
        lev = int(level)

    # 做一次分解，确定切片布局
    coeffs0 = pywt.wavedec2(img0, w, mode=mode, level=lev)
    arr0, slices = pywt.coeffs_to_array(coeffs0)
    Hc, Wc = arr0.shape

    arrays = np.zeros((n_genes, Hc, Wc), dtype=np.float32)

    # 对每个基因分解
    for i in range(n_genes):
        img, _ = make_image(GXS[i], locates, plat)
        img = np.nan_to_num(img, nan=fillna)
        coeffs = pywt.wavedec2(img, w, mode=mode, level=lev)
        arr, _ = pywt.coeffs_to_array(coeffs)  # 与 slices 兼容
        arrays[i] = arr.astype(np.float32, copy=False)

    # 保存结果与元信息
    adata.uns[f"{store_key}_array"] = arrays
    adata.uns[f"{store_key}_slices"] = slices
    adata.uns[f"{store_key}_meta"] = {
        "wavelet": wavelet,
        "level": lev,
        "mode": mode,
        "packed_shape": (Hc, Wc),
        "image_shape": img0.shape,
        "platform": plat,
    }

    if return_arrays:
        return adata, arrays, slices
    return adata


def visualize_wavelet_multiscale(adata, gene, wavelet=None, level=None, mode=None,
                                 platform=None, cmap="viridis", figsize=(10, 6)):
    """
    第三步：可视化单个基因在不同尺度下的小波子带（不计算评分）。
    - 左侧显示插值后的原始表达图
    - 右侧按层（1..L）分别显示 |H|、|V|、|D| 三个细节子带的幅值

    参数:
      gene: 基因索引(int)或基因名(str)
      wavelet/level/mode: 若为 None，则优先读取 wavelet_decompose 保存的 meta；仍为 None 时给出默认
      platform: 若为 None，从 adata.uns['wavelet_platform'] 取
      cmap: 显示色图
      figsize: 画布大小

    返回:
      fig, axes
    """
    import pywt
    import matplotlib.pyplot as plt

    # 解析基因索引
    if isinstance(gene, str):
        gi = int(np.where(adata.var_names == gene)[0][0])
    else:
        gi = int(gene)

    plat = platform or adata.uns.get("wavelet_platform", "visium")
    locates = adata.uns["locates"]

    # 获取表达向量与插值图
    if sp.issparse(adata.X):
        gx = adata.X[:, gi].A.flatten()
    else:
        gx = np.asarray(adata.X[:, gi]).flatten()
    img, _ = make_image(gx, locates, plat)
    img = np.nan_to_num(img, nan=0.0)

    # 从已保存的 meta 中获取参数；若无则使用默认
    meta = adata.uns.get("wavelet_meta", None)
    meta2 = adata.uns.get("wavelet_meta", None)  # 占位，兼容不同 key；下面更稳妥地读取
    saved = adata.uns.get("wavelet_meta") or adata.uns.get("wavelet_meta", {})
    saved = adata.uns.get("wavelet_meta") or adata.uns.get("wavelet_meta", {})
    # 更通用地从 wavelet_decompose 的存储中读取
    meta_pack = adata.uns.get("wavelet_meta") or adata.uns.get("wavelet_meta", {})
    meta_pack = adata.uns.get("wavelet_meta") or adata.uns.get("wavelet_meta", {})
    meta_auto = adata.uns.get("wavelet_meta")
    # 直接从 wavelet_decompose 存的 key 读取（默认 store_key="wavelet"）
    meta_used = adata.uns.get("wavelet_meta")
    if meta_used is None:
        meta_used = adata.uns.get("wavelet_meta", {})
    meta_used = adata.uns.get("wavelet_meta") or adata.uns.get("wavelet_meta", {})
    # 实际使用：从 wavelet_decompose 保存的 key 中读取
    pack_meta = adata.uns.get("wavelet_meta")
    if pack_meta is None:
        pack_meta = adata.uns.get("wavelet_meta", {})

    # 如果上面读不到，直接读 wavelet_decompose 的默认存储 key
    meta2 = adata.uns.get("wavelet_meta") or {}
    meta3 = adata.uns.get("wavelet_meta") or {}
    # 简化：优先读取 wavelet_decompose 存储的元信息
    wk = "wavelet_meta"
    default_store_key = "wavelet"
    meta_info = adata.uns.get(f"{default_store_key}_meta", {})

    wname = wavelet or meta_info.get("wavelet", "db2")
    lev = level or meta_info.get("level", None)
    md = mode or meta_info.get("mode", "periodization")

    w = pywt.Wavelet(wname)
    if lev is None:
        lev = max(1, pywt.dwt_max_level(min(img.shape), w.dec_len))

    # 小波分解
    coeffs = pywt.wavedec2(img, w, mode=md, level=lev)
    # coeffs 结构: [cA_L, (cH_L, cV_L, cD_L), ..., (cH_1, cV_1, cD_1)]

    # 作图：原图 + 各层细节
    nrows = max(2, lev + 1)  # 第一行放原图；后续每行一个 level 的子带网格
    fig = plt.figure(figsize=figsize)

    # 原图
    ax0 = plt.subplot(nrows, 4, 1)
    im0 = ax0.imshow(img, cmap=cmap)
    ax0.set_title(f"{adata.var_names[gi]}: original")
    ax0.axis("off")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

    # 最高层近似系数（可选显示）
    cA = coeffs[0]
    axA = plt.subplot(nrows, 4, 2)
    imA = axA.imshow(np.abs(cA), cmap=cmap)
    axA.set_title(f"A_{lev}")
    axA.axis("off")
    plt.colorbar(imA, ax=axA, fraction=0.046, pad=0.02)

    # 留出两格占位，使每行 4 列整齐
    ax_placeholder1 = plt.subplot(nrows, 4, 3); ax_placeholder1.axis("off")
    ax_placeholder2 = plt.subplot(nrows, 4, 4); ax_placeholder2.axis("off")

    # 各层细节 |H|, |V|, |D|
    for li in range(lev, 0, -1):
        idx_row = (lev - li + 1)  # 从第二行开始
        cH, cV, cD = coeffs[lev - li + 1]
        axH = plt.subplot(nrows, 4, idx_row * 4 + 1)
        axV = plt.subplot(nrows, 4, idx_row * 4 + 2)
        axD = plt.subplot(nrows, 4, idx_row * 4 + 3)
        imH = axH.imshow(np.abs(cH), cmap=cmap); axH.set_title(f"H_{li}"); axH.axis("off")
        imV = axV.imshow(np.abs(cV), cmap=cmap); axV.set_title(f"V_{li}"); axV.axis("off")
        imD = axD.imshow(np.abs(cD), cmap=cmap); axD.set_title(f"D_{li}"); axD.axis("off")
        plt.colorbar(imH, ax=axH, fraction=0.046, pad=0.02)
        plt.colorbar(imV, ax=axV, fraction=0.046, pad=0.02)
        plt.colorbar(imD, ax=axD, fraction=0.046, pad=0.02)
        ax_blank = plt.subplot(nrows, 4, idx_row * 4 + 4); ax_blank.axis("off")

    plt.tight_layout()
    return fig, fig.axes



def cal_wavelet_pattern_score(adata, store_key="wavelet", weight_mode="coarse", p=1.0,
                              key_added="WPS", return_details=False, eps=1e-12):
    """
    计算基于小波系数的空间模式强度评分（每基因）。
    思路：
      - 对每个基因图像做 2D DWT 后，细节系数(H,V,D)的能量反映空间变化强度；
      - 将各层细节能量占总能量的分数按权重加权求和，得到 [0,1] 间的评分。
        分数越大，空间模式越显著。

    参数:
      store_key: wavelet_decompose 保存结果的前缀（默认 "wavelet"）
      weight_mode:
        - "uniform": 各层等权
        - "coarse" : 越粗尺度权重越大（层号 l 越大权重越大）
        - "fine"   : 越细尺度权重越大（层号 l 越小权重越大）
      p: 权重幂指数，>1 更强调所选尺度，=1 为线性
      key_added: 写入 adata.var 的列名（默认 "WPS"）
      return_details: 若为 True，返回 (adata, details)；details 含每层能量占比
      eps: 数值稳定项

    写入:
      adata.var[key_added] -> 每个基因的评分（[0,1]）

    依赖:
      需先运行 wavelet_decompose，确保 adata.uns 存在:
        - f"{store_key}_array"
        - f"{store_key}_slices"
        - f"{store_key}_meta"
    """
    import numpy as np
    import pywt

    arrays = adata.uns.get(f"{store_key}_array", None)
    slices = adata.uns.get(f"{store_key}_slices", None)
    meta   = adata.uns.get(f"{store_key}_meta", {})
    if arrays is None or slices is None:
        raise ValueError(f"Missing wavelet results. Run wavelet_decompose(..., store_key='{store_key}') first.")

    n_genes = arrays.shape[0]
    # 用第一个基因恢复系数结构，确定层数 L
    coeffs0 = pywt.array_to_coeffs(arrays[0], slices, output_format="wavedec2")
    L = len(coeffs0) - 1  # 不含近似系数 cA_L 的层级数

    # 构建层权重（level 从 1..L；1=最细，L=最粗）
    levels = np.arange(1, L + 1, dtype=float)
    if weight_mode == "uniform":
        w = np.ones(L, dtype=float)
    elif weight_mode == "coarse":
        w = levels ** float(p)              # 粗尺度权重大
    elif weight_mode == "fine":
        w = (levels.max() - levels + 1) ** float(p)  # 细尺度权重大
    else:
        raise ValueError("weight_mode must be one of {'uniform','coarse','fine'}")
    w = w / (w.sum() + eps)  # 归一化权重，保持得分落在 [0,1]

    scores = np.zeros(n_genes, dtype=np.float64)
    per_level_frac = np.zeros((n_genes, L), dtype=np.float32) if return_details else None

    # 逐基因计算
    for i in range(n_genes):
        coeffs = pywt.array_to_coeffs(arrays[i], slices, output_format="wavedec2")
        # coeffs: [cA_L, (cH_L,cV_L,cD_L), ..., (cH_1,cV_1,cD_1)]
        # 计算各层细节能量，并映射到 level=1..L
        dE_levels = np.zeros(L, dtype=np.float64)
        # 总能量可直接用打包数组的平方和（等价于逐子带平方和）
        E_total = float(np.sum(arrays[i] ** 2))

        for idx, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
            lvl = L - idx + 1  # idx=1 => lvl=L(最粗)；idx=L => lvl=1(最细)
            dE = float(np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2))
            dE_levels[lvl - 1] = dE

        # 各层细节能量占总能量（避免受绝对强度影响）
        d_frac = dE_levels / (E_total + eps)  # 每层在[0,1]内
        score = float(np.sum(w * d_frac))     # 加权和，保持在[0,1]
        scores[i] = score
        if return_details:
            per_level_frac[i] = d_frac.astype(np.float32, copy=False)

    adata.var[key_added] = scores

    if return_details:
        details = {
            "levels": np.arange(1, L + 1, dtype=int),
            "weights": w,
            "per_level_fraction": per_level_frac,  # 形状 [n_genes, L]，每层细节能量占比
            "meta": meta,
            "key_added": key_added,
            "weight_mode": weight_mode,
            "p": p,
        }
        return adata, details
    return adata


cal_wavelet_pattern_score = cal_wavelet_pattern_score