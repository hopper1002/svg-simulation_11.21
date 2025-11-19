import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from scipy.stats import multivariate_normal

# 新增：稳定的非负链接函数（softplus），确保表达值 >= 0
def _softplus(x):
    abs_x = np.abs(x)
    return np.log1p(np.exp(-abs_x)) + np.maximum(x, 0)

def generate_circle_coords(n, radius=10, center=(0, 0)):
    """生成均匀分布在圆形内部的空间坐标"""
    r = radius * np.sqrt(np.random.uniform(0, 1, n))  # 半径均匀分布
    theta = np.random.uniform(0, 2 * np.pi, n)        # 角度均匀分布
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return pd.DataFrame({'x': x, 'y': y})

def simulate_data(n=2000, m=50, r=0.9, tau=0.2, tau1=0.5, tau2=0.1, tau3=0.05, tau4=0.01, sig_sq=0.25, kern_para=5, mean_exp=0, cov_st="CS"):
    """
    生成50个基因的模拟数据：
    1-10:  indept1-10        独立表达（与原先一致）
    11-20: correlated1-10    仅共表达（与原先一致）
    21-25: stripe1-5         过中心的斜线，距线越近表达越高（线性衰减）
    26-30: radial1-5         单点热点，距点越近表达越高（高斯衰减）
    31-35: linperiod1-5      线性周期（三角波）
    36-40: nonlinperiod1-5   非线性周期（正弦）
    41-45: multipk1-5        2-3个热点之和（高斯衰减）
    46-50: ring1-5           2-3个圆环之和，距环越近表达越高
    """
    # 要求固定为50个基因
    if m != 50:
        raise ValueError("本版本固定生成50个基因，请设置 m=50")

    # 1. 生成空间坐标（圆盘内均匀）
    coord_df = generate_circle_coords(n)
    coord_df['barcode'] = [f"spot_{i}" for i in range(n)]
    coord_df.set_index('barcode', inplace=True)
    coords = coord_df[['x', 'y']].values
    rng = np.random.default_rng()

    # 2. d11 与 d1（各10个基因，保持与原来一致）
    m0 = 10
    # d11: 独立表达（潜在高斯）→ softplus 非负
    d11_latent = np.random.normal(mean_exp, 1, size=(n, m0))
    d11 = _softplus(d11_latent)

    # 共表达协方差
    if cov_st == "CS":
        R1 = np.full((m0, m0), r); np.fill_diagonal(R1, 1)
    elif cov_st == "AR1":
        R1 = np.array([[r ** abs(i - j) for j in range(m0)] for i in range(m0)])
    else:
        raise ValueError("Select AR1 or CS covariance structure")

    # d1: 仅共表达（潜在多元高斯）→ softplus 非负
    sigma1 = R1 + tau1 * np.eye(m0)
    d1_latent = multivariate_normal.rvs(mean=np.full(m0, mean_exp), cov=sigma1, size=n)
    d1 = _softplus(d1_latent)




    # 3. 各种空间模式的辅助函数
    def dist_to_line_through_origin(points, theta):
        # 线过原点，方向向量 (cosθ, sinθ)，点到线的垂直距离公式 |x sinθ - y cosθ|
        c, s = np.cos(theta), np.sin(theta)
        return np.abs(points[:, 0]*s - points[:, 1]*c)

    def gaussian_decay(d2, sigma):
        # d2: 距离的平方
        return np.exp(-d2/(2.0*sigma**2))

    def triangle_wave(t, period, phase=0.0):
        # 线性周期（三角波），范围 [0,1]
        x = (t/period + phase) % 1.0
        return 1.0 - 2.0*np.abs(x - 0.5)

    def sine_wave(t, period, phase=0.0):
        # 非线性周期（正弦），范围 [0,1]
        return 0.5*(1.0 + np.sin(2.0*np.pi*(t/period + phase)))

    # 估计尺度（用于宽度/衰减）
    radius_est = np.sqrt((coords**2).sum(axis=1)).max()  # 约为生成半径
    # 4. 21-25: 过中心斜线，距线越近表达越高（线性衰减）
    stripe = np.zeros((n, 5))
    for g in range(5):
        theta = rng.uniform(0, np.pi)  # 任意斜率（过中心）
        width = rng.uniform(0.8, 1.2) * (0.25*radius_est)  # 线宽控制
        d = dist_to_line_through_origin(coords, theta)
        val = np.clip(1.0 - d/width, 0.0, None)
        val += rng.normal(0, 0.05, size=n)  # 轻微噪声
        stripe[:, g] = val
    stripe = _softplus(stripe)

    # 5. 26-30: 单点热点，高斯衰减
    radial = np.zeros((n, 5))
    for g in range(5):
        center = coords[rng.integers(0, n)]
        d2 = ((coords - center)**2).sum(axis=1)
        sigma_r = rng.uniform(0.15, 0.3) * radius_est
        val = gaussian_decay(d2, sigma_r)
        val += rng.normal(0, 0.05, size=n)
        radial[:, g] = val
    radial = _softplus(radial)

    # 6. 31-35: 线性周期（三角波）
    linperiod = np.zeros((n, 5))
    for g in range(5):
        theta = rng.uniform(0, np.pi)   # 沿某方向投影
        c, s = np.cos(theta), np.sin(theta)
        proj = coords[:, 0]*c + coords[:, 1]*s
        period = rng.uniform(0.6, 1.0) * (0.8*radius_est)
        phase = rng.uniform(0, 1)
        val = triangle_wave(proj, period, phase)  # [0,1]
        val += rng.normal(0, 0.05, size=n)
        linperiod[:, g] = val
    linperiod = _softplus(linperiod)

    # 7. 36-40: 非线性周期（正弦）
    nonlinperiod = np.zeros((n, 5))
    for g in range(5):
        theta = rng.uniform(0, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        proj = coords[:, 0]*c + coords[:, 1]*s
        period = rng.uniform(0.5, 0.9) * (0.8*radius_est)
        phase = rng.uniform(0, 1)
        val = sine_wave(proj, period, phase)  # [0,1]
        # 可添加二维干涉条纹
        if rng.random() < 0.5:
            theta2 = (theta + rng.uniform(0.2, 1.0)) % np.pi
            c2, s2 = np.cos(theta2), np.sin(theta2)
            proj2 = coords[:, 0]*c2 + coords[:, 1]*s2
            period2 = rng.uniform(0.5, 0.9) * (0.8*radius_est)
            phase2 = rng.uniform(0, 1)
            val = 0.6*val + 0.4*sine_wave(proj2, period2, phase2)
        val += rng.normal(0, 0.05, size=n)
        nonlinperiod[:, g] = val
    nonlinperiod = _softplus(nonlinperiod)

    # 8. 41-45: 2-3个热点之和（高斯衰减）
    multipk = np.zeros((n, 5))
    for g in range(5):
        k = rng.integers(2, 4)  # 2或3个热点
        val = np.zeros(n)
        for _ in range(k):
            center = coords[rng.integers(0, n)]
            sigma_r = rng.uniform(0.12, 0.25) * radius_est
            d2 = ((coords - center)**2).sum(axis=1)
            val += gaussian_decay(d2, sigma_r)
        val /= k
        val += rng.normal(0, 0.05, size=n)
        multipk[:, g] = val
    multipk = _softplus(multipk)

    # 9. 46-50: 环状表达（2-3个圆环之和）
    ring = np.zeros((n, 5))
    for g in range(5):
        k = rng.integers(2, 4)  # 2或3个圆环
        val = np.zeros(n)
        for _ in range(k):
            center = coords[rng.integers(0, n)]
            r0 = rng.uniform(0.25, 0.85) * radius_est
            band = rng.uniform(0.06, 0.12) * radius_est
            d = np.linalg.norm(coords - center, axis=1)
            val += np.exp(-((np.abs(d - r0))**2) / (2.0*band**2))
        val /= k
        val += rng.normal(0, 0.05, size=n)
        ring[:, g] = val
    ring = _softplus(ring)

    # 10. 合并所有数据（n x 50）
    data = np.hstack([d11, d1, stripe, radial, linperiod, nonlinperiod, multipk, ring])

    # 基因名
    genes = (
        [f"indept{i+1}" for i in range(10)] +           # 1-10
        [f"correlated{i+1}" for i in range(10)] +       # 11-20
        [f"stripe_{i+1}" for i in range(5)] +           # 21-25
        [f"radial_{i+1}" for i in range(5)] +           # 26-30
        [f"linperiod_{i+1}" for i in range(5)] +        # 31-35
        [f"nonlinperiod_{i+1}" for i in range(5)] +     # 36-40
        [f"multipk_{i+1}" for i in range(5)] +          # 41-45
        [f"ring_{i+1}" for i in range(5)]               # 46-50
    )

    # 构建 AnnData 对象
    adata = anndata.AnnData(
        X=data,
        obs=coord_df,
        var=pd.DataFrame(index=genes)
    )
    adata.obsm['spatial'] = adata.obs[['x', 'y']].values

    # 保存
    adata.write_h5ad("simulated_visium_data.h5ad")
    print("模拟数据已保存为 simulated_visium_data.h5ad")


if __name__ == "__main__":
    simulate_data(n=2000, m=10)