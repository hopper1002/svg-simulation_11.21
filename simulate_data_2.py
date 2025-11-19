import numpy as np
import pandas as pd
import anndata
import scanpy as sc

def generate_circle_coords(n, radius=10, center=(0, 0)):
    """生成均匀分布在圆形内部的空间坐标"""
    r = radius * np.sqrt(np.random.uniform(0, 1, n))  # 半径均匀分布
    theta = np.random.uniform(0, 2 * np.pi, n)        # 角度均匀分布
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return pd.DataFrame({'x': x, 'y': y})

def _nb_rvs(mu, theta, rng):
    """
    负二项采样：Poisson-Gamma 混合实现。
    E[Y]=mu, Var[Y]=mu + mu^2/theta
    支持mu为向量或矩阵，theta可为标量或同形数组。
    """
    mu = np.asarray(mu, dtype=float)
    theta_arr = np.broadcast_to(theta, mu.shape).astype(float)
    # Gamma(shape=k, scale=theta): mean = k*scale
    lam = rng.gamma(shape=np.maximum(theta_arr, 1e-8),
                    scale=np.maximum(mu, 1e-12) / np.maximum(theta_arr, 1e-8))
    return rng.poisson(lam)

def simulate_data(n=2000, m=50, r=0.9, tau=0.2, tau1=0.5, tau2=0.1, tau3=0.05, tau4=0.01, sig_sq=0.25, kern_para=5, mean_exp=0, cov_st="CS"):
    """
    生成50个基因的模拟数据：
    1-10:  indept1-10        独立表达（无空间）
    11-20: correlated1-10    仅共表达（spot级共享因子产生相关）
    21-25: stripe_1-5        过中心的条带，距线越近表达越高
    26-30: radial_1-5        单点热点，高斯衰减
    31-35: periodic_stripe_1-5     周期性条纹（三角波）
    36-40: nonlinear_superimposed_1-5  非线性叠加（正弦/干涉）
    41-45: multipk_1-5       2-3个热点的叠加
    46-50: ring_1-5          2-3个环的叠加
    以上全部用负二项分布生成计数数据。
    """
    if m != 50:
        raise ValueError("本版本固定生成50个基因，请设置 m=50")

    # 1) 坐标
    coord_df = generate_circle_coords(n)
    coord_df['barcode'] = [f"spot_{i}" for i in range(n)]
    coord_df.set_index('barcode', inplace=True)
    coords = coord_df[['x', 'y']].values
    rng = np.random.default_rng()

    # 一些小工具（空间模式）
    def dist_to_line_through_origin(points, theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.abs(points[:, 0]*s - points[:, 1]*c)

    def gaussian_decay(d2, sigma):
        return np.exp(-d2/(2.0*sigma**2))

    def triangle_wave(t, period, phase=0.0):
        x = (t/period + phase) % 1.0
        return 1.0 - 2.0*np.abs(x - 0.5)  # [0,1]

    def sine_wave(t, period, phase=0.0):
        return 0.5*(1.0 + np.sin(2.0*np.pi*(t/period + phase)))  # [0,1]

    radius_est = np.sqrt((coords**2).sum(axis=1)).max()

    # ---------------------------
    # 2) 1-10: 独立表达（无空间）
    # ---------------------------
    indept = np.zeros((n, 10), dtype=int)
    # 每个基因一个基线均值；每个spot/基因一个独立的（轻微）乘性噪声
    indept_base = rng.uniform(0.5, 3.0, size=10)           # 基线均值
    indept_theta = rng.uniform(6.0, 12.0, size=10)         # 离散度（越大越接近Poisson）
    for j in range(10):
        eta = rng.gamma(shape=10.0, scale=1/10.0, size=n)  # 期望≈1，方差小
        mu = indept_base[j] * eta
        indept[:, j] = _nb_rvs(mu, indept_theta[j], rng)

    # ---------------------------------------------
    # 3) 11-20: 仅共表达（spot级共享因子产生相关）
    # ---------------------------------------------
    correlated = np.zeros((n, 10), dtype=int)
    corr_base = rng.uniform(0.8, 4.0, size=10)
    corr_theta = rng.uniform(6.0, 12.0, size=10)
    g_shared = rng.gamma(shape=2.5, scale=1/2.5, size=n)   # 共享因子（均值≈1，方差较大→相关性更明显）
    for j in range(10):
        eps = rng.gamma(shape=60.0, scale=1/60.0, size=n)  # 轻微基因内噪声
        mu = corr_base[j] * g_shared * eps
        correlated[:, j] = _nb_rvs(mu, corr_theta[j], rng)

    # ---------------------------------------------
    # 4) 21-25: 条纹（距线越近表达越高）
    # ---------------------------------------------
    stripe = np.zeros((n, 5), dtype=int)
    stripe_theta = rng.uniform(5.0, 10.0, size=5)
    for g in range(5):
        theta = rng.uniform(0, np.pi)
        width = rng.uniform(0.8, 1.2) * (0.25*radius_est)
        d = dist_to_line_through_origin(coords, theta)
        val = np.clip(1.0 - d/width, 0.0, 1.0)            # [0,1]
        base = rng.uniform(0.2, 1.0)
        amp = rng.uniform(6.0, 12.0)
        eta = rng.gamma(shape=25.0, scale=1/25.0, size=n)  # 轻微乘性噪声
        mu = (base + amp*val) * eta
        stripe[:, g] = _nb_rvs(mu, stripe_theta[g], rng)

    # ---------------------------------------------
    # 5) 26-30: 单点热点（高斯衰减）
    # ---------------------------------------------
    radial = np.zeros((n, 5), dtype=int)
    radial_theta = rng.uniform(5.0, 10.0, size=5)
    for g in range(5):
        center = coords[rng.integers(0, n)]
        d2 = ((coords - center)**2).sum(axis=1)
        sigma_r = rng.uniform(0.15, 0.3) * radius_est
        val = gaussian_decay(d2, sigma_r)                  # [0,1]
        base = rng.uniform(0.2, 1.0)
        amp = rng.uniform(6.0, 12.0)
        eta = rng.gamma(shape=25.0, scale=1/25.0, size=n)
        mu = (base + amp*val) * eta
        radial[:, g] = _nb_rvs(mu, radial_theta[g], rng)

    # ---------------------------------------------
    # 6) 31-35: 周期性条纹（三角波）
    # ---------------------------------------------
    periodic_stripe = np.zeros((n, 5), dtype=int)
    periodic_stripe_theta = rng.uniform(5.0, 10.0, size=5)
    for g in range(5):
        theta = rng.uniform(0, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        proj = coords[:, 0]*c + coords[:, 1]*s
        period = rng.uniform(0.6, 1.0) * (0.8*radius_est)
        phase = rng.uniform(0, 1)
        val = triangle_wave(proj, period, phase)           # [0,1]
        base = rng.uniform(0.2, 1.0)
        amp = rng.uniform(6.0, 12.0)
        eta = rng.gamma(shape=25.0, scale=1/25.0, size=n)
        mu = (base + amp*val) * eta
        periodic_stripe[:, g] = _nb_rvs(mu, periodic_stripe_theta[g], rng)

    # ---------------------------------------------
    # 7) 36-40: 非线性周期（正弦，含可选干涉）
    # ---------------------------------------------
    # 7) 36-40: 非线性叠加（正弦，含可选干涉）
    # ---------------------------------------------
    nonlinear_superimposed = np.zeros((n, 5), dtype=int)
    nonlinear_superimposed_theta = rng.uniform(5.0, 10.0, size=5)
    for g in range(5):
        theta = rng.uniform(0, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        proj = coords[:, 0]*c + coords[:, 1]*s
        period = rng.uniform(0.5, 0.9) * (0.8*radius_est)
        phase = rng.uniform(0, 1)
        val = sine_wave(proj, period, phase)               # [0,1]
        if rng.random() < 0.5:
            theta2 = (theta + rng.uniform(0.2, 1.0)) % np.pi
            c2, s2 = np.cos(theta2), np.sin(theta2)
            proj2 = coords[:, 0]*c2 + coords[:, 1]*s2
            period2 = rng.uniform(0.5, 0.9) * (0.8*radius_est)
            phase2 = rng.uniform(0, 1)
            val = 0.6*val + 0.4*sine_wave(proj2, period2, phase2)
        val = np.clip(val, 0.0, 1.0)
        base = rng.uniform(0.2, 1.0)
        amp = rng.uniform(6.0, 12.0)
        eta = rng.gamma(shape=25.0, scale=1/25.0, size=n)
        mu = (base + amp*val) * eta
        nonlinear_superimposed[:, g] = _nb_rvs(mu, nonlinear_superimposed_theta[g], rng)

    # ---------------------------------------------
    # 8) 41-45: 多热点（高斯叠加）
    # ---------------------------------------------
    multipk = np.zeros((n, 5), dtype=int)
    multipk_theta = rng.uniform(5.0, 10.0, size=5)
    for g in range(5):
        k = rng.integers(2, 4)
        val = np.zeros(n)
        for _ in range(k):
            center = coords[rng.integers(0, n)]
            sigma_r = rng.uniform(0.12, 0.25) * radius_est
            d2 = ((coords - center)**2).sum(axis=1)
            val += gaussian_decay(d2, sigma_r)
        val = np.clip(val / k, 0.0, 1.0)
        base = rng.uniform(0.2, 1.0)
        amp = rng.uniform(6.0, 12.0)
        eta = rng.gamma(shape=25.0, scale=1/25.0, size=n)
        mu = (base + amp*val) * eta
        multipk[:, g] = _nb_rvs(mu, multipk_theta[g], rng)

    # ---------------------------------------------
    # 9) 46-50: 环状（环带叠加）
    # ---------------------------------------------
    ring = np.zeros((n, 5), dtype=int)
    ring_theta = rng.uniform(5.0, 10.0, size=5)
    for g in range(5):
        k = rng.integers(2, 4)
        val = np.zeros(n)
        for _ in range(k):
            center = coords[rng.integers(0, n)]
            r0 = rng.uniform(0.25, 0.85) * radius_est
            band = rng.uniform(0.06, 0.12) * radius_est
            d = np.linalg.norm(coords - center, axis=1)
            val += np.exp(-((np.abs(d - r0))**2) / (2.0*band**2))
        val = np.clip(val / k, 0.0, 1.0)
        base = rng.uniform(0.2, 1.0)
        amp = rng.uniform(6.0, 12.0)
        eta = rng.gamma(shape=25.0, scale=1/25.0, size=n)
        mu = (base + amp*val) * eta
        ring[:, g] = _nb_rvs(mu, ring_theta[g], rng)

    # 10) 合并与输出
    data = np.hstack([
        indept, correlated, stripe, radial,
        periodic_stripe, nonlinear_superimposed,
        multipk, ring
    ]).astype(np.int32)

    genes = (
        [f"indept{i+1}" for i in range(10)] +           # 1-10
        [f"correlated{i+1}" for i in range(10)] +       # 11-20
        [f"stripe_{i+1}" for i in range(5)] +           # 21-25
        [f"radial_{i+1}" for i in range(5)] +           # 26-30
        [f"periodic_stripe_{i+1}" for i in range(5)] +  # 31-35
        [f"nonlinear_superimposed_{i+1}" for i in range(5)] + # 36-40
        [f"multipk_{i+1}" for i in range(5)] +          # 41-45
        [f"ring_{i+1}" for i in range(5)]               # 46-50
    )

    adata = anndata.AnnData(
        X=data,
        obs=coord_df,
        var=pd.DataFrame(index=genes)
    )
    adata.obsm['spatial'] = adata.obs[['x', 'y']].values

    adata.write_h5ad("simulated_visium_data.h5ad")
    print("模拟数据已保存为 simulated_visium_data.h5ad（负二项计数版）")

if __name__ == "__main__":
    simulate_data(n=2000, m=50)