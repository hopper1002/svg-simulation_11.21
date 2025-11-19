from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np











def plot_3d_gene_expression(adata, gene_list, cmap='hot', figsize=(20, 4)):
    """
    并排绘制多个基因的三维空间表达散点图

    参数:
        adata: AnnData对象
        gene_list: 基因名列表
        cmap: 颜色映射
        figsize: 画布大小
    """
    fig = plt.figure(figsize=figsize)
    for i, gene in enumerate(gene_list):
        # 获取空间坐标
        if "array_row" in adata.obs and "array_col" in adata.obs:
            coords = adata.obs[["array_row", "array_col"]].values
        else:
            coords = adata.obsm["spatial"]
        # 获取表达量
        expr = adata[:, gene].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray().flatten()
        else:
            expr = np.array(expr).flatten()
        ax = fig.add_subplot(1, len(gene_list), i+1, projection='3d')
        sc_plot = ax.scatter(coords[:,0], coords[:,1], expr, c=expr, cmap=cmap, s=20)
        ax.set_title(gene)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Expr')
        plt.colorbar(sc_plot, ax=ax, shrink=0.5)
    plt.tight_layout()
    plt.show()