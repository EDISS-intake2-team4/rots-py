from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_rots(rots_res, fdr=0.05, type=None):  
  # Check for plot type
  if type is not None:
    if type not in ["volcano", "heatmap", "ma", "reproducibility", "pvalue", "pca"]:
      raise ValueError("Plot type not available. The options are: 'volcano', 'heatmap', 'ma', 'reproducibility', 'pvalue', 'pca'")
  else:
    raise ValueError("Plot type not selected. The options are: 'volcano', 'heatmap', 'ma', 'reproducibility', 'pvalue', 'pca'")
  
  # Differentially expressed features
  de = np.where(rots_res["FDR"] < fdr)[0]

  # Volcano plot
  if type == "volcano":    
    plt.scatter(rots_res["logfc"], -np.log10(rots_res["p"]), color="black", s=1)
    plt.scatter(rots_res["logfc"][de], -np.log10(rots_res["p"][de]), color="red", s=1)
    plt.xlabel("log2(fold change)")
    plt.ylabel("-log10(p-value)")
    plt.show()

  # Heatmap
  if type == "heatmap":    
    sns.heatmap(rots_res["data"].iloc[de, :], cmap="RdBu_r", center=0)
    plt.show()
  
  # MA plot
  if type == "ma":    
    plt.scatter(rots_res["data"].mean(axis=1)*0.5, rots_res["logfc"], color="black", s=1)
    plt.scatter(rots_res["data"].mean(axis=1).iloc[de], rots_res["logfc"][de], color="red", s=1)
    plt.xlabel("Mean")
    plt.ylabel("log2(fold change)")
    plt.show()

  # Reproducibility plot
  if type == "reproducibility":    
    z = rots_res["ztable"][rots_res["ztable"].index == f'{rots_res["a1"]:.2f}']
    k = rots_res["ztable"].columns.astype(int)
    plt.scatter(k, z, color="black", s=1)
    plt.scatter(k[np.where(z == np.max(z.values))[1]], z.iloc[0, np.where(z==np.max(z.values))[1][0]], color="red", s=1)
    plt.xlabel("Top list size")
    plt.ylabel("Reproducibility Z-score")
    plt.show()
  
  # Histogram of p-values
  if type == "pvalue":    
    plt.hist(rots_res["p"], bins=50)
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.show()

  # PCA plot
  if type == "pca":
    if len(de) > 0:
        pca = PCA(n_components=2)
        dt = rots_res["data"].iloc[de, :].fillna(0)
        X = pca.fit_transform(dt.T)
        plt.scatter(X[:,0], X[:,1], c=rots_res['cl'], s=1)
        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        plt.show()



