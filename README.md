# rots-py

Python implementation of the __Reproducibility-Optimized Test Statistic (ROTS)__ for gene ranking from the [Bioconductor ROTS](https://www.bioconductor.org/packages/release/bioc/html/ROTS.html) package.

ROTS belongs to a familly of gene ranking statistics that aim to rank genes based on evidence for differential expression in two-group comparisons. ROTS is a non-parametric method that uses a permutation test to assess the significance of the observed differential expression. ROTS is designed to be robust to outliers and to be reproducible across different studies.

__NOTE__: Data should have at least two non-null values per row for both groups.

# Installation 
```
pip install rots-py
```

# Usage
```python  
import rots

# Load data
data = ...
group = ...

# Run ROTS
result = rots.rots(data, group, B=500, log=True, verbose=True, progress=True)

# Get the ranking
ranking_statistic = result["d"]
fdr = result["fdr"]
logFC = result["logfc"]
pvalue = result["p"]
```

# Methods
## rots
Runs the ROTS analysis on the given data. Returns a Python dictionary.

## Parameters
- `data`: A pandas dataframe with genes/proteins as rows and samples as columns. (required)
- `group`: A pandas series with the group labels for each sample. (required)
- `B`: Number of permutations to perform. Default is 500. (optional)
- `K`: Top-list size. (optional)
- `paried`: Whether the samples are paired. Default is False. (optional)
- `seed`: Seed for the random number generator. Default is None. (optional)
- `a1`: Parameter for the ROTS statistic. If both a1 and a2 are specified optimization step is skipped. (optional)
- `a2`: Parameter for the ROTS statistic. If both a1 and a2 are specified optimization step is skipped. (optional)
- `log`: Whether data is log-transformed. Default is False. (optional)
- `progress`: Whether to show a progress bar. Default is False. (optional)
- `verbose`: Whether to print the progress of the analysis. Default is False. (optional)

## Returns
Python `dict` object with the following keys:
- `data`: The original dataframe used for the input
- `B`: Number of permutations
- `d`: ROTS test statistic for each gene/protein
- `logfc`: Log2 fold change
- `p`: P-value
- `FDR`: False Detection Rate
- `a1`: Optimized parameter a1
- `a2`: Optimized parameter a2
- `k`: Top list size (*`None` if optimization skipped*)
- `R`: Reproducibility score (*`None` if optimization skipped*)
- `Z`: Z-score (*`None` if optimization skipped*)
- `ztable`: Z-score table
- `cl`: Group labels for each sample

# Acknowledgements
This package was developed as part of the [EDISS](https://www.master-ediss.eu/) program in collaboration with Coffey Lab at the [Turku Bioscience](https://bioscience.fi/) center.

# Changelog
## 1.1.0
- Ported parts of code to Cython for better performance
- Fixed bugs

## 1.0.3
- Bug fixes

## 1.0.2
- Bug fixes 
- Added numba for better performance

## 1.0.0
- Initial release

