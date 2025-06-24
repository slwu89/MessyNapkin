# lnam

The Linear Network Autocorrelation Model is:

$$ y = W_{1} y + X \beta + \epsilon $$

$$ \epsilon = W_{2} \epsilon + \eta $$

where $\eta \sim N(0,\sigma^2)$ and the matrices $W_{1}$ and $W_{2}$ include the "AR" and "MA" parameters:

$W_{1} = \sum_{i=1}^{p} \rho_{1,i} W_{1,i}$ and $W_{2} = \sum_{i=1}^{q} \rho_{2,i}W_{2,i}$

  * the code from sna: https://github.com/cran/sna/blob/master/R/models.R#L942
  * description for easy reading: https://rdrr.io/cran/sna/man/lnam.html
  * my comparison with purely spatial regression approaches https://gist.github.com/slwu89/f4ed5a57f5490ec52648bbd687a6a3e7#file-netreg-r

In the spatial/economics literature, it seems that this model is called the "spatial autoregressive combined" model, and some references which specifically refer to it as such are at:

  * https://journals.sagepub.com/doi/10.1177/0049124119882467
  * https://link.springer.com/article/10.1007/s43071-022-00023-w

To-do:

  * basic version (direct translation of R)
  * JuMP version for MLE
  * RXInfer https://rxinfer.ml/
  * Turing https://turinglang.org/stable/

Other:

  * Static arrays as much as possible. Not sure if we can do static sparse matrices, but that would be a huge speed improvement.
  * Lots of computation can probably be cached during each iteration (Xb, "aggregated Ws", etc)
  * Investigate alternative autodiff backends
  * See: https://discourse.julialang.org/t/sparse-matrix-error-with-forward-diff/108170 autodiff with logabsdet and sparse matrix input fails
  * faster linaglg: https://github.com/JuliaLinearAlgebra/MKL.jl, https://github.com/JuliaLinearAlgebra/Octavian.jl
  
