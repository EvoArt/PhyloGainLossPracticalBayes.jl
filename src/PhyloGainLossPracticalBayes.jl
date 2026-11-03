module PhyloGainLossPracticalBayes

# The glue between PhyloGainLoss.jl (PPL-agnostic two-state CTMC likelihood,
# tree type and tip observation model) and PracticalBayes.jl. PhyloGainLoss
# deliberately knows nothing about any PPL; this package supplies the small,
# repetitive PracticalBayes-shaped pieces you would otherwise hand-write per
# model to fit one of those likelihoods:
#
#   * the hierarchical gain-loss model over many gene families, with per-family
#     rates regressed on a family-level covariate (gainloss.jl),
#   * a phylogenetic `RelatednessStructure` constructor, so a Brownian/OU tree
#     effect reuses the same cached-K-inverse machinery the animal model uses
#     for pedigree A and genomic G (relatedness.jl),
#   * the conjugate variance-component kernel for the partial-pooling scales.
#
# The design mirrors PracticalEpiBayes' own rule: nothing here constrains the
# general case. Each builder returns `(model, spl)` ready for
# `AbstractMCMC.sample`, and a user who wants a bespoke blocking still writes
# the `@model` and the `Gibbs(...)` by hand exactly as they would without this
# package.
#
# On what is and is not a latent block here: unlike an epidemic trajectory, the
# ancestral states are MARGINALISED by Felsenstein pruning rather than sampled,
# so there is no discrete block for an `AbstractGibbsKernel` to update. The
# rate parameters are continuous and go through NUTS; only the variance
# components have closed-form conditionals worth a kernel. `marginal_ancestral`
# recovers ancestral states after fitting, as a posterior summary.

using PracticalBayes
using PracticalBayes: AbstractGibbsKernel, ModelConditional, gibbs_step
using PhyloGainLoss: PhyloGainLoss, felsenstein_loglik, PrunableTree, nnodes
using Distributions
using LinearAlgebra
using Random: AbstractRNG
using ForwardDiff
using DiffResults

include("relatedness.jl")
include("varkernel.jl")
include("gainloss.jl")
include("loo.jl")
include("blockgrad.jl")

# Phylogenetic relatedness, in the same shape the genomics package uses
export PhyloRelatedness, phylo_relatedness, brownian_covariance

# Conjugate scale kernel for partial-pooling standard deviations
export InvGammaVarKernel

# Model builders: each returns (model, spl)
export gainloss_model, gainloss_pooled_model

# Pointwise log-likelihoods for PSIS-LOO / blocked CV. Chain conversion itself
# needs nothing from us: FlexiChains ships the InferenceObjects and
# PosteriorStats extensions, so `PosteriorStats.loo(chn)` and
# `convert_to_inference_data(chn)` already work on what PracticalBayes returns.
export pointwise_loglik, blocked_folds, elpd_holdout

# Blockwise-gradient posterior: O(G) instead of O(G^2). See docs/08_performance.md.
export GainLossPosterior, check_blockgrad

end # module PhyloGainLossPracticalBayes
