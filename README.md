# PhyloGainLossPracticalBayes.jl

The glue between [PhyloGainLoss.jl](../PhyloGainLoss) (PPL-agnostic two-state CTMC likelihood)
and [PracticalBayes.jl](../PracticalBayes).

Same layering as `EpidemicTrajectories` + `PracticalEpiBayes`: the computational package knows
nothing about any PPL; this package supplies the small, repetitive PracticalBayes-shaped
pieces you would otherwise hand-write per model.

## Plug and play

```julia
using PhyloGainLoss, PhyloGainLossPracticalBayes, AbstractMCMC

tree = from_newick(readnw(read("species.nwk", String)))

# one tip-likelihood matrix per gene family, carrying the detection model
tips = [detection_tiploglik(calls[g], p_detect[g]) for g in 1:G]

model, spl = gainloss_model(tree, tips, covariate)      # covariate: one value per family
chain = sample(model, spl, 2000)
```

`gainloss_model` returns `(model, spl)` ready for `AbstractMCMC.sample`, matching
`PracticalBayesianGenomics.animal_model`'s convention.

Start with the null to check the wiring before fitting the hierarchy:

```julia
nullmodel, nullspl = gainloss_pooled_model(tree, tips)   # one shared rate pair
```

## What the blocking does

```julia
Gibbs(
    (:a_λ, :b_λ, :c_λ, :a_μ, :b_μ) => NUTS(0.8),   # rate coefficients
    (:z_λ, :z_μ)                   => NUTS(0.8),   # non-centred family effects
    :σ²λ                           => InvGammaVarKernel(...),   # conjugate, exact
    :σ²μ                           => InvGammaVarKernel(...),
)
```

The variance components have closed-form Inverse-Gamma conditionals, so they are drawn exactly
rather than costing an AD gradient and an HMC sub-chain each — the same win
`PracticalBayesianGenomics`' `variance_kernels.jl` documents.

**There is no latent Gibbs block.** Unlike an epidemic trajectory, the ancestral states are
*marginalised* by Felsenstein pruning rather than sampled, so nothing discrete remains for an
`AbstractGibbsKernel` to update. Recover ancestral states after fitting with
`PhyloGainLoss.marginal_ancestral`.

## Phylogenetic random effects

A phylogenetic random effect is a relatedness structure by another name — under Brownian
motion, `cov(i,j)` is the shared root-to-MRCA path length, playing the role the numerator
relationship matrix `A` plays in an animal model.

```julia
rel = phylo_relatedness(tree)              # cached K⁻¹, never re-inverted
rel = phylo_relatedness(tree; lambda = 0)  # Pagel's λ = 0: no phylogenetic structure (the null)
```

`lambda` scales off-diagonals toward a star phylogeny; `jitter` ridges the diagonal, which is
required because a pruned backbone with zero-length branches gives a singular covariance —
report the jitter used, since a large one quietly weakens the effect.

## Non-centred parameterisation

Family effects are `σ · z` with `z ~ Normal(0,1)`, not `ε ~ Normal(0, σ)`. The centred form
funnels badly when `σ` is small — precisely the "families are exchangeable" case these models
sit near — and the non-centred form is also what makes the variance conditional conjugate.

If you need an `Exponential` prior on the standard deviation instead, write the `@model` by
hand and put `σ` in a NUTS block. These builders are a convenience, not a constraint.

## Diagnostics and model comparison

PracticalBayes returns a `FlexiChains.SymChain`, and FlexiChains already ships the
`PosteriorStats` and `InferenceObjects` extensions — so these work with no conversion code:

```julia
using PosteriorStats, InferenceObjects
loo(chain)                          # PSIS-LOO straight off the chain
convert_to_inference_data(chain)    # ArviZ's InferenceData
```

What this package adds is the input `loo` needs. The model contributes its likelihood via
`@addlogprob!`, which collapses to one scalar per draw, so the pointwise values are recomputed:

```julia
ll = pointwise_loglik(chain, tree, tips, covariate)   # draws x families
loo(ll)
```

The exchangeable unit is the **gene family** — tips are coupled through the tree, so
leave-one-tip-out is a different question and is not what this computes.

For blocked cross-validation (whole groups held out, refitting per fold):

```julia
for held in blocked_folds(study_id)
    # refit without `held`, then:
    elpd_holdout(pointwise_loglik(chain_wo, tree, tips, cov), held)
end
```

`elpd_holdout` is log-sum-exp throughout: a naive `log(mean(exp(.)))` underflows to `-Inf`
at realistic log-likelihood magnitudes.

## Tests

```julia
using Pkg; Pkg.test("PhyloGainLossPracticalBayes")
```

Covers the covariance construction (checked against hand-computed MRCA depths and an
independent ancestor walk), `Kinv * K ≈ I`, the conjugate kernel's draws against the analytic
Inverse-Gamma conditional, and end-to-end recovery of a known covariate effect from simulated
data.
