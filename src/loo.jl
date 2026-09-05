# ---------------------------------------------------------------------------
# Pointwise log-likelihoods, for PSIS-LOO and blocked cross-validation.
#
# PracticalBayes already returns a `FlexiChains.SymChain`, and FlexiChains
# already ships the extensions that matter:
#
#   * `PosteriorStats.loo(chn)`                  (FlexiChainsPosteriorStatsExt)
#   * `convert_to_inference_data(chn)`           (FlexiChainsInferenceObjectsExt)
#
# So nothing here converts chains. What IS missing is the input `loo` needs: a
# `draws x observations` matrix of POINTWISE log-likelihoods. The model
# contributes its likelihood through `@addlogprob!` as one scalar per family, so
# the pointwise values have to be recomputed from the posterior draws — which is
# cheap, because the tree and tip data are fixed and only the rates change.
#
# "Observation" here is one GENE FAMILY: that is the exchangeable unit the S7
# likelihood factorises over. Leaving-one-tip-out would be a different
# (and much harder) question, since tips are coupled through the tree.
# ---------------------------------------------------------------------------

"""
    pointwise_loglik(chain, tree, tiploglik, covariate; habitat = nothing)
        -> Matrix{Float64}

Recompute the per-family log-likelihood at every posterior draw, returning a
`n_draws x n_families` matrix suitable for `PosteriorStats.loo` /
`PosteriorStats.waic`.

The model adds its likelihood via `@addlogprob!`, which collapses to one scalar
per draw, so this walks the draws and re-evaluates `felsenstein_loglik` per
family. Only the two rates change between draws — the tree and the tip
likelihoods are fixed — so each evaluation is a single pruning pass.

```julia
using PosteriorStats
ll = pointwise_loglik(chain, tree, tips, covariate)
loo(ll)
```
"""
function pointwise_loglik(chain, tree::PrunableTree, tiploglik, covariate; habitat = nothing)
    G = length(tiploglik)
    cov = collect(float.(covariate))
    hab = habitat === nothing ? zeros(G) : collect(float.(habitat))
    use_hab = habitat !== nothing

    a_λ = _draws(chain, :a_λ)
    b_λ = _draws(chain, :b_λ)
    a_μ = _draws(chain, :a_μ)
    b_μ = _draws(chain, :b_μ)
    z_λ = _draws(chain, :z_λ)
    z_μ = _draws(chain, :z_μ)
    σ²λ = _draws(chain, :σ²λ)
    σ²μ = _draws(chain, :σ²μ)
    c_λ = use_hab ? _draws(chain, :c_λ) : nothing

    n = length(a_λ)
    ll = Matrix{Float64}(undef, n, G)
    for d in 1:n
        σλ = sqrt(σ²λ[d])
        σμ = sqrt(σ²μ[d])
        for g in 1:G
            ηλ = a_λ[d] + b_λ[d] * cov[g] + σλ * z_λ[d][g]
            use_hab && (ηλ += c_λ[d] * hab[g])
            loss = exp(ηλ)
            gain = exp(a_μ[d] + b_μ[d] * cov[g] + σμ * z_μ[d][g])
            ll[d, g] = felsenstein_loglik(tree, tiploglik[g], loss, gain)
        end
    end
    ll
end

"""
    _draws(chain, name) -> Vector

Pull one parameter's draws out of a chain as a plain vector, flattening the
draw/chain dimensions. Written against `chain[name]` so it works with a
`FlexiChains.SymChain` and with the raw `Vector{NamedTuple}` you get from
`chain_type = nothing`.
"""
function _draws(chain, name::Symbol)
    if chain isa AbstractVector && !isempty(chain) && first(chain) isa NamedTuple
        return [getproperty(t, name) for t in chain]
    end
    vec(collect(chain[name]))
end

"""
    blocked_folds(group) -> Vector{Vector{Int}}

Partition observation indices into folds by a grouping label, one fold per
distinct value.

For the S7 case `group` is one label per gene family. The analysis plan requires
model comparison with **whole groups held out** rather than naive LOO, because
observations within a group are not exchangeable — in bqdiff the group is the
study a sample came from (CLAUDE.md 5.5), and study correlates with sequencing
depth, assembler and habitat all at once.

```julia
folds = blocked_folds(study_id)
elpds = [elpd_holdout(ll, f) for f in folds]
```
"""
function blocked_folds(group::AbstractVector)
    idx = Dict{Any,Vector{Int}}()
    for (i, g) in enumerate(group)
        push!(get!(idx, g, Int[]), i)
    end
    collect(values(idx))
end

"""
    elpd_holdout(ll, held) -> Float64

Expected log pointwise predictive density for the held-out observations `held`,
computed from the pointwise log-likelihood matrix `ll` (`draws x observations`)
as `Σ_i log mean_d exp(ll[d, i])`.

This is the plain held-out elpd, NOT importance-weighted: it assumes `ll` came
from a model fitted WITHOUT the held-out group. Use it for honest blocked
cross-validation, refitting per fold. For the cheap in-sample approximation use
`PosteriorStats.loo` on the full matrix instead — and note that naive LOO is
what the analysis plan explicitly rules out for the primary comparison.
"""
function elpd_holdout(ll::AbstractMatrix, held::AbstractVector{Int})
    total = 0.0
    n = size(ll, 1)
    for i in held
        m = maximum(@view ll[:, i])
        s = 0.0
        for d in 1:n
            s += exp(ll[d, i] - m)
        end
        total += m + log(s / n)
    end
    total
end
