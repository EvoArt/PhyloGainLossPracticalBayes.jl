# ---------------------------------------------------------------------------
# The hierarchical gain-loss model, as a one-call builder returning
# `(model, spl)` — the same interface shape as
# `ParcticalBayesianGenomics.animal_model` and `PracticalEpiBayes`' blocks.
#
# Blocking rationale (this is the whole reason the package exists):
#
#   * the rate coefficients (a, b, c) and the non-centred family effects z are
#     CONTINUOUS with no closed-form conditional -> NUTS.
#   * the pooling variances have a conjugate Inverse-Gamma conditional ->
#     `InvGammaVarKernel`, drawn exactly, no AD, no HMC sub-chain.
#   * the ancestral states are MARGINALISED by Felsenstein pruning, so unlike an
#     epidemic trajectory there is no discrete latent block at all.
#
# Note the model is written on the variance scale (`σ²`) rather than the sd
# scale, because that is what makes the Inverse-Gamma conjugacy available. The
# analysis-plan priors are stated as `Exponential` on sd; see `gainloss_model`'s
# docstring for how to keep those instead and pay a NUTS block for them.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The `@model` definitions live at MODULE TOP LEVEL, not inside the builder
# functions below. `@model` expands to global method definitions — including a
# `PracticalBayes.gradmode_plan(::typeof(...))` method — and Julia rejects a
# global method definition placed inside a function body ("Global method
# definition ... needs to be placed at the top level"). The builders therefore
# define the model once here and merely *call* it, which is also cheaper: the
# model type is compiled once for the module rather than re-expanded per call.
#
# Every value the bodies need is passed in as an explicit argument, so they
# capture nothing from the builders' scope.
# ---------------------------------------------------------------------------

@model function gainloss_internal(tree, tiploglik, cov, hab, use_hab, G,
                                  νλ, Sλ2, νμ, Sμ2, b_scale, a_scale)
    a_λ ~ Normal(0, a_scale)
    b_λ ~ Normal(0, b_scale)
    c_λ ~ Normal(0, b_scale)
    a_μ ~ Normal(0, a_scale)
    b_μ ~ Normal(0, b_scale)

    σ²λ ~ InverseGamma(νλ / 2, νλ * Sλ2 / 2)
    σ²μ ~ InverseGamma(νμ / 2, νμ * Sμ2 / 2)

    z_λ ~ filldist(Normal(0, 1), G)
    z_μ ~ filldist(Normal(0, 1), G)

    @addlogprob! _family_loglik(tree, tiploglik, cov, hab, use_hab, G,
                               a_λ, b_λ, c_λ, a_μ, b_μ, sqrt(σ²λ), sqrt(σ²μ),
                               z_λ, z_μ)
end

"""
    _family_loglik(...) -> Real

Total log-likelihood over all gene families. Factored out of the `@model` body for two
reasons: it collapses G separate `@addlogprob!` calls into one, and it is the natural place
to thread.

**Threading**: families are independent given the parameters — the same fixed tree, different
rates — so this is embarrassingly parallel. Measured **3.94x on 8 threads** at 4096 leaves and
32 families, with results identical to the serial version.

Threading is applied ONLY when the element type is a plain `Float64`, i.e. NOT during a
gradient pass. ForwardDiff's `Dual` numbers carry a per-call-site tag and its config is not
safe to share across threads; a threaded gradient here silently risks wrong derivatives, which
is far worse than a slow correct one. NUTS spends most of its time in gradient calls, so the
practical win is smaller than 3.94x — see `docs/08_performance.md`. Set
`ENV["PHYLOGAINLOSS_THREADS"] = "0"` to disable entirely.
"""
function _family_loglik(tree, tiploglik, cov, hab, use_hab, G,
                        a_λ, b_λ, c_λ, a_μ, b_μ, σλ, σμ, z_λ, z_μ)
    T = promote_type(typeof(a_λ), typeof(σλ), eltype(z_λ))
    if T === Float64 && Threads.nthreads() > 1 && get(ENV, "PHYLOGAINLOSS_THREADS", "1") != "0"
        parts = zeros(Float64, Threads.nthreads())
        Threads.@threads for g in 1:G
            parts[Threads.threadid()] += _one_family(tree, tiploglik, cov, hab, use_hab, g,
                                                     a_λ, b_λ, c_λ, a_μ, b_μ, σλ, σμ, z_λ, z_μ)
        end
        return sum(parts)
    end
    total = zero(T)
    for g in 1:G
        total += _one_family(tree, tiploglik, cov, hab, use_hab, g,
                             a_λ, b_λ, c_λ, a_μ, b_μ, σλ, σμ, z_λ, z_μ)
    end
    total
end

@inline function _one_family(tree, tiploglik, cov, hab, use_hab, g,
                             a_λ, b_λ, c_λ, a_μ, b_μ, σλ, σμ, z_λ, z_μ)
    ηλ = a_λ + b_λ * cov[g] + σλ * z_λ[g]
    use_hab && (ηλ += c_λ * hab[g])
    felsenstein_loglik(tree, tiploglik[g], exp(ηλ),
                       exp(a_μ + b_μ * cov[g] + σμ * z_μ[g]))
end

@model function pooled_internal(tree, tiploglik, a_scale)
    a_λ ~ Normal(0, a_scale)
    a_μ ~ Normal(0, a_scale)
    loss = exp(a_λ)
    gain = exp(a_μ)
    for tl in tiploglik
        @addlogprob! felsenstein_loglik(tree, tl, loss, gain)
    end
end

"""
    gainloss_model(tree, tiploglik, covariate; habitat = nothing,
                   νλ = 3.0, Sλ2 = 1.0, νμ = 3.0, Sμ2 = 1.0,
                   b_scale = 1.0, a_scale = 1.5,
                   hmc = NUTS(0.8)) -> (model, spl)

Build the hierarchical two-state gain-loss model over many gene families, and a
ready-to-use blocked sampler for it.

Per family `g` on a fixed tree:

    loss_g = exp(a_λ + b_λ·covariate_g + c_λ·habitat_g + σ_λ·z_λ_g)
    gain_g = exp(a_μ + b_μ·covariate_g +                σ_μ·z_μ_g)

with partial pooling across families through `σ_λ`, `σ_μ`. **The hypothesis is
normally about `b_λ` alone** — where transfer/introgression can move the trait
between lineages, the gain rate absorbs it, so `b_μ` is reported but should not
be read as an evolutionary gain effect.

Arguments:

- `tree` — a `PhyloGainLoss.PrunableTree`, fixed across the fit.
- `tiploglik` — length-`G` vector of `nleaves x 2` matrices,
  `log P(observation | true state)` per family, e.g. from
  `PhyloGainLoss.detection_tiploglik`.
- `covariate` — length-`G`, standardised. The family-level predictor of the
  rates.
- `habitat` — optional length-`G` second covariate on the loss rate; pass
  `nothing` to drop the `c_λ` term.
- `(νλ, Sλ2)`, `(νμ, Sμ2)` — scaled-inv-χ² prior d.f./scale for `σ²_λ`, `σ²_μ`.
- `b_scale`, `a_scale` — prior sd for the slope and intercept coefficients.
- `hmc` — the sampler for the continuous blocks.

Returns `(model, spl)`; sample with `AbstractMCMC.sample(model, spl, N)`.

The family effects use a NON-CENTRED parameterisation (`z ~ Normal(0,1)`, scaled
by `σ`) because the centred form funnels badly when `σ` is small — which is
precisely the "families are exchangeable" case these models are usually near.
This is also what makes the variance conditional conjugate given `z`.

!!! note "Variance vs sd priors"
    The variances carry Inverse-Gamma priors so their conditionals are closed
    form and can be drawn by a kernel instead of NUTS. If you specifically need
    an `Exponential` prior on the standard deviation, write the `@model` by hand
    with `σ ~ Exponential(...)` and put `σ` in a NUTS block — this package's
    builders are a convenience, not a constraint.
"""
function gainloss_model(tree::PrunableTree, tiploglik, covariate;
                        habitat = nothing,
                        νλ = 3.0, Sλ2 = 1.0, νμ = 3.0, Sμ2 = 1.0,
                        b_scale = 1.0, a_scale = 1.5,
                        hmc = NUTS(0.8))
    G = length(tiploglik)
    length(covariate) == G ||
        throw(DimensionMismatch("covariate has length $(length(covariate)) but there are $G families"))
    habitat === nothing || length(habitat) == G ||
        throw(DimensionMismatch("habitat has length $(length(habitat)) but there are $G families"))
    for (g, tl) in enumerate(tiploglik)
        size(tl) == (tree.nleaves, 2) || throw(DimensionMismatch(
            "tiploglik[$g] is $(size(tl)), expected $((tree.nleaves, 2)) to match the tree"))
    end

    hab = habitat === nothing ? zeros(G) : collect(float.(habitat))
    use_hab = habitat !== nothing
    cov = collect(float.(covariate))

    model = gainloss_internal(tree, tiploglik, cov, hab, use_hab, G,
                              νλ, Sλ2, νμ, Sμ2, b_scale, a_scale)

    # Conjugate scale updates: given the non-centred z, the sum of squares is
    # just dot(z, z) with G degrees of freedom.
    σ²λ_kernel = InvGammaVarKernel(:σ²λ, (νλ, Sλ2)) do c
        z = c.values.z_λ
        (length(z), dot(z, z))
    end
    σ²μ_kernel = InvGammaVarKernel(:σ²μ, (νμ, Sμ2)) do c
        z = c.values.z_μ
        (length(z), dot(z, z))
    end

    spl = Gibbs(
        (:a_λ, :b_λ, :c_λ, :a_μ, :b_μ) => hmc,
        (:z_λ, :z_μ)                   => hmc,
        :σ²λ                           => σ²λ_kernel,
        :σ²μ                           => σ²μ_kernel,
    )
    return model, spl
end

"""
    gainloss_pooled_model(tree, tiploglik; νλ = 3.0, Sλ2 = 1.0, hmc = NUTS(0.8))
        -> (model, spl)

The no-covariate special case: one loss and one gain rate shared by every
family, with no pooling structure. Useful as the null model to compare against
[`gainloss_model`](@ref), and as a fast sanity check that the tree, tip data and
sampler are wired together correctly before fitting the full hierarchy.
"""
function gainloss_pooled_model(tree::PrunableTree, tiploglik;
                               a_scale = 1.5, hmc = NUTS(0.8))
    for (g, tl) in enumerate(tiploglik)
        size(tl) == (tree.nleaves, 2) || throw(DimensionMismatch(
            "tiploglik[$g] is $(size(tl)), expected $((tree.nleaves, 2)) to match the tree"))
    end

    model = pooled_internal(tree, tiploglik, a_scale)
    spl = Gibbs((:a_λ, :a_μ) => hmc)
    return model, spl
end
