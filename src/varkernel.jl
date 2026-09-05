# ---------------------------------------------------------------------------
# Conjugate variance-component kernel.
#
# Same rationale as ParcticalBayesianGenomics' `InvGammaVarKernel`: a variance
# whose full conditional is closed-form should be drawn exactly, not pushed
# through NUTS, which costs an AD gradient and an HMC sub-chain for a one-line
# update and mixes worse.
#
# Duplicated rather than imported so this package does not depend on the whole
# genomics stack for one struct. Kept deliberately identical in contract so the
# two can be unified later — the right place for it is PracticalBayes itself,
# alongside the :beta / :dirichlet cases `ConjugateGibbs` already ships.
# ---------------------------------------------------------------------------

"""
    InvGammaVarKernel(name, prior, ss)
    InvGammaVarKernel(ss, name, prior)      # `do`-block form

An `AbstractLatentKernel` that resamples the variance block `name` from its
conjugate Inverse-Gamma full conditional.

The model must place `name ~ InverseGamma(ν/2, ν·S²/2)` on the variance;
`prior = (ν, S²)` names those same hyperparameters. The kernel adds the data's
degrees of freedom and sum of squares and draws

    name ~ InverseGamma((ν + n)/2, (ν·S² + SS)/2)

with `(n, SS)` from the user's `ss(c) -> (n, SS)` closure, which does all the
model-specific work.

# Example — the pooling scale of a non-centred family effect `z`:

```julia
k = InvGammaVarKernel(:σ², (3.0, 1.0)) do c
    z = c.values.z
    (length(z), dot(z, z))
end
```
"""
struct InvGammaVarKernel{F} <: AbstractLatentKernel
    name::Symbol
    prior::Tuple{Float64,Float64}
    ss::F
end

InvGammaVarKernel(name::Symbol, prior, ss) =
    InvGammaVarKernel(name, (Float64(prior[1]), Float64(prior[2])), ss)
# ss-first method so the `do`-block form reads naturally, matching
# ConjugateGibbs' own count-first constructor.
InvGammaVarKernel(ss, name::Symbol, prior) =
    InvGammaVarKernel(name, (Float64(prior[1]), Float64(prior[2])), ss)

function PracticalBayes.latent_step(rng::AbstractRNG, k::InvGammaVarKernel,
                                    block_names, c::ModelConditional)
    block_names == (k.name,) ||
        error("InvGammaVarKernel owns block $(k.name), got $(block_names)")
    ν, S² = k.prior
    n, SS = k.ss(c)
    draw = rand(rng, InverseGamma((ν + n) / 2, (ν * S² + SS) / 2))
    return NamedTuple{(k.name,)}((draw,))
end
