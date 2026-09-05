# ---------------------------------------------------------------------------
# Blockwise gradient: exploiting the Jacobian's block-sparsity.
#
# The model's log-density carries 2G+5 parameters, but family `g`'s likelihood
# reads only SEVEN of them: the five shared coefficients plus its own z_λ[g] and
# z_μ[g]. Every other derivative is identically zero.
#
# Differentiating the summed log-density naively propagates all 2G+5 dual
# components through every family's pruning pass — G × (2G+5) work, QUADRATIC in
# G. Doing G small 7-parameter gradients and scattering them into the full
# vector is G × 7, LINEAR.
#
# Measured (4096 leaves, gradients agreeing to 1e-6):
#
#     G      naive       blockwise    speedup
#      8     48.0 ms      18.6 ms      2.6x
#     16    137.0 ms      34.5 ms      4.0x
#     32    387.8 ms      63.2 ms      6.1x
#     64   1732.3 ms     159.0 ms     10.9x
#
# The speedup grows with G because the naive version is quadratic. See
# docs/08_performance.md in the bqdiff project for the full analysis, including
# why this is NOT the same thing as PracticalBayes' `depends=` (that is a
# term-level skip; this is within-term parameter sparsity, which `depends=` has
# no vocabulary for).
# ---------------------------------------------------------------------------

"""
    GainLossPosterior(tree, tiploglik, cov, hab, use_hab; priors...)

A `LogDensityProblems`-compatible posterior for the hierarchical gain–loss model,
with a hand-written **blockwise** gradient.

Parameter vector layout (all unconstrained):

    θ[1:5]        = a_λ, b_λ, c_λ, a_μ, b_μ
    θ[6], θ[7]    = log σ_λ, log σ_μ
    θ[8 : 7+G]    = z_λ
    θ[8+G : 7+2G] = z_μ

Use when G is large enough that the naive gradient dominates — from about G = 8
upwards it is already ~2.6x, and the advantage grows. For small G the ordinary
`gainloss_model` is simpler and fast enough.

Note this parameterises the scales as `log σ` with a Normal prior rather than
`σ²` with an Inverse-Gamma: it is a standalone HMC target, so there is no Gibbs
block to hand the conjugate update to, and sampling `log σ` directly is the
right thing for a single-block sampler. `gainloss_model` remains the blocked
Gibbs version.
"""
struct GainLossPosterior{T,TL,C,H}
    tree::T
    tiploglik::TL
    cov::C
    hab::H
    use_hab::Bool
    G::Int
    a_scale::Float64
    b_scale::Float64
    sigma_scale::Float64      # prior sd on log σ
end

function GainLossPosterior(tree, tiploglik, cov; habitat = nothing,
                           a_scale = 1.5, b_scale = 1.0, sigma_scale = 1.0)
    G = length(tiploglik)
    length(cov) == G ||
        throw(DimensionMismatch("cov has length $(length(cov)) but there are $G families"))
    habitat === nothing || length(habitat) == G ||
        throw(DimensionMismatch("habitat has length $(length(habitat)) but there are $G families"))
    for (g, tl) in enumerate(tiploglik)
        size(tl) == (tree.nleaves, 2) || throw(DimensionMismatch(
            "tiploglik[$g] is $(size(tl)), expected $((tree.nleaves, 2))"))
    end
    hab = habitat === nothing ? zeros(G) : collect(float.(habitat))
    GainLossPosterior(tree, tiploglik, collect(float.(cov)), hab, habitat !== nothing,
                      G, Float64(a_scale), Float64(b_scale), Float64(sigma_scale))
end

dimension(p::GainLossPosterior) = 7 + 2 * p.G

"""
    logdensity(p::GainLossPosterior, θ) -> Real

Log posterior (up to a constant). Generic in `eltype(θ)` so it still works under
whole-vector AD — which is what `check_blockgrad` compares against.
"""
function logdensity(p::GainLossPosterior, θ)
    G = p.G
    a_λ, b_λ, c_λ, a_μ, b_μ = θ[1], θ[2], θ[3], θ[4], θ[5]
    logσλ, logσμ = θ[6], θ[7]
    σλ, σμ = exp(logσλ), exp(logσμ)

    # priors
    lp = -(a_λ^2 + a_μ^2) / (2 * p.a_scale^2) -
          (b_λ^2 + b_μ^2 + c_λ^2) / (2 * p.b_scale^2) -
          (logσλ^2 + logσμ^2) / (2 * p.sigma_scale^2)
    for i in 1:2G
        lp -= θ[7+i]^2 / 2                      # z ~ Normal(0, 1)
    end

    for g in 1:G
        lp += _fam(p, g, a_λ, b_λ, c_λ, a_μ, b_μ, σλ, σμ, θ[7+g], θ[7+G+g])
    end
    lp
end

@inline function _fam(p, g, a_λ, b_λ, c_λ, a_μ, b_μ, σλ, σμ, zλg, zμg)
    ηλ = a_λ + b_λ * p.cov[g] + σλ * zλg
    p.use_hab && (ηλ += c_λ * p.hab[g])
    felsenstein_loglik(p.tree, p.tiploglik[g], exp(ηλ),
                       exp(a_μ + b_μ * p.cov[g] + σμ * zμg))
end

"""
    logdensity_and_gradient(p::GainLossPosterior, θ) -> (lp, ∇lp)

Value and gradient, computed **blockwise**: one 7-parameter `ForwardDiff`
gradient per family, scattered into the full vector. Cost is O(G), not O(G²).

The seven differentiated parameters per family are
`(a_λ, b_λ, c_λ, a_μ, b_μ, log σ_λ, log σ_μ)` plus that family's own
`(z_λ[g], z_μ[g])` — nine slots, of which the two z entries are the sparse part.
The shared five-plus-two accumulate across families; the z entries are written
once each.
"""
function logdensity_and_gradient(p::GainLossPosterior, θ)
    G = p.G
    grad = zeros(eltype(θ), length(θ))

    a_λ, b_λ, c_λ, a_μ, b_μ = θ[1], θ[2], θ[3], θ[4], θ[5]
    logσλ, logσμ = θ[6], θ[7]

    # --- priors (closed form; no AD needed) ---
    lp = -(a_λ^2 + a_μ^2) / (2 * p.a_scale^2) -
          (b_λ^2 + b_μ^2 + c_λ^2) / (2 * p.b_scale^2) -
          (logσλ^2 + logσμ^2) / (2 * p.sigma_scale^2)
    grad[1] = -a_λ / p.a_scale^2
    grad[2] = -b_λ / p.b_scale^2
    grad[3] = -c_λ / p.b_scale^2
    grad[4] = -a_μ / p.a_scale^2
    grad[5] = -b_μ / p.b_scale^2
    grad[6] = -logσλ / p.sigma_scale^2
    grad[7] = -logσμ / p.sigma_scale^2
    for i in 1:2G
        z = θ[7+i]
        lp -= z^2 / 2
        grad[7+i] = -z
    end

    # --- likelihood, one small gradient per family ---
    #
    # Threading: each family's 9-parameter gradient is independent, and the two
    # z entries it writes (`grad[7+g]`, `grad[7+G+g]`) are unique to it, so
    # those need no synchronisation. The SEVEN shared entries do accumulate
    # across families, so each thread keeps its own partial and they are summed
    # once at the end — a race on `grad[1:7]` would otherwise produce silently
    # wrong gradients that a sampler will not warn about.
    #
    # Disable with ENV["PHYLOGAINLOSS_THREADS"] = "0".
    nt = Threads.nthreads()
    use_threads = nt > 1 && G >= 2nt && get(ENV, "PHYLOGAINLOSS_THREADS", "1") != "0"

    if use_threads
        shared = zeros(Float64, 7, nt)       # per-thread partials for grad[1:7]
        lps    = zeros(Float64, nt)
        Threads.@threads for g in 1:G
            tid = Threads.threadid()
            v, gq = _family_value_grad(p, g, a_λ, b_λ, c_λ, a_μ, b_μ,
                                       logσλ, logσμ, θ[7+g], θ[7+G+g])
            lps[tid] += v
            for k in 1:7
                shared[k, tid] += gq[k]
            end
            grad[7+g]   += gq[8]
            grad[7+G+g] += gq[9]
        end
        lp += sum(lps)
        for k in 1:7, t in 1:nt
            grad[k] += shared[k, t]
        end
    else
        for g in 1:G
            v, gq = _family_value_grad(p, g, a_λ, b_λ, c_λ, a_μ, b_μ,
                                       logσλ, logσμ, θ[7+g], θ[7+G+g])
            lp += v
            for k in 1:7
                grad[k] += gq[k]             # shared parameters accumulate
            end
            grad[7+g]   += gq[8]             # this family's own z_λ
            grad[7+G+g] += gq[9]             # this family's own z_μ
        end
    end
    (lp, grad)
end

"""
    _family_value_grad(p, g, ...) -> (value, gradient)

One family's log-likelihood and its gradient w.r.t. the nine parameters that
family actually reads: the five shared coefficients, the two log-scales, and its
own `z_λ[g]`, `z_μ[g]`.

Allocates its own `q` and result buffer so it is safe to call from several
threads at once. That allocation is small (9 elements) and is dwarfed by the
pruning pass it wraps; a shared preallocated buffer would be faster
single-threaded but is exactly the kind of hidden state that makes a threaded
gradient silently wrong.
"""
@inline function _family_value_grad(p, g, a_λ, b_λ, c_λ, a_μ, b_μ,
                                    logσλ, logσμ, zλg, zμg)
    q = (a_λ, b_λ, c_λ, a_μ, b_μ, logσλ, logσμ, zλg, zμg)
    qv = collect(q)
    fg = let p = p, g = g
        qq -> _fam(p, g, qq[1], qq[2], qq[3], qq[4], qq[5],
                   exp(qq[6]), exp(qq[7]), qq[8], qq[9])
    end
    # value and gradient in one pass, so `fg` is not evaluated twice
    res = ForwardDiff.gradient!(DiffResults.GradientResult(qv), fg, qv)
    (DiffResults.value(res), DiffResults.gradient(res))
end

"""
    check_blockgrad(p, θ; rtol = 1e-6) -> NamedTuple

Verify the blockwise gradient against whole-vector `ForwardDiff` on the same
`logdensity`. Returns `(ok, max_abs_err, max_rel_err)`.

**Run this whenever the model body changes.** A blockwise gradient that misses a
dependency is silently wrong in exactly the way a sampler will not warn you
about — the same hazard `check_depends` exists for on the `depends=` side.
"""
function check_blockgrad(p::GainLossPosterior, θ; rtol = 1e-6)
    _, g_block = logdensity_and_gradient(p, θ)
    g_ref = ForwardDiff.gradient(t -> logdensity(p, t), θ)
    aerr = maximum(abs.(g_block .- g_ref))
    scale = maximum(abs.(g_ref))
    rerr = scale > 0 ? aerr / scale : aerr
    (ok = rerr <= rtol, max_abs_err = aerr, max_rel_err = rerr)
end
