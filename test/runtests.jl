using PhyloGainLossPracticalBayes
using PhyloGainLoss
using PracticalBayes
using AbstractMCMC
using Distributions
using LinearAlgebra
using StableRNGs
using Statistics
using Test

# ((A:0.1,B:0.2):0.3,C:0.4) — root=1, x=2, C=3, A=4, B=5
smalltree() = PrunableTree([0, 1, 1, 2, 2], [0.0, 0.3, 0.4, 0.1, 0.2],
                           ["root", "x", "C", "A", "B"])

fourleaf() = PrunableTree([0, 1, 1, 2, 2, 3, 3],
                          [0.0, 0.3, 0.6, 0.4, 0.55, 0.25, 0.7],
                          ["root", "x", "y", "A", "B", "C", "D"])

# --- helpers -------------------------------------------------------------------------

# A trivial model just to satisfy ModelConditional's field; the kernel never evaluates it.
@model function _dummy() ; x ~ Normal(0, 1) ; end
_dummy_model() = _dummy()

_leafnodes(t) = [i for i in 1:PhyloGainLoss.nnodes(t) if PhyloGainLoss.leafindex(t, i) > 0]

"""Forward-simulate true states down the tree; returns a state per node (1 = absent, 2 = present)."""
function _simulate_states(rng, t, loss, gain)
    n = PhyloGainLoss.nnodes(t)
    s = zeros(Int, n)
    rt = PhyloGainLoss.root(t)
    s[rt] = rand(rng) < gain / (loss + gain) ? 2 : 1
    _descend!(rng, s, t, rt, loss, gain)
    s
end

function _descend!(rng, s, t, node, loss, gain)
    for c in PhyloGainLoss.children(t, node)
        (p11, p12, p21, p22) = transition_probabilities(loss, gain, PhyloGainLoss.branchlength(t, c))
        s[c] = if s[node] == 1
            rand(rng) < p12 ? 2 : 1
        else
            rand(rng) < p21 ? 1 : 2
        end
        _descend!(rng, s, t, c, loss, gain)
    end
end

@testset "PhyloGainLossPracticalBayes" begin

    @testset "brownian covariance" begin
        t = smalltree()
        K, names = brownian_covariance(t)
        iA = findfirst(==("A"), names)
        iB = findfirst(==("B"), names)
        iC = findfirst(==("C"), names)

        @test K[iA, iA] ≈ 0.4        # root-to-tip: 0.3 + 0.1
        @test K[iB, iB] ≈ 0.5        # 0.3 + 0.2
        @test K[iC, iC] ≈ 0.4
        @test K[iA, iB] ≈ 0.3        # shared MRCA depth
        @test K[iA, iC] ≈ 0.0        # share only the root
        @test K[iB, iC] ≈ 0.0
        @test issymmetric(K)
        @test all(eigvals(Symmetric(K)) .>= -1e-12)
    end

    @testset "phylo_relatedness" begin
        t = fourleaf()
        rel = phylo_relatedness(t)
        @test rel.kind == :brownian
        @test size(rel) == (4, 4)
        @test rel.leaf_names == leaforder(t)
        @test issymmetric(rel.Kinv)

        # Kinv really is the inverse of the (standardised, jittered) covariance
        K, _ = brownian_covariance(t)
        K ./= sum(diag(K)) / 4
        K += 1e-8I
        @test rel.Kinv * K ≈ I(4) atol = 1e-6

        # lambda = 0 removes phylogenetic structure: off-diagonals vanish, so
        # K is diagonal and so is its inverse
        rel0 = phylo_relatedness(t; lambda = 0.0)
        offdiag = [rel0.Kinv[i, j] for i in 1:4, j in 1:4 if i != j]
        @test all(abs.(offdiag) .< 1e-8)

        @test_throws ArgumentError phylo_relatedness(t; lambda = 1.5)
        @test_throws ArgumentError phylo_relatedness(t; kind = :ou)
    end

    @testset "InvGammaVarKernel draws from the right conditional" begin
        # With a known z, the conditional is InverseGamma((ν+n)/2, (ν·S² + zᵀz)/2).
        # Check the sample mean against that distribution's mean.
        ν, S² = 4.0, 1.0
        z = [0.5, -1.2, 0.3, 0.8, -0.4]
        n, SS = length(z), dot(z, z)
        target = InverseGamma((ν + n) / 2, (ν * S² + SS) / 2)

        k = InvGammaVarKernel(:σ², (ν, S²)) do c
            zz = c.values.z
            (length(zz), dot(zz, zz))
        end

        # Minimal stand-in for ModelConditional: the kernel only reads c.values.
        rng = StableRNG(20260905)
        draws = [PracticalBayes.gibbs_step(rng, k, (:σ²,),
                    ModelConditional(_dummy_model(), (; z = z, σ² = 1.0))).σ² for _ in 1:20_000]
        @test mean(draws) ≈ mean(target) rtol = 0.05
        @test std(draws) ≈ std(target) rtol = 0.10
    end

    @testset "model builders return a usable (model, spl)" begin
        t = fourleaf()
        tips = [detection_tiploglik([true, false, true, false], [0.9, 0.8, 1.0, 0.7]),
                detection_tiploglik([false, false, true, true], [0.85, 0.6, 0.95, 0.9]),
                detection_tiploglik([true, true, false, false], [1.0, 0.75, 0.8, 0.7])]
        cov = [-1.0, 0.0, 1.0]

        model, spl = gainloss_model(t, tips, cov)
        @test spl isa Gibbs
        # every assumed variable is in exactly one block — Gibbs checks this on first step
        chain = AbstractMCMC.sample(StableRNG(1), model, spl, 20; progress = false)
        @test length(vec(collect(chain[:a_λ]))) == 20

        nullmodel, nullspl = gainloss_pooled_model(t, tips)
        nullchain = AbstractMCMC.sample(StableRNG(1), nullmodel, nullspl, 20; progress = false)
        @test length(vec(collect(nullchain[:a_λ]))) == 20
    end

    @testset "builders reject mismatched shapes" begin
        t = fourleaf()
        tips = [detection_tiploglik([true, false, true, false], fill(0.9, 4))]
        @test_throws DimensionMismatch gainloss_model(t, tips, [1.0, 2.0])       # wrong G
        @test_throws DimensionMismatch gainloss_model(t, tips, [1.0]; habitat = [1.0, 2.0])

        # tip matrix that does not match the tree's leaf count
        badtips = [detection_tiploglik([true, false], fill(0.9, 2))]
        @test_throws DimensionMismatch gainloss_model(t, badtips, [1.0])
    end

    @testset "blocked_folds partitions by group" begin
        f = blocked_folds(["s1", "s1", "s2", "s3", "s2", "s1"])
        @test length(f) == 3
        @test sort(sort.(f)) == sort([[1, 2, 6], [3, 5], [4]])
        @test sort(vcat(f...)) == collect(1:6)      # every index exactly once
        @test length(blocked_folds([1, 1, 1])) == 1
        @test length(blocked_folds(1:5)) == 5
    end

    @testset "elpd_holdout matches a hand calculation and is stable" begin
        ll = [log(0.1) log(0.4); log(0.2) log(0.5); log(0.3) log(0.6)]
        w1 = log(mean([0.1, 0.2, 0.3]))
        w2 = log(mean([0.4, 0.5, 0.6]))
        @test elpd_holdout(ll, [1]) ≈ w1 atol = 1e-12
        @test elpd_holdout(ll, [2]) ≈ w2 atol = 1e-12
        @test elpd_holdout(ll, [1, 2]) ≈ w1 + w2 atol = 1e-12
        @test elpd_holdout(ll, Int[]) == 0.0

        # log-sum-exp, not naive mean(exp(.)): at this magnitude the naive form underflows
        tiny = fill(-10_000.0, 5, 1)
        @test elpd_holdout(tiny, [1]) ≈ -10_000.0 atol = 1e-9
        @test log(mean(exp.(tiny[:, 1]))) == -Inf
    end

    @testset "pointwise_loglik reproduces per-family likelihoods at each draw" begin
        t = fourleaf()
        tips = [detection_tiploglik([true, false, true, false], [0.9, 0.8, 1.0, 0.7]),
                detection_tiploglik([false, false, true, true], [0.85, 0.6, 0.95, 0.9]),
                detection_tiploglik([true, true, false, false], [1.0, 0.75, 0.8, 0.7])]
        cov = [-1.0, 0.0, 1.0]
        model, spl = gainloss_model(t, tips, cov)
        chain = AbstractMCMC.sample(StableRNG(3), model, spl, 30; progress = false)

        ll = pointwise_loglik(chain, t, tips, cov)
        @test size(ll) == (30, 3)
        @test all(isfinite, ll)

        # Recomputing family-by-family must reproduce what the model added: check one draw
        # against a direct evaluation at that draw's parameters.
        d = 1
        σλ = sqrt(vec(collect(chain[:σ²λ]))[d])
        σμ = sqrt(vec(collect(chain[:σ²μ]))[d])
        aλ = vec(collect(chain[:a_λ]))[d]; bλ = vec(collect(chain[:b_λ]))[d]
        aμ = vec(collect(chain[:a_μ]))[d]; bμ = vec(collect(chain[:b_μ]))[d]
        zλ = vec(collect(chain[:z_λ]))[d]; zμ = vec(collect(chain[:z_μ]))[d]
        for g in 1:3
            want = felsenstein_loglik(t, tips[g],
                                      exp(aλ + bλ * cov[g] + σλ * zλ[g]),
                                      exp(aμ + bμ * cov[g] + σμ * zμ[g]))
            @test ll[d, g] ≈ want atol = 1e-10
        end
    end

    @testset "recovers a known covariate effect" begin
        # Simulate families whose loss rates really do depend on the covariate, then check
        # the posterior for b_λ is positive. This is the end-to-end check that the model,
        # the blocking and the likelihood are wired together correctly.
        rng = StableRNG(20260905)
        t = fourleaf()
        G = 12
        cov = collect(range(-1.5, 1.5; length = G))
        tips = map(1:G) do g
            loss = exp(-0.5 + 1.2 * cov[g])       # true b_λ = 1.2
            gain = exp(-0.5)
            # simulate presence/absence down the tree
            states = _simulate_states(rng, t, loss, gain)
            obs = [states[i] == 2 for i in _leafnodes(t)]
            detection_tiploglik(obs, fill(1.0, t.nleaves))
        end

        model, spl = gainloss_model(t, tips, cov)
        chain = AbstractMCMC.sample(StableRNG(7), model, spl, 400; progress = false)
        b = vec(collect(chain[:b_λ]))[201:end]
        @test mean(b) > 0        # the sign is recovered
    end
end

