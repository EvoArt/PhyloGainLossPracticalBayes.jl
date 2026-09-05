# ---------------------------------------------------------------------------
# Phylogenetic relatedness.
#
# A phylogenetic random effect is a relatedness structure by another name. Under
# Brownian motion on a tree, cov(i, j) = the shared path length from the root to
# tips i and j, which plays exactly the role the numerator relationship matrix A
# plays in an animal model. ParcticalBayesianGenomics already has the machinery
# for "a precomputed K inverse plus a label saying where it came from", reused
# every sweep and never re-inverted; this is the fourth constructor for it.
#
# Deliberately mirrors `RelatednessStructure` rather than depending on it: this
# package must not pull in the whole genomics stack for one struct. If the two
# ever need to interoperate, the right move is to lift the abstraction into
# PracticalBayes itself and have both depend on that.
# ---------------------------------------------------------------------------

"""
    PhyloRelatedness

Holds a precomputed inverse phylogenetic covariance `Kinv` (`K⁻¹`), the `kind`
label (`:brownian` or `:ou`), and the leaf order the matrix is indexed by.

Built ONCE via [`phylo_relatedness`](@ref) and reused every sweep. The same
contract as `ParcticalBayesianGenomics.RelatednessStructure`: nothing downstream
re-inverts `K`.

`leaf_names` is carried so a caller can check its trait table is in the same
order the matrix assumes — a silently mis-ordered relatedness matrix is a
particularly hard bug to see, because the model still fits.
"""
struct PhyloRelatedness{M<:AbstractMatrix}
    Kinv::M
    kind::Symbol
    leaf_names::Vector{String}
end

Base.size(r::PhyloRelatedness, args...) = size(r.Kinv, args...)

"""
    brownian_covariance(tree) -> (K, leaf_names)

Phylogenetic covariance matrix under Brownian motion: `K[i,j]` is the summed
branch length from the root to the most recent common ancestor of leaves `i`
and `j`, and `K[i,i]` is the root-to-tip distance.

Returned in the leaf order of `PhyloGainLoss.leaforder(tree)`, which is also
returned so the caller can align a trait table against it.

Computed by accumulating root-to-node depths once, then walking each internal
node and adding its depth to every pair of leaves whose MRCA it is — O(n²) in
the number of leaves, done once before sampling.
"""
function brownian_covariance(tree::PrunableTree)
    n = tree.nleaves
    depth = zeros(Float64, nnodes(tree))
    _depths!(depth, tree, PhyloGainLoss.root(tree), 0.0)

    K = zeros(Float64, n, n)
    # For each node, every pair of leaves in DIFFERENT child subtrees has this
    # node as its MRCA; every leaf below it gets this node's depth on the
    # diagonal path. Collect leaf sets bottom-up.
    leaves_below = Vector{Vector{Int}}(undef, nnodes(tree))
    _collect_leaves!(leaves_below, tree, PhyloGainLoss.root(tree))

    for node in 1:nnodes(tree)
        PhyloGainLoss.isleaf(tree, node) && continue
        kids = PhyloGainLoss.children(tree, node)
        d = depth[node]
        for a in eachindex(kids), b in eachindex(kids)
            a < b || continue
            for i in leaves_below[kids[a]], j in leaves_below[kids[b]]
                K[i, j] = d
                K[j, i] = d
            end
        end
    end
    for node in 1:nnodes(tree)
        li = PhyloGainLoss.leafindex(tree, node)
        li == 0 && continue
        K[li, li] = depth[node]
    end
    K, PhyloGainLoss.leaforder(tree)
end

function _depths!(depth, tree, node, acc)
    depth[node] = acc
    for c in PhyloGainLoss.children(tree, node)
        _depths!(depth, tree, c, acc + PhyloGainLoss.branchlength(tree, c))
    end
    return
end

function _collect_leaves!(store, tree, node)
    if PhyloGainLoss.isleaf(tree, node)
        store[node] = [PhyloGainLoss.leafindex(tree, node)]
        return store[node]
    end
    acc = Int[]
    for c in PhyloGainLoss.children(tree, node)
        append!(acc, _collect_leaves!(store, tree, c))
    end
    store[node] = acc
    acc
end

"""
    phylo_relatedness(tree; kind = :brownian, lambda = 1.0, jitter = 1e-8)
        -> PhyloRelatedness

Build the inverse phylogenetic covariance for a Brownian tree effect.

- `tree` — a `PhyloGainLoss.PrunableTree`.
- `lambda` — Pagel's λ, scaling the off-diagonals toward a star phylogeny.
  `λ = 1` is pure Brownian; `λ = 0` removes phylogenetic structure entirely and
  is the natural null for "is the tree effect doing anything?".
- `jitter` — added to the diagonal before inversion. A phylogenetic covariance
  is singular whenever two tips are identical (zero-length branches, which a
  pruned backbone can easily contain), so some ridge is required. Report it:
  a large jitter quietly weakens the tree effect.

The covariance is standardised to unit mean diagonal before inversion, so the
variance component `σ²` it multiplies is on a comparable scale regardless of
the tree's absolute branch lengths.
"""
function phylo_relatedness(tree::PrunableTree; kind::Symbol = :brownian,
                           lambda::Real = 1.0, jitter::Real = 1e-8)
    kind === :brownian || throw(ArgumentError("only :brownian is implemented, got $kind"))
    (0 <= lambda <= 1) || throw(ArgumentError("lambda must be in [0, 1], got $lambda"))

    K, names = brownian_covariance(tree)
    n = size(K, 1)

    # Pagel's lambda: shrink off-diagonals, keep the diagonal.
    if lambda != 1
        d = diag(K)
        K .*= lambda
        for i in 1:n
            K[i, i] = d[i]
        end
    end

    # Standardise scale so sigma^2 means the same thing across trees.
    m = sum(diag(K)) / n
    m > 0 || throw(ArgumentError("tree has zero total depth; check branch lengths were parsed"))
    K ./= m

    for i in 1:n
        K[i, i] += jitter
    end

    F = cholesky(Symmetric(K); check = false)
    issuccess(F) || throw(ArgumentError(
        "phylogenetic covariance is not positive definite even with jitter=$jitter; " *
        "increase jitter, or check the tree for zero-length or negative branches"))
    Kinv = inv(F)
    PhyloRelatedness(Matrix(Kinv), kind, names)
end
