# Handover — recovered context

The session that built this package (`black-queen-22`) ended before it could
report back. The code survived intact on disk; this file records the state it
was in and what it had found, so a new session can pick it up cold.

Written by a peer session (`practicalbayes-de`) from our message exchanges, not
from that session's own transcript — so treat "what it was doing" as reliable
and "what it intended next" as inference.

## TWO packages were built, not one

That session created a pair, and both were untracked until now:

  * **`PhyloGainLoss`** (`~/.julia/dev/PhyloGainLoss`, 719 lines, commit
    `3326804`) — the computational package. Two-state gain/loss CTMC on a
    phylogeny with an explicit tip observation model, so an observed absence is
    not assumed to be a true absence. Deliberately PPL-agnostic:
    `felsenstein_loglik` is a plain function of reals. `src/` is
    `ctmc.jl`, `ancestral.jl`, `tree.jl`, `newick.jl`; there is a `docs/src`
    and a 217-line test suite. Its README argues the case against CAFE5 /
    BadiRate / corHMM (per-family covariate on the rate, partial pooling,
    detection model).
  * **`PhyloGainLossPracticalBayes`** (this package, 792 lines, commit
    `ffbdb1e`) — the PracticalBayes glue.

Both now have `.gitignore` (Manifest excluded) and an initial commit under
Arthur's identity. Neither has a remote.

## What this package is

Glue between [PhyloGainLoss.jl](../PhyloGainLoss) (PPL-agnostic two-state CTMC
likelihood) and [PracticalBayes.jl](../PracticalBayes). Same layering as
`EpidemicTrajectories` + `PracticalEpiBayes`: the computational package knows
nothing about any PPL; this one supplies the small repetitive PracticalBayes
pieces you would otherwise hand-write per model.

792 lines across `src/` (5 files) and `test/runtests.jl`. Now under git —
`ffbdb1e` is the initial commit and contains everything as that session left it.

  * `gainloss.jl` (164) — model builders returning `(model, spl)`, matching the
    shape of `PracticalBayesianGenomics.animal_model`
  * `loo.jl` (137)
  * `relatedness.jl` (157)
  * `varkernel.jl` (61) — the conjugate InverseGamma variance kernel; see below
  * `test/runtests.jl` (217) — 4-leaf trees, a few hundred draws

## Exactly where it got to

It was **running `Pkg.test()` and had not reported the result**. Its own
estimate was "minutes, not hours, unless precompilation of the full stack
dominates". So:

**The first thing to do is re-run `Pkg.test()`** — the suite's status is
genuinely unknown, not known-good. Do not assume the package is green because
the code looks finished. Both packages have a 217-line `test/runtests.jl`, so
run both.

Machine state when the session ended: 13 idle `julia.exe` processes on 1.12.4,
no 1.12.7, ~3.5 GB free. Those were cleared afterwards.

## The one design finding worth acting on

`varkernel.jl` duplicates `PracticalBayesianGenomics`'s `InvGammaVarKernel`
almost exactly. That session flagged it rather than let it spread, and it is
right — I checked both:

  * PracticalBayes' `ConjugateGibbs` ships only `:beta` and `:dirichlet`, and
    dispatches its draw on `Val(family)` — a clean extension point.
  * `InvGammaVarKernel` matches it structurally: same `AbstractGibbsKernel`
    supertype, same `gibbs_step(rng, k, block_names, c)` contract, same
    closure-first constructor so the `do`-block form reads naturally, same
    "user closure supplies the model-specific sufficient statistics" split.
    It IS `ConjugateGibbs` with `family = :invgamma`, where the closure returns
    `(n, SS)` instead of counts.

`PracticalBayesianGenomics/src/variance_kernels.jl` says so itself: "the
Inverse-Gamma sibling of PracticalBayes' own ConjugateGibbs. Same contract."

Three packages now carry the same struct. Upstreaming it into `ConjugateGibbs`
as a third family is a small change, **but it is Arthur's call** — it widens
PracticalBayes' public API and commits him to maintaining it. Not done.

If it does land, there is one real design decision: this package (and
PracticalBayesianGenomics) take `prior = (ν, S²)` in the scaled-inverse-χ²
parameterisation and convert internally, whereas `ConjugateGibbs` passes
`prior` straight to the distribution. Either the shared kernel takes
`InverseGamma` hyperparameters directly (consistent with the existing families,
but call sites change) or it keeps the (ν, S²) convention and documents the
difference. I would lean toward the former.

## A tip that never reached that session

Arthur asked me to pass on: **`Pkg.add(url=...)` may get you around the
`[sources]` issue.** Relevant if this package ever depends on something
unregistered.

`[sources]` has two sharp edges, both hit in PracticalBayes:

  * It is silently IGNORED before Julia 1.11, so on 1.10 the dependency becomes
    unresolvable ("X has no known versions!").
  * An entry may only name a package that also appears in `[deps]` or
    `[extras]`. A `[weakdeps]`-only entry is rejected outright ("Sources for `X`
    not listed in `deps` or `extras` section"), breaking buildpkg on every Julia
    version.

`Pkg.add(url=...)` sidesteps both: it writes a `repo-url` plus a pinned
`git-tree-sha1` into the manifest, so resolution never depends on `[sources]`
being understood.

**Caveat:** it also replaces any local `dev` path. Good for a reproducible run,
bad while you are editing that dependency — it will silently use the GitHub copy
instead of your local edits. `pkgdir(TheDep)` tells you which one loaded.

## Unrelated, but you will hit it if you touch PracticalBayes/benchmarks

`benchmarks/sweep.jl` used to crash Julia 1.12.7 at startup — bare
`RaiseException` inside libjulia-codegen/LLVM, empty stdout, dead before a
single script line ran. Cause was `Pkg.instantiate()` running unconditionally
in the script header. Fixed: `develop`/`instantiate` are now skipped when the
environment already exists, with `PB_SWEEP_INSTANTIATE=1` to force them.
