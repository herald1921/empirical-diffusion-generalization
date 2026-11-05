# Projected-neighborhood diversity for the synthetic supports

This calculation adds only `G(r) = D_p(Pi(U_r))`, with ambient Euclidean
transport cost. It does not recompute or alter the published W2, WT, WN
curves, empirical endpoint estimates, training points, snapshots or topology.

`diversity_curves.csv` has columns `shape,eta,r,G,G_over_SN,defined,exact_zero`.
There are 422 eta values per included support. For plotting use eta > 0;
the eta=0 row is a diagnostic. Undefined values are `nan`, not zero, and
must not be joined across the definition boundary. Exact zero is retained
as zero and is outside the finite logarithmic vertical range. There is no
figure-eight row.

## Domains and calibration

In all rows, r = (sqrt(D) + sqrt(2 log(20))) eta, as in the original
neighborhood experiment. The ellipse has D=2 and reach 1/2, so G is shown
only for eta < 0.12946792538299864. The sphere has D=3 and reach 2, so
eta < 0.47849206423244633. The segment is closed and convex: the identical
projected-neighborhood formula is well-defined at every radius using its
globally unique projection, despite the segment having boundary. The
self-crossing figure eight has zero reach and is omitted, as requested.

## Ellipse

The population is y(theta)=(2 cos(theta),sin(theta)), with probability
density (1+0.1 cos(theta))/(2 pi) in theta. Use the frozen 16-site quantile
design, not a new design. Below reach, projection of each sample ball is a
connected parameter arc. Its endpoints solve

`abs((z_i-y(theta)) dot unit_tangent(theta)) = r`.

The normal coordinate at the corresponding normal-line foot is strictly
smaller than reach. Thus it is indeed a point whose nearest ellipse point
is y(theta). The first such root in either direction bounds the projected
ball image. Union the arcs, then integrate squared distance on their
uncovered complementary arcs against the stated density.

For an uncovered target point, the nearest point of the projected union
must be one of its arc endpoints. Otherwise the nonzero joining chord is
normal at an interior nearest point, and the reach normal-chord inequality
forces its length to be at least twice reach, namely 1. But an original
sample is always within 0.437: max|y'|=2 and the maximum sample angular
gap is less than 0.437. Since every sample lies in the projected union,
the nearest distance is strictly less than 1, a contradiction. This also
explains why taking the minimum over endpoints is a continuous-support
calculation, not a replacement by a discretized support.

The implementation solves roots to about machine precision and uses
adaptive one-dimensional quadrature. Independent 2^18-angle midpoint
integration differs by at most 1.29e-10 in G squared at the checked radii.
Independent Lagrange-multiplier nearest-point projection of 1024 boundary
directions per sample at r=.08,.24,.49 produces no arc violations; endpoint
projection and ball-distance errors are below 1.9e-15.

## Sphere

For radius R=2 and r<R, each projected sample ball is a spherical cap of
angular half-width beta=asin(r/R). If theta is the angle from a uniform
target point to its nearest sample direction, then

`G(r)^2 = 2 R^2 E[1-cos((theta-beta)_+)]`.

The computation uses 2^20 scrambled Sobol target directions. It checks
2^18 refinement and an independent 2^20 scramble. The largest independent
G discrepancy at the checked radii is 9.32e-7 in absolute distance. These
are numerical sensitivity checks, not certified continuum error bounds.
The cap-covering threshold is computed from spherical Voronoi vertices,
equivalently the outward normals of the sample convex-hull facets. It is
eta = 0.3021109143286826, after which G is exactly zero within the domain.

## Segment

Let a=sqrt(3) and ell=a/16. Projection gives clipped intervals of radius
r about the equally spaced cell midpoints. Uniform one-dimensional
integration gives the exact formula

`G(r)^2 = ((ell-r)_+)^3/(3 ell)`.

It vanishes at eta = 0.028030628089236242. The formula is independently
checked by direct one-dimensional quadrature.

## Reproduction and normalization

Run `python compute_diversity.py`. Bundled inputs make this independent
of the original repository checkout. Optionally pass `--repository PATH`
to verify source-design hashes and the frozen empirical normalizations
against that repository. Both training arrays also match the original
`synthetic_metadata.json` hashes.

The baseline SN values deliberately remain the original 512-point
assignment estimates. In general G(0) permits arbitrary mass allocation,
so it need not equal the equal-weight empirical W2. A separate small
discretization effect occurs for the segment: its exact continuum G(0)
is 1/16, which divided by the frozen estimated SN is 1.000488639. This is
not a failure of the continuum lower-bound relation. The plotted grid
starts at eta=.008 and does not display that diagnostic endpoint.
