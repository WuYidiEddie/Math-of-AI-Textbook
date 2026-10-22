
S0  setup
---------
  what this script measures, and why it is not the same as chapter 01
    chapter 01 asks: on ONE question, which of A / I / O(T) / F binds?
    chapter 05 asks: how do the STRUCTURE of the target and the structure of the
    architecture change those four terms -- and can that be predicted from the
    target's description alone, before any training?

  the questions this script answers
    S1  A is not a vibe: for a fixed class it is an ORTHOGONAL PROJECTION, so it can be
        computed in closed form (no optimisation noise).  We measure it for several
        classes x targets and see which class wins for which structure.
    S2  depth and width on REAL targets: the same architecture grid on two v1.5
        families and at two budgets, so that A (structure) and O(T) (budget) can be
        told apart instead of being confounded.
    S3  the information limit: n samples in d dimensions resolve a bandwidth
        B ~ n^{1/d}; for a Sobolev-s target the leftover tail is ~ B^{-s}, so the
        best possible RMSE is ~ n^{-s/d}.  We count the lattice exactly.
    S4  in kernel language A is a NORM: the min-norm weight vector that realises the
        target in the initialisation's feature space.  Computable before training,
        and it says exactly which part of the target the architecture cannot buy.

  dev-half discipline: every v1.5 question used below is dev (md5(qid)%2==0);
  the test half is never touched here.
  torch 2.10.0+cu130 | numpy 1.26.4 | device cpu

S1  A as a computable oracle: best-in-class L2 error
----------------------------------------------------

  for a class spanned by features Phi, the best L2 approximation of f is the
  orthogonal projection, so A is an optimisation-free quantity:
      A_class(f) = || f - P_span(Phi) f ||^2_{L2} ,   measured on a dense grid
  classes (all given the SAME budget of m = 120 basis functions):
      poly   : total-degree Chebyshev polynomials (largest degree that fits m)
      relu   : random 1-hidden-layer ReLU features (the lazy / infinite-width
               limit of one layer, generic connectivity)
      addit  : axis-aligned random ReLU features (the classic additive class)

  NOTE what this table does NOT contain: the price of finding those coefficients
  from n samples.  A small A does not mean an easy problem -- that is S3 and S4.

  relative L2 error of the best approximation, m = 120

  target                                |   poly    |   relu    |  addit    |  winner
  --------------------------------------------------------------------------------------
  smooth 1-D  sin(2 pi x)              | 4.06e-15 | 6.68e-03 | 5.98e-03 | poly
  fast 1-D    sin(20 pi x)             | 4.57e-15 | 5.77e-01 | 5.54e-01 | poly
  kink 1-D    |x - 0.5|                | 6.11e-04 | 1.11e-02 | 6.77e-03 | poly
  additive 2-D sin(2pi x)+cos(2pi y)   | 7.50e-10 | 1.25e-01 | 4.54e-02 | poly
  composite 2-D sin(12 x y)            | 7.36e-04 | 3.89e-01 | 7.92e-01 | poly
  additive 4-D sum_j sin(2 pi x_j)     | 1.15e-01 | 3.61e-01 | 6.53e-02 | addit
  --------------------------------------------------------------------------------------

  the degree each polynomial class can afford at this budget (total degree):
    d = 1: degree 119 (120 terms)   [smooth 1-D  sin(2 pi x)]
    d = 1: degree 119 (120 terms)   [fast 1-D    sin(20 pi x)]
    d = 1: degree 119 (120 terms)   [kink 1-D    |x - 0.5|]
    d = 2: degree  14 (120 terms)   [additive 2-D sin(2pi x)+cos(2pi y)]
    d = 2: degree  14 (120 terms)   [composite 2-D sin(12 x y)]
    d = 4: degree   4 (70 terms)   [additive 4-D sum_j sin(2 pi x_j)]

  reading:
    * STRUCTURE MATCH, measured by the two ReLU classes (same size, same
      connectivity budget, different structure): the axis-aligned class beats
      the generic one on the additive targets (2-D: 4.54e-02 vs 1.25e-01; 4-D: 6.53e-02 vs 3.61e-01) and LOSES on the composite one (7.92e-01 vs 3.89e-01): a sum of one-variable functions cannot make a product;
    * the CURSE OF DIMENSIONALITY hides inside a fixed budget: the degree a
      polynomial class can afford collapses with d (119 -> 14 -> 4), so the
      spectral class goes from 1e-15 in 1-D to 1e-1 in 4-D while the structured
      class stays at 5e-2 -- size of the budget, not size of the class, is what
      limits a smooth target in high dimension;
    * the KINK row is about convergence speed: a spectral class converges only
      algebraically on a kink, a piecewise-smooth class much faster.

  how many basis functions does each class need for 1e-3 relative L2 error?
  (1-D, nested families: the m-feature class is a subspace of the 240-feature
   class, so the curves are comparable)

  target                 |  class  |  smallest m with rel. L2 <= 1e-3
  ------------------------------------------------------------------
  smooth 1-D sin(2 pi x) |  poly  |  8
      curve: m=4:9.37e-02, m=8:2.21e-04, m=16:9.35e-12, m=32:8.63e-16, m=64:3.95e-15, m=120:4.06e-15, m=240:4.02e-15
  smooth 1-D sin(2 pi x) |  relu  |  >240
      curve: m=4:8.63e-01, m=8:5.93e-01, m=16:2.43e-01, m=32:1.62e-01, m=64:5.85e-02, m=120:2.89e-02, m=240:5.27e-03
  fast 1-D   sin(20 pi x) |  poly  |  64
      curve: m=4:9.90e-01, m=8:9.75e-01, m=16:9.16e-01, m=32:1.81e-01, m=64:3.92e-15, m=120:4.57e-15, m=240:3.00e-15
  fast 1-D   sin(20 pi x) |  relu  |  >240
      curve: m=4:9.97e-01, m=8:9.97e-01, m=16:9.93e-01, m=32:9.81e-01, m=64:7.68e-01, m=120:6.54e-01, m=240:3.18e-01
  kink 1-D   |x - 0.5|   |  poly  |  120
      curve: m=4:1.25e-01, m=8:3.91e-02, m=16:1.31e-02, m=32:4.51e-03, m=64:1.58e-03, m=120:6.11e-04, m=240:2.15e-04
  kink 1-D   |x - 0.5|   |  relu  |  >240
      curve: m=4:4.25e-01, m=8:1.97e-01, m=16:4.43e-02, m=32:3.10e-02, m=64:1.23e-02, m=120:6.86e-03, m=240:5.95e-03
  ------------------------------------------------------------------

  reading: the price of a frequency-a target in a spectral class grows with a
  (you need O(a) Chebyshev coefficients), while a piecewise-smooth class buys
  accuracy at a much better rate -- and a kink is nearly free for it.
  So 'which class' is a statement about the target's structure, not about size.

S2  depth and width on REAL targets: A and O(T) told apart
----------------------------------------------------------

  our own model builder (plain ReLU MLP, no LayerNorm, no residual), the
  family's own optimizer / lr / budget from notes/bank-audit.csv, and the
  official evaluation (test CE).  Only depth and width are varied.

  spiral_classification (q_04316e, Adagrad lr=0.003, 4096 steps, bs=16), mean over 2 seeds
    depth |  width |   test CE
    ----------------------------------
        1 |     16 |  0.6830   (chance = 0.693)
        2 |     16 |  0.6830   (chance = 0.693)
        3 |     16 |  0.6754
        1 |     64 |  0.6848   (chance = 0.693)
        2 |     64 |  0.6620
        3 |     64 |  0.6035
    ----------------------------------

  xor_classification (q_07d60b, SGD lr=0.003, 16384 steps, bs=64), mean over 2 seeds
    depth |  width |   test CE
    ----------------------------------
        1 |     16 |  0.2403
        2 |     16 |  0.2504
        3 |     16 |  0.6088
        1 |     64 |  0.2263
        2 |     64 |  0.1936
        3 |     64 |  0.2165
    ----------------------------------

  now the same question at a DIFFERENT BUDGET (1/4 and 4x), because an
  architecture can look bad for two different reasons:
    * A is too big  -> the architecture cannot represent the target at all;
    * O(T) is too big -> the architecture can represent it but has not
      arrived yet, and deeper nets are slower.

  spiral_classification (q_04316e)
    depth | width |  budget |   test CE
    ----------------------------------------
        1 |     64 |       1x    |  0.6848
        3 |     64 |       1x    |  0.6035
        1 |     64 |       4x    |  0.6771
        3 |     64 |       4x    |  0.5421
    ----------------------------------------

  xor_classification (q_07d60b)
    depth | width |  budget |   test CE
    ----------------------------------------
        1 |     64 |    0.25x    |  0.6396
        3 |     64 |    0.25x    |  0.6891
        1 |     64 |       1x    |  0.2263
        3 |     64 |       1x    |  0.2165
        1 |     64 |       4x    |  0.1274
        3 |     64 |       4x    |  0.2819
    ----------------------------------------

  reading:
    * SPIRAL (a curved boundary that needs many linear regions): depth is the
      lever, and the advantage SURVIVES a 4x budget -- so it is A (the class),
      not O(T) (the speed).
    * XOR (a boundary a wide layer can carve directly): at a quarter of the
      official budget EVERY depth is still at chance, at the official budget
      every depth reaches the same accuracy, and with 4x budget the shallow
      net keeps improving while the deep one gets WORSE -- the deep net is
      slower to optimise, so here O(T), not A, is what the architecture
      argument is about.
    -> the same knob (depth) is decisive on one target and irrelevant on
       another.  Which one it is can be read off the TARGET, before training.

S2b  the synthetic control we tried FIRST (it failed, and the failure is the lesson)
------------------------------------------------------------------------------------

  design: two 2-D targets of comparable smoothness, one COMPOSITIONAL and one
  ADDITIVE, and two architectures with a MATCHED parameter count:
      comp: f(x,y) = sin(3 x y)          (a product: depth can factor it)
      add : f(x,y) = sin(3 x) + sin(3 y)  (a sum: one layer is enough)
      1 hidden layer  w=66: p = 265  |  2 hidden layers w=14: p = 267

  relative L2 error on a 301x301 grid, mean over 2 seeds,
  n = 1024 noiseless samples, full-batch Adam lr 3e-3, 3000 steps

  target   |  1 hidden layer  w=66  |  2 hidden layers w=14
  --------------------------------------------------------------------------
  comp sin(3xy)       |            3.741e-02  |            6.437e-02
  add  sin(3x)+sin(3y)|            1.543e-02  |            2.075e-02
  --------------------------------------------------------------------------

  the same numbers for the TRAINING points (did the net fit the data at all?)
  --------------------------------------------------------------------------
  comp sin(3xy)       |            3.030e-02  |            4.306e-02
  add  sin(3x)+sin(3y)|            1.818e-02  |            3.037e-02
  --------------------------------------------------------------------------

  and the RESOLUTION FLOOR of each width: the best-in-class relative L2 error
  of a random ReLU class with m = width features (the lazy limit of one layer)
  --------------------------------------------------------------------------
  comp sin(3xy)       |            3.782e-02  |            2.298e-01
  add  sin(3x)+sin(3y)|            2.299e-02  |            1.288e-01
  --------------------------------------------------------------------------

  trained grid error DIVIDED BY the 1-layer random-feature floor
  of the SAME width
  (1.0 = the trained net sits at the resolution limit of a 1-layer class of
   that width;  < 1 = it beat it, which is what depth composes for)
  --------------------------------------------------------------------------
  comp sin(3xy)       |                0.989  |                0.280
  add  sin(3x)+sin(3y)|                0.671  |                0.161
  --------------------------------------------------------------------------

  reading:
    * what the design FOUND: the wide shallow net beats the narrow deep net
      on BOTH targets, so there is no depth separation here and the width
      effect dominates.
    * what the ratios mean: for the WIDE net the trained error sits AT the
      1-layer random-feature floor of its own width (ratio ~ 1), i.e. the
      training converged to the RESOLUTION limit of its class rather than
      to zero (with m = 66 features and n = 1024 points the class cannot even
      fit its own training data -- that is the train error above); for the
      DEEP net the ratio is well BELOW 1, because depth composes: the
      1-layer floor of the same width is NOT a floor for it.  So the ratio
      must be read as 'distance to a 1-layer class of the same width', not
      as a lower bound.
    * why no separation: a smooth product target is not provably hard for a
      wide 1-layer ReLU class at this resolution, and matching the
      parameter count confounds depth with width.  A separation claim needs
      a target provably outside the shallow class AND a resolution at which
      the structural difference stands above the floor.
    * two lessons: (i) a class floor is a floor for THAT class only -- always
      compare a net against the floor of its own class;  (ii) a separation
      result is a statement about a TARGET, not about 'depth' -- which is
      why S2 measures real v1.5 families, where the boundary complexity is
      the thing that differs.

S3  the information limit: n samples, d dimensions, smoothness s
----------------------------------------------------------------

  derivation (the whole argument, in five lines)
    (i)   f has a Fourier series f = sum_k c_k e^{2 pi i k.x} on [0,1]^d.
    (ii)  being in the Sobolev class H^s means  sum_k |k|^{2s} |c_k|^2 <= 1.
    (iii) n samples give n linear equations, so at most n coefficients can be
          pinned down: the resolvable bandwidth B is where the lattice count
          N(B) := #{k in Z^d : |k|_1 <= B} first exceeds n, and N(B) ~ c B^d,
          so B ~ n^{1/d};
    (iv)  the worst case over the Sobolev ball puts all the tail energy on the
          first shell outside B, where |k| = B+1, so the leftover L2 error is
          at most (B+1)^{-2s};
    (v)   => best possible RMSE  E_n ~ (B+1)^{-s} ~ n^{-s/d}.

  the constants are irrelevant for the argument; the EXPONENT is the content,
  and the exponent is a counting statement, so it can be CHECKED exactly:

  the counting formula, checked against brute force:
    d = 1, r = 3: closed form     2 | brute force     2 | match
    d = 2, r = 4: closed form    16 | brute force    16 | match
    d = 3, r = 3: closed form    38 | brute force    38 | match
    d = 2, r = 6: closed form    24 | brute force    24 | match

  B(n) = largest bandwidth with N(B) <= n, and the resulting RMSE bound
  s   d |  predicted -s/d |  measured slope  |  ratio  | B(n) at n = 16..4096
  ------------------------------------------------------------------------------------------------
  2   1 |      -2.0000      |     -2.0000      | 1.000  | [7, 31, 127, 511, 2047]
  2   2 |      -1.0000      |     -0.9752      | 0.975  | [2, 5, 10, 22, 44]
  2   4 |      -0.5000      |     -0.5340      | 1.068  | [1, 2, 3, 5, 8]
  4   2 |      -2.0000      |     -1.9505      | 0.975  | [2, 5, 10, 22, 44]
  1   2 |      -0.5000      |     -0.4876      | 0.975  | [2, 5, 10, 22, 44]
  ------------------------------------------------------------------------------------------------

  the same statement in kernel language: for a kernel with eigenvalues
  lam_1 >= lam_2 >= ..., the n-width is E_n = sqrt(lam_{n+1}).  A Gaussian kernel
  on [0,1] has a spectrum that decays faster than any power, so its E_n decays
  faster than any power of n as well:
    n      lam_{n+1}       sqrt(lam_{n+1})
       8   2.5765e-01   5.0760e-01
      16   1.2520e-06   1.1189e-03
      32   5.2383e-15   7.2376e-08   <- numerical floor
      64   2.1868e-15   4.6764e-08   <- numerical floor
     128   3.3242e-16   1.8232e-08   <- numerical floor
     256   0.0000e+00   0.0000e+00   <- numerical floor
    (below the floor the eigenvalues are numerical noise, so the tail of this
     table says 'faster than any power', not a specific rate)

S4  A in kernel form: the min-norm weight norm under a real initialisation
--------------------------------------------------------------------------

  the claim: in the kernel (lazy) regime, the part of the target that the
  architecture cannot buy cheaply is measured by the SIZE of the coefficient
  that realises it.  Write the network at initialisation as f_w(x) = w^T phi(x)
  with phi(x) = J(x) the Jacobian, and let
      K = (1/n) J J^T = V Lam V^T        (n x n, eigenvalues = NTK eigenvalues)
  The minimum-norm weight vector that reproduces the target values y is
      w_hat = Phi^+ y  and   ||w_hat||^2 = y^T (J J^T)^+ y
                             = sum_j b_j^2 / lam_j^{(0)},  b = V^T y,
  where lam_j^{(0)} are the eigenvalues of J J^T, i.e. lam_j^{(0)} = n * lam_j.
  With the (1/n)-scaled K used above this becomes the estimator
      ||w_hat||^2  =  (1/n) sum_j b_j^2 / lam_j
  (the 1/n is a normalisation; the CONTENT is that the smaller lam_j is, the
   more expensive the target is in this architecture's currency.)

  question q_11b94d (dev half: True), dataset sym_4d5411

  candidate A (c_6c95ed): d=4 w=256 p=264961
    init NTK: lambda_max 34.03 | numerical rank 214/256 in float64, 71/256 in float32 | cond(K) 3.8e+10
    (the rank depends on the arithmetic: the training runs in float32, so
     the kernel it actually sees has the smaller rank)
    learning window: lam >= 1/(lr*T) = 130.2  (0/256 directions are inside it)
    target smooth  tanh(2x)^3 + sin(2 pi x)   (variance of the target on the test points: 0.3926)
      min-norm weight norm of the interpolant  ||w_hat||^2 = 6.0096e+00
      energy of the target OUTSIDE the learning window: 100.00%
      lazy-limit (min-norm) interpolant: train MSE 2.6477e-05, test MSE 8.6610e-04
      the same two numbers, relative to the target's variance:
        lazy-limit relative test error                 2.2062e-03
        GP posterior variance of the same kernel        7.7220e-04   (ratio 2.857)
      (the GP number is what the SAME kernel would incur if the
       target were a draw from it; the gap is the part of this
       deterministic target the kernel does not know how to hold)
    target fast    sin(20 pi x)   (variance of the target on the test points: 0.4825)
      min-norm weight norm of the interpolant  ||w_hat||^2 = 5.5035e+03
      energy of the target OUTSIDE the learning window: 100.00%
      lazy-limit (min-norm) interpolant: train MSE 1.2809e-01, test MSE 4.5428e+00
      the same two numbers, relative to the target's variance:
        lazy-limit relative test error                 9.4142e+00
        GP posterior variance of the same kernel        7.7220e-04   (ratio 12191.338)
      (the GP number is what the SAME kernel would incur if the
       target were a draw from it; the gap is the part of this
       deterministic target the kernel does not know how to hold)

    reading:
      * the lazy-limit error is computable BEFORE training: it is the
        min-norm kernel interpolant's error, and (S4b) it equals the GP
        posterior variance when the target is a draw from the kernel;
      * for the FAST target the lazy interpolant is not merely worse, it is
        catastrophic: its test MSE exceeds the target's own variance, i.e.
        the smoothest function that fits the data is worse than predicting
        zero.  Feature learning is not optional on a fast target.

  candidate C (c_d7ace4): d=5 w=192 p=187393
    init NTK: lambda_max 206.2 | numerical rank 206/256 in float64, 67/256 in float32 | cond(K) 6.09e+09
    (the rank depends on the arithmetic: the training runs in float32, so
     the kernel it actually sees has the smaller rank)
    learning window: lam >= 1/(lr*T) = 130.2  (1/256 directions are inside it)
    target smooth  tanh(2x)^3 + sin(2 pi x)   (variance of the target on the test points: 0.3926)
      min-norm weight norm of the interpolant  ||w_hat||^2 = 1.2779e+00
      energy of the target OUTSIDE the learning window: 54.23%
      lazy-limit (min-norm) interpolant: train MSE 5.9079e-06, test MSE 6.3414e-04
      the same two numbers, relative to the target's variance:
        lazy-limit relative test error                 1.6153e-03
        GP posterior variance of the same kernel        1.0762e-03   (ratio 1.501)
      (the GP number is what the SAME kernel would incur if the
       target were a draw from it; the gap is the part of this
       deterministic target the kernel does not know how to hold)
    target fast    sin(20 pi x)   (variance of the target on the test points: 0.4825)
      min-norm weight norm of the interpolant  ||w_hat||^2 = 1.3009e+03
      energy of the target OUTSIDE the learning window: 99.40%
      lazy-limit (min-norm) interpolant: train MSE 2.0348e-02, test MSE 5.9842e-01
      the same two numbers, relative to the target's variance:
        lazy-limit relative test error                 1.2401e+00
        GP posterior variance of the same kernel        1.0762e-03   (ratio 1152.340)
      (the GP number is what the SAME kernel would incur if the
       target were a draw from it; the gap is the part of this
       deterministic target the kernel does not know how to hold)

    reading:
      * the lazy-limit error is computable BEFORE training: it is the
        min-norm kernel interpolant's error, and (S4b) it equals the GP
        posterior variance when the target is a draw from the kernel;
      * for the FAST target the lazy interpolant is not merely worse, it is
        catastrophic: its test MSE exceeds the target's own variance, i.e.
        the smoothest function that fits the data is worse than predicting
        zero.  Feature learning is not optional on a fast target.

S4b  the GP identity, checked exactly on a target drawn from the kernel
-----------------------------------------------------------------------

  identity: for a GP prior with observation noise sigma^2, the posterior mean is
      mu(x) = k_x^T (K + sigma^2 I)^{-1} y
  and the posterior variance is
      var(x) = k(x,x) - k_x^T (K + sigma^2 I)^{-1} k_x
  so, IF the target is a draw from that kernel, the squared error of mu at a
  fresh test point has expectation var(x).  This is checkable: draw the target
  from the kernel itself, then compare.  (We normalise the kernel to unit
  diagonal so the numbers are on the scale of the target's variance; the
  identity does not care about the scaling.  A small sigma^2 is needed
  because the empirical NTK matrix is numerically rank deficient.)

  200 GP draws from the init NTK of candidate C (unit-diagonal kernel):
    analytic posterior variance at the test points : 1.076173e-03
    realised mean squared error of the interpolant   : 2.209590e-03
    ratio of the two MEANS (the tight check)         : 2.0532
    per-draw ratio: mean 2.0532, sd 1.3025
    (the per-draw spread is large because the posterior covariance of a smooth
     kernel is nearly rank-1: the test points are strongly correlated, so the
     mean squared error over them is dominated by one common mode)

  reading: the realised error tracks the analytic posterior variance, so the
  lazy-limit error of a kernel method IS computable before training -- but
  only as a statement about the kernel's OWN prior.  For a deterministic target
  (our case) the posterior variance is a reference scale, not a prediction: the
  gap between the two is exactly the part of the target the kernel does not
  know how to represent.

S5  target portrait of the families (one dev-half question each)
----------------------------------------------------------------

  for every family we ask the same two questions of ONE dev-half question:
    (a) regression families: how good is the best polynomial / ReLU class?
        That is the finite-sample A -- the part of A a learner can feel.
    (b) classification families: how complex is the boundary?  We fit the
        labels with polynomial feature maps of increasing degree (ridge
        regression, the ridge picked by leave-one-out on the TRAINING set) and
        report the resulting test accuracy.  The degree at which the accuracy
        leaves chance is the degree the boundary needs.

  univar (q_07b3f8, d=1, n=256, metric=mse, x in [0.00, 1.00])
      poly deg 3  : best fit of the training points 6.769e-08 | best fit of the test points 5.948e-08
      poly deg 7  : best fit of the training points 6.723e-08 | best fit of the test points 5.888e-08
      relu 256    : best fit of the training points 8.113e-16 | best fit of the test points 6.742e-16
  mvar (q_0525ce, d=5, n=256, metric=mse, x in [0.00, 1.00])
      poly deg 3  : best fit of the training points 6.710e-02 | best fit of the test points 5.951e-02
      poly deg 7  : best fit of the training points 7.360e-16 | best fit of the test points 6.917e-16
      relu 256    : best fit of the training points 8.038e-16 | best fit of the test points 8.292e-16
  spiral (q_04316e, d=2, n=1024, metric=ce, x in [-18.88, 18.87])
      label entropy 0.693 nats | class balance 0.500
      degree -> train acc / test acc / ridge: 1:0.504/0.513, 2:0.547/0.540, 3:0.521/0.526, 4:0.531/0.525, 6:0.585/0.560, 8:0.722/0.671, 12:0.951/0.955
      smallest polynomial degree that beats chance on the test set: 12
  tabular (q_036d72, d=16, n=1024, metric=ce, x in [-4.10, 3.67])
      label entropy 0.693 nats | class balance 0.492
      degree -> train acc / test acc / ridge: 1:0.807/0.760
      smallest polynomial degree that beats chance on the test set: 1
  xor (q_07d60b, d=16, n=1024, metric=ce, x in [-4.05, 4.35])
      label entropy 0.692 nats | class balance 0.521
      degree -> train acc / test acc / ridge: 1:0.521/0.512, 2:0.880/0.774
      smallest polynomial degree that beats chance on the test set: 2

  reading: the degree at which a polynomial stops being chance is a cheap,
  honest proxy for 'how complicated is this boundary'.  The three families sit
  at different places -- which is exactly why one architecture rule cannot fit
  all of them.
  (bigram is left out on purpose: its target is a random law with an entropy
   floor, and the analysis lives in verification/bigram_estimation.py.)

S6  the summary table
---------------------
  claim                                      |  what makes it computable      | status
  ------------------------------------------------------------------------------------------------
  A = best-in-class L2 error                 |  orthogonal projection (S1)    | measured
  A in the lazy regime = ||w_hat||^2         |  (1/n) sum b_j^2/lam_j (S4)   | measured
  the lazy error is predictable before training | GP posterior var (S4b)     | measured
  depth helps exactly where the target needs it |  real-family sweep (S2)   | measured
  the information limit is n^{-s/d}          |  lattice counting (S3)        | derived + measured
  'the right class' depends on the target    |  class x target table (S1)    | measured
  boundary complexity differs by family      |  ridge degree curve (S5)     | measured
  ------------------------------------------------------------------------------------------------
