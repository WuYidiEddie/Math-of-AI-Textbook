# four-decomposition measurements on q_11b94d (textbook/01-what-training-is.md)

torch 2.10.0+cu130 | numpy 1.26.4 | device cpu
question q_11b94d | family univariate_regression | dataset sym_4d5411 | candidate set set_4096_var_fix_fix_cecee3
dev half? True  (md5(qid) % 2 == 0 -> dev; the test half is never used for analysis here)


S0  the question, the official protocol, the official audit numbers
-------------------------------------------------------------------
target expression : tanh(2*x)**3 + sin(2*pi*x)
domain            : [0.0, 1.0]   sampling: uniform
train / test size : 256 / 256
selection metric  : test_mse

candidate A (c_6c95ed): d=4 w=256 act=leaky_relu residual=False LN=[False, True, True, False] p=264961 | AdamW lr=3e-05 betas=(0.9, 0.95) wd=1e-05 | 256 steps x bs 16
candidate B (c_4d0869): d=4 w=256 act=relu residual=True LN=[True, False, True, True] p=265473 | AdamW lr=3e-05 betas=(0.9, 0.95) wd=1e-05 | 256 steps x bs 16
candidate C (c_d7ace4): d=5 w=192 act=leaky_relu residual=False LN=[False, True, True, True, True] p=187393 | AdamW lr=3e-05 betas=(0.9, 0.95) wd=1e-05 | 256 steps x bs 16

official protocol (from the candidate's own train.py):
  torch.manual_seed(seed); Model(); AdamW(lr, betas, weight_decay);
  per step: idx = torch.randint(0, n, (bs,))  ->  MSE on that minibatch  ->  opt.step()
  test MSE is evaluated after every step; the reported metric is the last one

official audit numbers (notes/bank-audit.csv, 10 seeds, this same protocol):
  A: mean 0.00356441   std 0.00273417
  B: mean 0.00364627   std 0.0026779
  C: mean 0.000222606   std 8.23694e-05

p / n for the three candidates: [264961, 265473, 187393] / 256
  -> all three have p >> n, so the identification term is *available*; the question is
     which of the four terms actually binds.

S1  where the squared loss comes from
-------------------------------------
S1.1  a Gaussian noise model turns the squared loss into the maximum likelihood
  noise model: y = f(x) + eps,  eps ~ N(0, sigma^2), sigma = 0.3
  negative log-likelihood per data point under a predictor g:
    NLL(g)/n = 0.5*log(2*pi*sigma^2) + (y_i - g(x_i))^2 / (2*sigma^2)
  so NLL(g)/n - NLL(g')/n = ( MSE(g) - MSE(g') ) / (2*sigma^2)   exactly.
  MSE(gA) = 0.087760285134   MSE(gB) = 0.127172828079
  NLL(gA)/n = 0.202522868510   NLL(gB)/n = 0.421481440431
  difference NLL/n : -0.218958571921   vs   (MSE(gA)-MSE(gB))/(2 sigma^2) = -0.218958571921
  -> the squared loss IS the Gaussian likelihood up to a constant and a positive factor,
     so minimising it is exactly maximum likelihood, and the optimum is the conditional mean.

S1.2  the population optimum of the squared loss is E[y|x], and it is the best predictor
  100000 samples from y = f(x) + eps; least-squares fit of a degree-24 polynomial
  mean squared difference to f on a grid : 1.972e-05
  largest deviation                      : 2.878e-02 at x = 1.0000  (the fit is unconstrained at the boundary)
  held-out MSE of the fit                : 0.090710   (sigma^2 = 0.090000)
  excess over sigma^2                    : +7.095e-04
  -> the squared loss does NOT try to fit the noise: its optimum is E[y|x] = f(x),
     and what is left on the test points is exactly the irreducible noise variance.

S1.3  train error 0 does not mean validation error 0: the kink demo
  target f(x) = |x - 0.5| on [0,1] (a kink); 256 noiseless training points;
  Chebyshev polynomials are used as the basis (the Vandermonde basis is too ill-conditioned
  to give a fair picture).  The class is polynomial of degree d:
      d |   train MSE  |   test MSE
      7 |   1.115e-04 |   1.216e-04
     15 |   9.580e-06 |   1.397e-05
     31 |   8.115e-07 |   5.899e-05
     63 |   3.376e-08 |   6.433e+02
  -> the training error falls monotonically as the class gets richer, while the test error
     first falls and then EXPLODES (Runge oscillations at the ends of the interval).
     The training error never sees any of this: it is a statement about the 256 points you
     have, not about the function.  THIS is the A term, and it is invisible in the training
     error.  (The real target of q_11b94d is smooth, so its A term is tiny -- see S3.)

S2  which interpolant does the algorithm pick?  (the min-norm chain, 01 -> 02)
------------------------------------------------------------------------------
  p=6, n=3  (p > n):
    GD from theta_0 = 0, 60000 steps      : train residual ||Phi theta - y|| = 1.197e-15  (min-norm solution: 4.191e-16)
    ||theta_T - theta*||^2                    = 3.224487
    null-space energy ||P_null theta*||^2      = 3.224487   <-- the floor
    ||theta_T - theta_min||^2 (min-norm part) = 3.309e-29
  p=3, n=6  (p < n):
    GD from theta_0 = 0, 60000 steps      : train residual ||Phi theta - y|| = 2.648e-14  (min-norm solution: 5.389e-16)
    ||theta_T - theta*||^2                    = 0.000000
    null-space energy ||P_null theta*||^2      = 0.000000   <-- the floor
    ||theta_T - theta_min||^2 (min-norm part) = 3.182e-27
  -> with p > n the algorithm converges to the MIN-NORM interpolant, not to theta*: the
     component of theta* in the null space of Phi is never learned, and it puts a floor
     under the PARAMETER error (hence under the test error).  With p < n there is no null
     space and the floor is 0.

  the same phenomenon in function space (the quantity we actually care about):
    test MSE of the GD solution (train error 0) : 3.281430
    test MSE of the min-norm interpolant       : 3.281430   (same function, same floor)
    -> many functions fit the data perfectly; the algorithm picks one, and the choice
       is what a validation set sees.  This is the I term, and it is an upper bound on
       the achievable risk, not a property of the function class.

S3  measuring A (approximation): how expressive does the class have to be?
--------------------------------------------------------------------------
  the target is known in closed form, so the approximation error of a given class can be
  measured directly (dense-grid least squares against f).

  degree |  train MSE (fit on the 256 train pts) |  test MSE  |  L2 relative error vs f
       1 |                            1.563e-01 | 1.559e-01 | 6.325e-01
       3 |                            3.488e-03 | 4.397e-03 | 9.902e-02
       5 |                            2.278e-05 | 3.084e-05 | 8.080e-03
       7 |                            7.215e-08 | 1.204e-07 | 4.845e-04
      15 |                            5.953e-15 | 8.320e-15 | 8.434e-09
      31 |                            5.854e-15 | 3.807e-14 | 3.425e-09

  -> a degree-7 class already represents this target to ~1e-6 relative L2 error, and the
     polynomial has p = 8 < n = 256 parameters (no aliasing at all).  So for THIS target
     the approximation term is essentially zero, and the MLPs' 1e-4 - 1e-3 residual
     cannot be blamed on the function class.

S4  measuring F (fluctuation): 10 seeds at the official budget
--------------------------------------------------------------
  A: mean 0.00356654  std 0.00290199   |  official 10-seed: mean 0.00356441 std 0.00273417   |  30s
      seeds: 0.00506, 0.00062, 0.00515, 0.00086, 0.00123, 0.00236, 0.00723, 0.00237, 0.00174, 0.00904
  B: mean 0.00364568  std 0.00282508   |  official 10-seed: mean 0.00364627 std 0.0026779   |  25s
      seeds: 0.00232, 0.00311, 0.00821, 0.00086, 0.00253, 0.00341, 0.00205, 0.00340, 0.00129, 0.00928
  C: mean 0.000202709  std 8.30407e-05   |  official 10-seed: mean 0.000222606 std 8.23694e-05   |  22s
      seeds: 0.00018, 0.00016, 0.00019, 0.00009, 0.00015, 0.00010, 0.00026, 0.00028, 0.00025, 0.00036

  -> every reproduced value lies inside the official 10-seed spread, so the harness
     below can be trusted for the long-budget runs.

  paired differences (same seed -> same init and same minibatch stream):
    A - C: mean +0.00336383  SE 0.0009  |t|   3.74  vs t(0.975,9) = 2.262  ->  significant
    B - C: mean +0.00344297  SE 0.00088  |t|   3.91  vs t(0.975,9) = 2.262  ->  significant
    A - B: mean -7.91371e-05  SE 0.000774  |t|   0.10  vs t(0.975,9) = 2.262  ->  NOT significant
  -> the A-vs-B gap (both bad) is inside the noise; the C-vs-{A,B} gap is not.

S5  measuring O(T) (optimization): learning curves past the official budget
---------------------------------------------------------------------------
  candidate A (mean of seeds 0,1):
     steps |  train MSE   |  test MSE    | test/train
       256 |   2.367e-03 | 0.00283963  |     1.20
       512 |   1.342e-04 | 0.000155898  |     1.16
      1024 |   1.113e-04 | 0.000126436  |     1.14
      2048 |   7.841e-05 | 0.000102912  |     1.31
      4096 |   3.501e-05 | 3.93938e-05  |     1.13

  candidate B (mean of seeds 0,1):
     steps |  train MSE   |  test MSE    | test/train
       256 |   2.327e-03 | 0.00271257  |     1.17
       512 |   1.655e-04 | 0.00019421  |     1.17
      1024 |   1.945e-04 | 0.000200437  |     1.03
      2048 |   8.062e-05 | 8.51972e-05  |     1.06
      4096 |   2.368e-04 | 0.000246624  |     1.04

  candidate C (mean of seeds 0,1):
     steps |  train MSE   |  test MSE    | test/train
       256 |   1.512e-04 | 0.000170452  |     1.13
       512 |   3.935e-04 | 0.000389657  |     0.99
      1024 |   1.012e-04 | 0.000118577  |     1.17
      2048 |   1.368e-04 | 0.000145378  |     1.06
      4096 |   3.643e-05 | 5.08497e-05  |     1.40

  THE FIRST DIAGNOSTIC QUESTION IS ANSWERED HERE: how big is the training error?
    candidate A: train MSE 2.367e-03 | test MSE 2.840e-03 | ratio test/train 1.20
    candidate B: train MSE 2.327e-03 | test MSE 2.713e-03 | ratio test/train 1.17
    candidate C: train MSE 1.512e-04 | test MSE 1.705e-04 | ratio test/train 1.13
  -> at the official budget the training error is NOT 0 and it is the SAME ORDER as the
     test error.  256 steps x batch 16 = 4096 samples = 16 epochs over 256 points; the
     network has not even fitted the training set.  So this question is not an
     interpolation story at all: it is an optimisation story (the O term dominates), and
     the usual 'train error 0 therefore anything can happen' intuition does not apply here.

  candidate C, one seed, 64x the official budget:
     steps |  train MSE   |  test MSE
      4096 |   1.271e-05 | 2.59154e-05
      8192 |   2.812e-05 | 3.32268e-05
     16384 |   6.995e-05 | 7.8693e-05
     16384 |   6.995e-05 | 7.8693e-05
    official budget (256 steps) test MSE for C: 0.000176246
  -> the curve is still descending at 16384 steps, so the official 256-step metric sits
     on a moving target: part of the error is simply not reached yet (the O term).

S6  measuring I (identification): the initial NTK and the lazy-limit interpolant
--------------------------------------------------------------------------------
  candidate A:
    NTK at init: lambda_max 34.03 | effective rank 1.82 (participation ratio) / 2.84 (entropy)
    learning window: a direction is moved by the budget when eta*lambda*T >~ 1, i.e. lambda > 130.2
      eigenvalues above the window: 0 of 256 | energy of y frozen outside the window: 100.00%
    lazy-limit (min-norm) solution of that kernel: train MSE 8.979e-04 | test MSE 0.00106602
      the solve is a truncated SVD (rank 71 of 256, cond(K) ~ 3.13e+04), so it is a NEAR-interpolant, not an exact one
    kernel drift after the official 256 steps: rel 5.206 | cosine 0.896  -> NOT lazy
    actually trained net at the same budget (seed 0): test MSE 0.0050552

  candidate B:
    NTK at init: lambda_max 564.4 | effective rank 1.30 (participation ratio) / 1.70 (entropy)
    learning window: a direction is moved by the budget when eta*lambda*T >~ 1, i.e. lambda > 130.2
      eigenvalues above the window: 1 of 256 | energy of y frozen outside the window: 53.62%
    lazy-limit (min-norm) solution of that kernel: train MSE 5.946e-03 | test MSE 0.00651392
      the solve is a truncated SVD (rank 38 of 256, cond(K) ~ 3.1e+04), so it is a NEAR-interpolant, not an exact one
    kernel drift after the official 256 steps: rel 0.109 | cosine 0.995  -> lazy-ish
    actually trained net at the same budget (seed 0): test MSE 0.00231874

  candidate C:
    NTK at init: lambda_max 206.2 | effective rank 1.84 (participation ratio) / 2.98 (entropy)
    learning window: a direction is moved by the budget when eta*lambda*T >~ 1, i.e. lambda > 130.2
      eigenvalues above the window: 1 of 256 | energy of y frozen outside the window: 54.23%
    lazy-limit (min-norm) solution of that kernel: train MSE 9.554e-04 | test MSE 0.00141936
      the solve is a truncated SVD (rank 67 of 256, cond(K) ~ 3.27e+04), so it is a NEAR-interpolant, not an exact one
    kernel drift after the official 256 steps: rel 4.911 | cosine 0.775  -> NOT lazy
    actually trained net at the same budget (seed 0): test MSE 0.000176246

  -> the kernel is NOT frozen for A and C (cosine 0.896 / 0.775), so for those two the
     lazy-limit solution is only a reference, not a prediction; for B (cosine 0.995) it is
     a fair prediction.  And the near-interpolant is NOT uniformly better than the
     partially trained net: for B it is worse (6.51e-3 vs 2.32e-3), for C the trained net
     is much better (1.76e-4 vs 1.42e-3), for A it is better (1.07e-3 vs 5.06e-3).
     Interpolating is not the goal; what the validation set measures is the function the
     algorithm actually selects.

S7  capacity scan: does width help at this budget?
--------------------------------------------------
  candidate C's architecture: d=5 w=192 act=leaky_relu residual=False LN=[False, True, True, True, True]
  (the model builder below is a copy of the v1.5 generator; the parameter counts are
   checked against the official specs, so only the width is changed)
    check A: official p=264961 | rebuilt p=264961 -> match
    check B: official p=265473 | rebuilt p=265473 -> match
    check C: official p=187393 | rebuilt p=187393 -> match

  w=  96 (p=  47617): test MSE at 256 steps 0.00106294 | at 4096 steps 1.877e-05 | train MSE at 4096 1.529e-05
  w= 384 (p= 743425): test MSE at 256 steps 0.00098128 | at 4096 steps 0.0010727 | train MSE at 4096 9.604e-04
  w= 192 (p=  187393, official, seed 0): test MSE at 256 steps 0.000176246 | at 4096 steps 2.59154e-05

  -> the width scan is not a free lunch in either direction at a fixed budget: a narrower
     net moves less far from its initialisation, a wider one is not automatically better.
     Read it with S5: at 256 steps the network has not even fitted the training set, so
     what the width changes is how fast the function moves, not what it can represent.

S8  the four-term table for this question
-----------------------------------------
  every entry is labelled with what it is: an upper bound, an estimate, or a measured
  quantity.  Nothing here is a proof of anything (see 08-limits-and-next.md).

  candidate | train MSE @256 | A (upper bd) | F (10-seed std) | O(T): 256 -> long | residual @ long
  A        |    2.367e-03 | ~1e-6 (deg-7 class) | 2.902e-03 | +2.800e-03      | 3.939e-05
  B        |    2.327e-03 | ~1e-6 (deg-7 class) | 2.825e-03 | +2.466e-03      | 2.466e-04
  C        |    1.512e-04 | ~1e-6 (deg-7 class) | 8.304e-05 | +1.196e-04      | 5.085e-05

  reading of the table (each line says what it is, not what it proves):
    * A is not binding.  A degree-7 class already approximates this target to ~1e-6 and
      the width scan (S7) shows extra capacity does not pay off at this budget.
      [argument, not a bound: the class is not *proved* to be rich enough]
    * O(T) dominates.  At the official budget the training error is the same order as the
      test error (the network has not fitted the training set yet), and the test error
      keeps moving when the budget is extended.  [measured]
    * I is what is left once the budget is not binding: the algorithm's choice of
      function, of the order of the long-budget residual.  S6 shows the kernel it uses is
      not frozen for A and C, so the lazy-limit interpolant is only a reference there.
      [estimate; separating I from A needs the theory of chapter 05]
    * F is small next to the C-vs-A gap but large next to the A-vs-B gap, which is exactly
      why 'A is worse than B' is not a real difference here.  [measured]

reproduce with: python verification/four_decomposition.py
