
S1  the conservation law, optimizer by optimizer (synthetic linear model)
-------------------------------------------------------------------------

  setup: y = Phi theta_true, Phi in R^{n x p} with p > n, theta_0 random (NOT 0).
  the claim to test: is P_null(theta_t) = P_null(theta_0) an ENFORCED constraint,
  or only a consequence of the loss not seeing the null directions?  If the latter,
  any mechanism that CAN see the null space (weight decay, Adam's elementwise
  second moment, moving features) is free to change it.

  p = 200, n = 32, rank(Phi) = 32, dim null(Phi) = 168, T = 3000 steps
  ||P_null theta_0|| = 9.198692   (this is what each optimizer is allowed to change)

  optimizer                          | drift of the null-space projection
  ------------------------------------------------------------------------------------------------
  full-batch GD, no wd               | ||P_null th_T - P_null th_0|| = 6.352e-06   (rel 6.905e-07)   | ||P_null th_T|| = 9.198691   (should be ~ machine eps: exact identity)
  minibatch SGD, no wd, bs=16        | ||P_null th_T - P_null th_0|| = 4.189e-06   (rel 4.554e-07)   | ||P_null th_T|| = 9.198690   (still in row(Phi): row(Phi_B) subset row(Phi))
  full-batch GD, wd=1e-3             | ||P_null th_T - P_null th_0|| = 4.122e-01   (rel 4.482e-02)   | ||P_null th_T|| = 8.786452   (should decay: (1-lr*wd)^T = 0.955184)
  Adam, no wd                        | ||P_null th_T - P_null th_0|| = 5.405e+00   (rel 5.876e-01)   | ||P_null th_T|| = 10.936105   (elementwise 2nd moment leaks out of row(Phi))
  AdamW, wd=1e-3                     | ||P_null th_T - P_null th_0|| = 5.382e+00   (rel 5.851e-01)   | ||P_null th_T|| = 10.851935   (Adam leak + decay: (1-lr*wd)^T = 0.991040)

  reading:
    * plain GD: the drift is at float32 machine precision -- the identity is EXACT,
      and nothing in the code 'enforces' it; the gradient simply has no component
      along null(Phi).
    * minibatch SGD is also exact: a minibatch's gradient lies in row(Phi_B), and
      row(Phi_B) is a subset of row(Phi) -- the noise explores the row space but
      cannot reach the null space.
    * weight DECAY changes the story: the decay term -wd*theta HAS a component in
      the null space, so the projection shrinks by (1 - lr*wd) every step.  It is
      not preserved -- it is actively removed.
    * Adam leaks even without weight decay: the update is m_hat/(sqrt(v_hat)+eps),
      and v_hat = mean of g^2 is NOT in row(Phi) (the elementwise square of a
      row-space vector need not be row-space), so the step leaves the affine
      subspace theta_0 + row(Phi).

S2  the function-space consequence: training error is blind, test error is not
------------------------------------------------------------------------------

  same synthetic model, but now we ask what the null component COSTS.  Run GD to
  interpolation from theta_0 = 0 and from theta_0 = random, then compare.

  train MSE, theta_0 = 0       : 1.017e-12
  train MSE, theta_0 = random  : 1.616e-12   <- identical: Phi P_null theta_0 = 0
  test  MSE, theta_0 = 0       : 1.868e+02   <- the I-term only
  test  MSE, theta_0 = random  : 3.301e+02   <- I-term + the permanent init offset
  the two runs differ ONLY in the null space: max |(f_R - f_0) - Phi_te P_null theta_0| = 2.865e-05
  mean (Phi_te P_null theta_0)^2 = 8.397e+01   (the offset's own energy)
  test_R - test_0 = 1.433e+02  = offset^2 (8.397e+01) + 2<residual_0, offset> (5.935e+01)
    check: offset^2 + cross = 1.433e+02   (== test_R - test_0)

  reading: the two runs fit the training data EQUALLY well (to machine precision),
  and on the test set they differ by EXACTLY the init's null-space component.  The
  extra test error is the offset's own energy PLUS a correlation term -- the offset
  is not orthogonal to the row-space residual, because both come from the same
  random geometry.  The training curve shows NEITHER: starting from theta_0 = 0
  zeroes the tax entirely, which is why it is the best case.  A random init is a
  permanent, training-error-invisible test-error offset.

S3  the real v1.5 question q_11b94d (dev half): AdamW, minibatches, moving features
-----------------------------------------------------------------------------------

  torch 2.10.0+cu130 | numpy 1.26.4 | device cpu
  question q_11b94d | family univariate_regression | dataset sym_4d5411 | candidate set set_4096_var_fix_fix_cecee3
  dev half? True  (md5(qid) % 2 == 0 -> dev; the test half is never used here)

  official protocol (from the candidate's own train.py):
    torch.manual_seed(seed); Model(); AdamW(lr, betas, weight_decay);
    per step: idx = torch.randint(0, n, (bs,)) -> MSE on that minibatch -> opt.step()
  -> the init is RANDOM (theta_0 != 0), the optimizer is AdamW WITH weight decay,
     and the batches are minibatches.  None of these is plain full-batch GD.

  candidate A (c_6c95ed): d=4 w=256 p=264961 | AdamW lr=3e-05 wd=1e-05 | 256 steps x bs 16
    (a) AdamW weight-decay factor on the null component, (1 - lr*wd)^T = 0.9999999232
    (b) drift of the null component after the official 256 steps:
        ||P_null (theta_T - theta_0)|| = 7.460e-01   (||P_null theta_0|| = 3.207e+01, rel 2.326e-02)
        init NTK: rank 235/256 (float64), cond 1.287e+13
    (c) the lazy-limit (frozen-init-NTK) view of the SAME protocol:
        init output on test, MSE vs noisy y     : 5.055e-03
        init output on test, MSE vs target      : 5.055e-03
        lazy-limit of the official run (GD-from-init): test MSE 5.383e-04
        lazy-limit of 'train-from-zero' (kernel)   : test MSE 5.586e-04
        the PERMANENT init tax: mean (unreachable part of f_init)^2 = 1.458e-05
        official test MSE (10 seeds, notes/bank-audit.csv): 3.564e-03
        (the lazy-limit error vs the deterministic target: 5.383e-04)

  candidate C (c_d7ace4): d=5 w=192 p=187393 | AdamW lr=3e-05 wd=1e-05 | 256 steps x bs 16
    (a) AdamW weight-decay factor on the null component, (1 - lr*wd)^T = 0.9999999232
    (b) drift of the null component after the official 256 steps:
        ||P_null (theta_T - theta_0)|| = 3.317e-01   (||P_null theta_0|| = 3.490e+01, rel 9.507e-03)
        init NTK: rank 228/256 (float64), cond 1.424e+13
    (c) the lazy-limit (frozen-init-NTK) view of the SAME protocol:
        init output on test, MSE vs noisy y     : 1.762e-04
        init output on test, MSE vs target      : 1.762e-04
        lazy-limit of the official run (GD-from-init): test MSE 1.230e-03
        lazy-limit of 'train-from-zero' (kernel)   : test MSE 1.064e-03
        the PERMANENT init tax: mean (unreachable part of f_init)^2 = 4.690e-04
        official test MSE (10 seeds, notes/bank-audit.csv): 2.226e-04
        (the lazy-limit error vs the deterministic target: 1.230e-03)


S4  summary
-----------

  claim                                            |  status
  --------------------------------------------------------------------------------------------
  P_null(theta_t) = P_null(theta_0) under plain GD |  exact identity (S1, machine eps)
  same under minibatch SGD                         |  exact (row(Phi_B) subset row(Phi))
  same under weight decay                          |  FALSE: decays by (1 - lr*wd) each step
  same under Adam (no wd)                          |  FALSE: elementwise 2nd moment leaks
  init offset invisible in training error          |  exact (Phi P_null = 0) (S2)
  init offset = permanent test-error offset        |  exact in the lazy limit (S2, S3)
  v1.5 runs start from a random init               |  yes (torch.manual_seed(seed); Model())

  bottom line: the null-space projection is NOT protected by anything.  Under plain
  GD it is conserved because the gradient is blind, not because it is constrained.
  AdamW (the v1.5 optimizer) is free to change it: weight decay shrinks it and the
  adaptive step leaks out of the row space.  At the official budget the change is
  small, but the invariant is a property of the setup, not a law.

reproduce with: python verification/zero_space_init.py
