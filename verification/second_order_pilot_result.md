# second-order tangent-model pilot (M4)

torch 2.10.0+cu130 | device cpu | seed 0 | dev half only
pilot questions: ['q_ab8774', 'q_0b5a9f', 'q_0525ce', 'q_8f876c']


q_ab8774  [univariate_regression / architecture_only]  official answer = A
--------------------------------------------------------------------------
  A: real 0.00299 | order1 0.32594 | order2 0.14331   (p=1537, t real/o1/o2 = 6/0/4 s)
      order2 curve [frac, test_mse, ||d||]: 0.049:0.456(|d|=0.528)  0.1:0.379(|d|=0.69)  0.199:0.328(|d|=0.911)  0.398:0.229(|d|=1.289)  0.699:0.171(|d|=1.562)  1.0:0.143(|d|=1.728)
  B: real 0.07261 | order1 0.18689 | order2 3.64725   (p=2209, t real/o1/o2 = 1/0/2 s)
      order2 curve [frac, test_mse, ||d||]: 0.049:0.663(|d|=0.971)  0.1:0.704(|d|=1.34)  0.199:0.974(|d|=1.801)  0.398:1.4(|d|=2.497)  0.699:1.82(|d|=3.298)  1.0:3.65(|d|=3.923)
  C: real 0.26226 | order1 0.53946 | order2 0.64366   (p=1169, t real/o1/o2 = 1/0/3 s)
      order2 curve [frac, test_mse, ||d||]: 0.049:0.894(|d|=0.823)  0.1:0.772(|d|=1.256)  0.199:0.871(|d|=1.86)  0.398:0.691(|d|=2.424)  0.699:0.666(|d|=2.869)  1.0:0.644(|d|=3.198)
  winner: real A | order1 B [wrong] | order2 A [HIT ]  -> order2 flips B -> A (FIXED)

q_0b5a9f  [univariate_regression / architecture_only]  official answer = A
--------------------------------------------------------------------------
  A: real 0.12081 | order1 0.35014 | order2 0.32865   (p=3265, t real/o1/o2 = 2/0/5 s)
      order2 curve [frac, test_mse, ||d||]: 0.049:0.415(|d|=0.353)  0.1:0.391(|d|=0.422)  0.199:0.369(|d|=0.524)  0.398:0.361(|d|=0.659)  0.699:0.35(|d|=0.885)  1.0:0.329(|d|=1.15)
  B: real 0.29910 | order1 0.25665 | order2 3.08788   (p=2593, t real/o1/o2 = 1/0/3 s)
      order2 curve [frac, test_mse, ||d||]: 0.049:0.567(|d|=0.257)  0.1:0.758(|d|=0.353)  0.199:0.809(|d|=0.466)  0.398:0.854(|d|=0.615)  0.699:1.49(|d|=0.823)  1.0:3.09(|d|=1.059)
  C: real 0.33083 | order1 0.39674 | order2 0.81154   (p=3265, t real/o1/o2 = 1/0/3 s)
      order2 curve [frac, test_mse, ||d||]: 0.049:0.463(|d|=0.387)  0.1:0.458(|d|=0.582)  0.199:0.472(|d|=0.868)  0.398:0.58(|d|=1.325)  0.699:0.729(|d|=1.876)  1.0:0.812(|d|=2.358)
  winner: real A | order1 B [wrong] | order2 A [HIT ]  -> order2 flips B -> A (FIXED)

q_0525ce  [multivariate_regression / architecture_only]  official answer = A
----------------------------------------------------------------------------
  A: real 1.35999 | order1 4.07423 | order2 24.73058   (p=21633, t real/o1/o2 = 1/0/3 s)
      order2 curve [frac, test_mse, ||d||]: 0.047:24.5(|d|=1.423)  0.098:22.8(|d|=1.895)  0.199:21.1(|d|=2.514)  0.398:21.5(|d|=3.233)  0.699:23.9(|d|=3.805)  1.0:24.7(|d|=4.137)
  B: real 1.86285 | order1 2.47643 | order2 18.00281   (p=19297, t real/o1/o2 = 0/0/2 s)
      order2 curve [frac, test_mse, ||d||]: 0.047:15.8(|d|=1.762)  0.098:15.1(|d|=2.45)  0.199:17.7(|d|=3.105)  0.398:18.6(|d|=3.622)  0.699:17.9(|d|=4.065)  1.0:18(|d|=4.366)
  C: real 1.77870 | order1 1.77416 | order2 1.75386   (p=28801, t real/o1/o2 = 0/0/2 s)
      order2 curve [frac, test_mse, ||d||]: 0.047:2.5(|d|=1.196)  0.098:1.84(|d|=1.274)  0.199:1.77(|d|=1.349)  0.398:1.72(|d|=1.428)  0.699:1.73(|d|=1.474)  1.0:1.75(|d|=1.576)
  winner: real A | order1 C [wrong] | order2 C [wrong]

q_8f876c  [univariate_regression / architecture_only]  official answer = C
--------------------------------------------------------------------------
  A: real 0.12792 | order1 0.26342 | order2 0.15864   (p=83201, t real/o1/o2 = 1/3/12 s)
      order2 curve [frac, test_mse, ||d||]: 0.049:0.915(|d|=0.098)  0.1:0.57(|d|=0.192)  0.199:0.335(|d|=0.289)  0.398:0.248(|d|=0.324)  0.699:0.185(|d|=0.375)  1.0:0.159(|d|=0.415)
  B: real 0.11434 | order1 0.09170 | order2 nan   (p=49921, t real/o1/o2 = 1/2/7 s)
      order2 curve [frac, test_mse, ||d||]: 0.049:0.75(|d|=0.149)  0.1:0.865(|d|=0.179)  0.199:nan(|d|=nan)  0.398:nan(|d|=nan)  0.699:nan(|d|=nan)  1.0:nan(|d|=nan)
  C: real 0.03073 | order1 0.03021 | order2 nan   (p=50433, t real/o1/o2 = 1/2/8 s)
      order2 curve [frac, test_mse, ||d||]: 0.049:nan(|d|=nan)  0.1:nan(|d|=nan)  0.199:nan(|d|=nan)  0.398:nan(|d|=nan)  0.699:nan(|d|=nan)  1.0:nan(|d|=nan)
  winner: real C | order1 C [HIT ] | order2 A [wrong]  -> order2 flips C -> A (now wrong)

stabilization probe: order2 at a smaller learning rate
------------------------------------------------------
  the order2 surrogate is a non-convex quartic; at the official lr it can diverge.
  re-run it at lr x {0.3, 0.1} to see whether the blow-ups are a stability artifact.

  q_ab8774 (official answer A):
    A: order2@1x 0.1433  | order2@0.3x 0.307  | order2@0.1x 0.5006   (real 0.002995)
    B: order2@1x 3.647  | order2@0.3x 0.8773  | order2@0.1x 0.7168   (real 0.07261)
    C: order2@1x 0.6437  | order2@0.3x 0.7943  | order2@0.1x 0.9499   (real 0.2623)
  q_0b5a9f (official answer A):
    A: order2@1x 0.3287  | order2@0.3x 0.3731  | order2@0.1x 0.4347   (real 0.1208)
    B: order2@1x 3.088  | order2@0.3x 0.7257  | order2@0.1x 0.5334   (real 0.2991)
    C: order2@1x 0.8115  | order2@0.3x 0.4719  | order2@0.1x 0.4661   (real 0.3308)
  q_0525ce (official answer A):
    A: order2@1x 24.73  | order2@0.3x 21.57  | order2@0.1x 27.14   (real 1.36)
    B: order2@1x 18  | order2@0.3x 14.71  | order2@0.1x 20.52   (real 1.863)
    C: order2@1x 1.754  | order2@0.3x 1.796  | order2@0.1x 2.442   (real 1.779)
  q_8f876c (official answer C):
    A: order2@1x 0.1586  | order2@0.3x 0.281  | order2@0.1x 0.5787   (real 0.1279)
    B: order2@1x nan  | order2@0.3x 3.236  | order2@0.1x 1.08   (real 0.1143)
    C: order2@1x nan  | order2@0.3x nan  | order2@0.1x 0.1211   (real 0.03073)

summary
-------
  qid        | truth | real | order1 | order2 | order2 fixes?
  ----------------------------------------------------------------------
  q_ab8774 |  A    |  A   |   B    |   A    | YES (was wrong, now right)
  q_0b5a9f |  A    |  A   |   B    |   A    | YES (was wrong, now right)
  q_0525ce |  A    |  A   |   C    |   C    | no (still wrong)
  q_8f876c |  C    |  C   |   C    |   A    | NO -- it BROKE a right answer

  order2 fixed 2 wrong answer(s), broke 1 right answer(s), out of 4 dev-half pilot questions.

  honest caveats:
    * 1 seed only; the official answer is a 10-seed mean, so near-ties are noise here.
    * the quadratic surrogate is NON-convex (quartic loss): it is trained, not solved,
      and could land in a different basin than the real run for reasons unrelated to
      the second-order term.
    * q_11b94d (if included) is a NEAR-TIE failure (feature-learning gap ~0): the
      second-order term is not expected to fix it -- it is the negative control.
