# moving second-order tangent models (M4)

torch 2.10.0+cu130 | device cpu | seed 0 | dev half only
pilot questions: ['q_ab8774', 'q_0b5a9f', 'q_0525ce', 'q_8f876c'] | S in (4, 16, 64)
prediction = REAL network at the surrogate-reached parameters
first-order comparison = saved moving-linearization run, same seed and protocol


q_ab8774  [univariate_regression / architecture_only]  official answer = A
--------------------------------------------------------------------------
  A: real 0.00299 | S4 first 0.3719 / second 0.2851 (|d|=1.37, 6s) | S16 first 0.02333 / second 0.003803 (|d|=1.35, 4s) | S64 first 0.007383 / second 0.002597 (|d|=1.33, 5s)
  B: real 0.07261 | S4 first 0.174 / second 0.1569 (|d|=3.65, 1s) | S16 first 0.1216 / second 0.1183 (|d|=3.65, 2s) | S64 first 0.09017 / second 0.1103 (|d|=3.69, 2s)
  C: real 0.26226 | S4 first 0.4589 / second 0.4214 (|d|=2.88, 2s) | S16 first 0.3378 / second 0.3831 (|d|=2.74, 2s) | S64 first 0.2773 / second 0.2895 (|d|=2.74, 3s)
  winner: real A | first S4 B | second S4 B | hybrid S4 B | first S16 A | second S16 A | hybrid S16 A | first S64 A | second S64 A | hybrid S64 A   (truth A)

q_0b5a9f  [univariate_regression / architecture_only]  official answer = A
--------------------------------------------------------------------------
  A: real 0.12081 | S4 first 0.3828 / second 0.2794 (|d|=1.18, 5s) | S16 first 0.336 / second 0.1275 (|d|=1.34, 6s) | S64 first 0.1357 / second 0.1262 (|d|=1.36, 8s)
  B: real 0.29910 | S4 first 0.3103 / second 0.3115 (|d|=0.95, 3s) | S16 first 0.3025 / second 0.3038 (|d|=0.94, 3s) | S64 first 0.2992 / second 0.3042 (|d|=0.96, 3s)
  C: real 0.33083 | S4 first 0.3881 / second 0.3729 (|d|=2.27, 3s) | S16 first 0.3454 / second 0.3581 (|d|=2.28, 3s) | S64 first 0.3353 / second 0.3496 (|d|=2.32, 4s)
  winner: real A | first S4 B | second S4 A | hybrid S4 A | first S16 B | second S16 A | hybrid S16 A | first S64 A | second S64 A | hybrid S64 A   (truth A)

q_0525ce  [multivariate_regression / architecture_only]  official answer = A
----------------------------------------------------------------------------
  A: real 1.35999 | S4 first 1.937 / second 1.809 (|d|=3.63, 3s) | S16 first 1.818 / second 1.77 (|d|=3.29, 6s) | S64 first 1.662 / second 1.641 (|d|=3.24, 13s)
  B: real 1.86285 | S4 first 1.934 / second 2.17 (|d|=4.24, 2s) | S16 first 1.89 / second 1.994 (|d|=4.06, 4s) | S64 first 1.868 / second 1.933 (|d|=3.99, 8s)
  C: real 1.77870 | S4 first 1.812 / second 1.801 (|d|=1.71, 3s) | S16 first 1.79 / second 1.791 (|d|=1.70, 5s) | S64 first 1.786 / second 1.783 (|d|=1.70, 13s)
  winner: real A | first S4 C | second S4 C | hybrid S4 C | first S16 C | second S16 A | hybrid S16 A | first S64 A | second S64 A | hybrid S64 A   (truth A)

q_8f876c  [univariate_regression / architecture_only]  official answer = C
--------------------------------------------------------------------------
  A: real 0.12792 | S4 first 0.1544 / second 0.1305 (|d|=0.39, 13s) | S16 first 0.135 / second 0.1283 (|d|=0.38, 20s) | S64 first 0.1296 / second 0.128 (|d|=0.38, 46s)
  B: real 0.11434 | S4 first 0.1249 / second nan (|d|=nan, 8s) | S16 first 0.1159 / second nan (|d|=nan, 11s) | S64 first 0.1147 / second 0.1152 (|d|=0.38, 22s)
  C: real 0.03073 | S4 first 0.04771 / second nan (|d|=nan, 9s) | S16 first 0.03672 / second nan (|d|=nan, 13s) | S64 first 0.03127 / second nan (|d|=nan, 29s)
  winner: real C | first S4 C | second S4 A | hybrid S4 C | first S16 C | second S16 A | hybrid S16 C | first S64 C | second S64 B | hybrid S64 C   (truth C)

summary: first / second / nonfinite-fallback hybrid at equal S
--------------------------------------------------------------
  qid        | truth | S4 f/s/h | S16 f/s/h | S64 f/s/h
  ------------------------------------------------------------------------------
  q_ab8774 |  A    | B /B /B  | A*/A*/A* | A*/A*/A*
  q_0b5a9f |  A    | B /A*/A* | B /A*/A* | A*/A*/A*
  q_0525ce |  A    | C /C /C  | C /A*/A* | A*/A*/A*
  q_8f876c |  C    | C*/A /C* | C*/A /C* | C*/B /C*

  (* = matches the official answer; h = first-order fallback iff second is nonfinite)
  first, S=4: 1/4
  first, S=16: 2/4
  first, S=64: 4/4
  second, S=4: 1/4
  second, S=16: 3/4
  second, S=64: 3/4
  hybrid, S=4: 2/4
  hybrid, S=16: 4/4
  hybrid, S=64: 4/4

  interpretation limits:
    * dev-half exploratory pilot, one seed; the official answer uses 10 seeds.
    * second order adds curvature within each stage, but its stage loss is still
      non-convex; re-linearization limits how far that approximation is used.
    * matching real training at large S is a consistency check, not evidence that
      the second-order term helps; the useful comparison is equal small S.
    * the nonfinite-fallback hybrid uses no labels, but was proposed AFTER seeing
      these dev results; it needs fresh questions before counting as an improvement.
