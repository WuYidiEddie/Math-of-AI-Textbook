# moving-linearization pilot (M4 route a): the feature-learning dial S

torch 2.10.0+cu130 | device cpu | seed 0 | dev half only
pilot questions: ['q_ab8774', 'q_0b5a9f', 'q_0525ce', 'q_8f876c']   S in [1, 4, 16, 64]

S = 1 is the lazy surrogate (kernel frozen at init); S = steps would be the real
training.  Prediction = the REAL net evaluated at the surrogate-reached parameters.


q_ab8774  [univariate_regression / architecture_only]  official answer = A
--------------------------------------------------------------------------
  A: real 0.00299 | S1 1.32670(0s) | S4 0.37186(0s) | S16 0.02333(1s) | S64 0.00738(2s)
  B: real 0.07261 | S1 2.29125(0s) | S4 0.17400(0s) | S16 0.12162(0s) | S64 0.09017(1s)
  C: real 0.26226 | S1 0.60936(0s) | S4 0.45887(0s) | S16 0.33784(0s) | S64 0.27735(1s)
  winner: real A | S1 C | S4 B | S16 A(=truth) | S64 A(=truth)   (truth A)

q_0b5a9f  [univariate_regression / architecture_only]  official answer = A
--------------------------------------------------------------------------
  A: real 0.12081 | S1 0.45848(0s) | S4 0.38275(0s) | S16 0.33601(1s) | S64 0.13571(2s)
  B: real 0.29910 | S1 0.37543(0s) | S4 0.31033(0s) | S16 0.30254(1s) | S64 0.29921(1s)
  C: real 0.33083 | S1 0.45607(0s) | S4 0.38815(0s) | S16 0.34541(1s) | S64 0.33534(1s)
  winner: real A | S1 B | S4 B | S16 B | S64 A(=truth)   (truth A)

q_0525ce  [multivariate_regression / architecture_only]  official answer = A
----------------------------------------------------------------------------
  A: real 1.35999 | S1 68.72273(0s) | S4 1.93651(1s) | S16 1.81813(2s) | S64 1.66237(7s)
  B: real 1.86285 | S1 8.27391(0s) | S4 1.93391(0s) | S16 1.88996(1s) | S64 1.86801(3s)
  C: real 1.77870 | S1 14.42766(0s) | S4 1.81242(1s) | S16 1.78954(2s) | S64 1.78565(6s)
  winner: real A | S1 B | S4 C | S16 C | S64 A(=truth)   (truth A)

q_8f876c  [univariate_regression / architecture_only]  official answer = C
--------------------------------------------------------------------------
  A: real 0.12792 | S1 1.39434(3s) | S4 0.15444(3s) | S16 0.13496(7s) | S64 0.12960(20s)
  B: real 0.11434 | S1 0.16227(2s) | S4 0.12488(2s) | S16 0.11589(4s) | S64 0.11472(9s)
  C: real 0.03073 | S1 0.12847(2s) | S4 0.04771(2s) | S16 0.03672(4s) | S64 0.03127(12s)
  winner: real C | S1 C(=truth) | S4 C(=truth) | S16 C(=truth) | S64 C(=truth)   (truth C)

summary: does the winner move toward the truth as S grows?
----------------------------------------------------------
  qid        | truth | real | S1 | S4 | S16 | S64
  ------------------------------------------------------------------------
  q_ab8774 |  A    |  A   | C  | B  | A* | A*
  q_0b5a9f |  A    |  A   | B  | B  | B  | A*
  q_0525ce |  A    |  A   | B  | C  | C  | A*
  q_8f876c |  C    |  C   | C* | C* | C* | C*

  (* = matches the official answer)

  reading:
    * if S=1 already wins, the question is near-lazy (feature learning irrelevant);
      if the winner only appears at S=4/16, the kernel movement is what decides it.
    * unlike the quadratic surrogate, every stage here is a CONVEX linear model,
      so there is no divergence -- the cost is S Jacobians per candidate.
    * 1 seed; the official answer is a 10-seed mean.
