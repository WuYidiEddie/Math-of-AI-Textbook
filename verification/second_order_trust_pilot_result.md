# second-order tangent model with a trust region (M4)

torch 2.10.0+cu130 | device cpu | seed 0 | dev half only
pilot questions: ['q_ab8774', 'q_0b5a9f', 'q_0525ce', 'q_8f876c']
trust radius r = c ||d_1||, c in (0.5, 1.0, 2.0); the same c is used for every candidate
projection after every optimizer step; official lr, batches and optimizer otherwise


q_ab8774  [univariate_regression / architecture_only]  official answer = A
--------------------------------------------------------------------------
  A: real 0.00299 | order1 0.32594 | order2 0.14331 | ||d_1|| 2.181
      c=0.5: test 0.23530 | final ||d|| 1.091 | r 1.091 | 5s
      c=1.0: test 0.14331 | final ||d|| 1.728 | r 2.181 | 5s
      c=2.0: test 0.14331 | final ||d|| 1.728 | r 4.362 | 5s
  B: real 0.07261 | order1 0.18689 | order2 3.64725 | ||d_1|| 4.132
      c=0.5: test 1.15398 | final ||d|| 2.066 | r 2.066 | 3s
      c=1.0: test 3.64725 | final ||d|| 3.923 | r 4.132 | 2s
      c=2.0: test 3.64725 | final ||d|| 3.923 | r 8.264 | 3s
  C: real 0.26226 | order1 0.53946 | order2 0.64366 | ||d_1|| 4.708
      c=0.5: test 0.68752 | final ||d|| 2.354 | r 2.354 | 2s
      c=1.0: test 0.64366 | final ||d|| 3.198 | r 4.708 | 2s
      c=2.0: test 0.64366 | final ||d|| 3.198 | r 9.416 | 3s
  winner: real A | order1 B | order2 A | tr0.5 A | tr1.0 A | tr2.0 A   (truth A)

q_0b5a9f  [univariate_regression / architecture_only]  official answer = A
--------------------------------------------------------------------------
  A: real 0.12081 | order1 0.35014 | order2 0.32865 | ||d_1|| 1.622
      c=0.5: test 0.34828 | final ||d|| 0.811 | r 0.811 | 5s
      c=1.0: test 0.32865 | final ||d|| 1.150 | r 1.622 | 5s
      c=2.0: test 0.32865 | final ||d|| 1.150 | r 3.244 | 5s
  B: real 0.29910 | order1 0.25665 | order2 3.08788 | ||d_1|| 1.308
      c=0.5: test 0.89429 | final ||d|| 0.654 | r 0.654 | 2s
      c=1.0: test 3.08788 | final ||d|| 1.059 | r 1.308 | 2s
      c=2.0: test 3.08788 | final ||d|| 1.059 | r 2.616 | 2s
  C: real 0.33083 | order1 0.39674 | order2 0.81154 | ||d_1|| 2.661
      c=0.5: test 0.58817 | final ||d|| 1.330 | r 1.331 | 3s
      c=1.0: test 0.81154 | final ||d|| 2.358 | r 2.661 | 3s
      c=2.0: test 0.81154 | final ||d|| 2.358 | r 5.322 | 3s
  winner: real A | order1 B | order2 A | tr0.5 A | tr1.0 A | tr2.0 A   (truth A)

q_0525ce  [multivariate_regression / architecture_only]  official answer = A
----------------------------------------------------------------------------
  A: real 1.35999 | order1 4.07423 | order2 24.73058 | ||d_1|| 7.150
      c=0.5: test 21.74210 | final ||d|| 3.575 | r 3.575 | 3s
      c=1.0: test 24.73058 | final ||d|| 4.137 | r 7.150 | 3s
      c=2.0: test 24.73058 | final ||d|| 4.137 | r 14.300 | 3s
  B: real 1.86285 | order1 2.47643 | order2 18.00281 | ||d_1|| 5.221
      c=0.5: test 12.36594 | final ||d|| 2.610 | r 2.611 | 2s
      c=1.0: test 18.00281 | final ||d|| 4.366 | r 5.221 | 2s
      c=2.0: test 18.00281 | final ||d|| 4.366 | r 10.442 | 2s
  C: real 1.77870 | order1 1.77416 | order2 1.75386 | ||d_1|| 2.597
      c=0.5: test 1.78048 | final ||d|| 1.299 | r 1.298 | 2s
      c=1.0: test 1.75386 | final ||d|| 1.576 | r 2.597 | 2s
      c=2.0: test 1.75386 | final ||d|| 1.576 | r 5.194 | 2s
  winner: real A | order1 C | order2 C | tr0.5 C | tr1.0 C | tr2.0 C   (truth A)

q_8f876c  [univariate_regression / architecture_only]  official answer = C
--------------------------------------------------------------------------
  A: real 0.12792 | order1 0.26342 | order2 0.15864 | ||d_1|| 0.561
      c=0.5: test 0.27674 | final ||d|| 0.281 | r 0.281 | 12s
      c=1.0: test 0.15864 | final ||d|| 0.415 | r 0.561 | 12s
      c=2.0: test 0.15864 | final ||d|| 0.415 | r 1.122 | 12s
  B: real 0.11434 | order1 0.09170 | order2 nan | ||d_1|| 0.403
      c=0.5: test 0.93870 | final ||d|| 0.165 | r 0.202 | 7s
      c=1.0: test 1064.76722 | final ||d|| 0.403 | r 0.403 | 7s
      c=2.0: test 9222.38802 | final ||d|| 0.806 | r 0.806 | 7s
  C: real 0.03073 | order1 0.03021 | order2 nan | ||d_1|| 0.265
      c=0.5: test 490.40994 | final ||d|| 0.132 | r 0.133 | 8s
      c=1.0: test 3361.04582 | final ||d|| 0.265 | r 0.265 | 9s
      c=2.0: test 22173.82326 | final ||d|| 0.530 | r 0.530 | 8s
  winner: real C | order1 C | order2 A | tr0.5 A | tr1.0 A | tr2.0 A   (truth C)

summary
-------
  qid        | truth | real | order1 | order2 | tr0.5 | tr1.0 | tr2.0
  --------------------------------------------------------------------------------------
  q_ab8774 |  A    |  A   | B  | A* | A* | A* | A*
  q_0b5a9f |  A    |  A   | B  | A* | A* | A* | A*
  q_0525ce |  A    |  A   | C  | C  | C  | C  | C 
  q_8f876c |  C    |  C   | C* | A  | A  | A  | A 

  (* = matches the official answer)
  order1: 1/4
  order2: 2/4
  tr0.5: 2/4
  tr1.0: 2/4
  tr2.0: 2/4

  interpretation limits:
    * dev-half exploratory pilot, one seed; the official answer uses 10 seeds.
    * the radius grid is reported in full, not selected per candidate by test MSE.
    * ||d_1|| comes from the order-1 training trajectory only, never from test labels.
    * projection prevents large displacements but cannot repair Taylor truncation
      error inside the ball, or a trajectory that needs feature re-linearization.
