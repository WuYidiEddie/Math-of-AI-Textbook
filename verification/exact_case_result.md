# exact-case derivation checks (textbook/02-exact-case.md)

numpy 1.26.4 | p=100 n=20 m=2000 eta=0.2 T=200 | seed 0

C1  grad L(theta) = A (theta - theta*),   L(theta) = ||Phi theta - y||^2 / (2n)
-------------------------------------------------------------------------------
max |grad_analytic - grad_finite_difference|      = 1.797e-09
max |grad_analytic - A (theta - theta*)|           = 6.661e-16
A (theta* - theta*)?  A theta* - (1/n) Phi^T y      = 3.331e-16  (0 => theta* is a fixed point)
PASS C1: True

C2  theta_T - theta* = (I - eta A)^T (theta_0 - theta*)   (T = step count)
--------------------------------------------------------------------------
max |GD run - closed form| (theta_0 = 0, T=200)   = 1.162e-14
max |GD run - closed form| (theta_0 != 0)           = 1.671e-14
PASS C2: True

C3  per-direction filter (spectral decomposition of A)
------------------------------------------------------
lambda range: [-0.0000, 9.6179] | #lambda>1e-10: 20
max |U^T (theta_T - theta*) - diag((1-eta*lambda)^T) U^T (theta_0 - theta*)| = 1.937e-14
lambda_min (should be ~0): -4.032e-15 | (1-eta*lambda_min)^T = 1.000000  <- null directions do NOT decay
PASS C3: True

C4  null space
--------------
rank(Phi) = 20 | dim null(Phi) = 80 | p - n = 80
max |A Pnull - 0|                                  = 6.566e-16
max |Pnull (theta_T) - Pnull (theta_0)|  (frozen)   = 8.132e-15
max |Pnull (theta_T) - Pnull (theta*)|  (never learnt) = 2.667e-01
||P_null theta*||^2 = 1.000000   ||P_row theta*||^2 = 1.000000
||P_null (theta_T - theta*)|| = 1.000000   <- the part of the target that was never learnt
||P_row (theta_T - theta*)|| = 1.237859e-08   <- the part that was learnt
smallest nonzero lambda = 1.748092 -> slowest filter factor (1-eta*lambda)^T at T=200 = 0.000000
max |(I - eta*Sigma)^20000 - P_null| (limit = NULL-space projector, since (1-eta*0)^T = 1) = 6.245e-12   (T_BIG=20000)
PASS C4: True

C5  the two floors
------------------
train error (1/n)||Phi theta_T - y||^2              = 1.473e-15   (target in class => 0)
test  error (1/m)||Phi_te theta_T - y_te||^2        = 0.918674
predicted floor (1/m)||Phi_te P_null theta*||^2      = 0.918674
||P_null theta*||^2 (isotropic proxy)               = 1.000000
test error when test points ARE the train points    = 0.000000   <- null direction invisible => floor 0
floor with theta_0 != 0: (1/m)||Phi_te P_null (theta_0 - theta*)||^2 = 1.737784   (vs 1.844724 in parameter space)
floor when the test design only spans the row space = 5.372e-32  (target invisible in the null direction)
PASS C5: floors depend on the (train, test) geometry, not only on p and n

C6  stability: |1 - eta lambda_max| < 1
---------------------------------------
eta =   0.1000 | eta*lambda_max =  0.962 | |1-eta*lambda_max| = 0.038 | ||theta_500|| = 2.583e-01
eta =   0.5000 | eta*lambda_max =  4.809 | |1-eta*lambda_max| = 3.809 | ||theta_500|| = 2.670e+12   (diverges)
eta =   0.2059 | eta*lambda_max =  1.980 | |1-eta*lambda_max| = 0.980 | ||theta_500|| = 2.583e-01
eta =   0.2100 | eta*lambda_max =  2.020 | |1-eta*lambda_max| = 1.020 | ||theta_500|| = 5.248e+02
eta =   0.4159 | eta*lambda_max =  4.000 | |1-eta*lambda_max| = 3.000 | ||theta_500|| = 2.100e+12   (diverges)
PASS C6: stable iff 0 < eta < 2/lambda_max = 0.2079

C7  train-before-you-train: closed-form R_te(T) vs the real GD run
------------------------------------------------------------------
T=   1  GD R_te = 1.079150   closed form = 1.079150   row-space part of ||theta_T-theta*||^2 = 0.141695
T=   5  GD R_te = 0.932007   closed form = 0.932007   row-space part of ||theta_T-theta*||^2 = 0.010026
T=  20  GD R_te = 0.918607   closed form = 0.918607   row-space part of ||theta_T-theta*||^2 = 0.000778
T=  50  GD R_te = 0.918611   closed form = 0.918611   row-space part of ||theta_T-theta*||^2 = 0.000006
T= 200  GD R_te = 0.918674   closed form = 0.918674   row-space part of ||theta_T-theta*||^2 = 0.000000
PASS C7: the closed form predicts the run without training

C8  isotropic random design: E R_te vs (1 - 1/gamma)||theta*||^2 + sigma^2/(gamma-1)
------------------------------------------------------------------------------------
test error averaged over the test distribution analytically (one test draw per trial would
add a chi-square fluctuation of order 1 and swamp the comparison)
p=60 n=15 gamma=4.0 sigma=0.1: E R_te = 0.7622 +/- 0.0038   formula = 0.7634   z = -0.32
p=60 n=30 gamma=2.0 sigma=0.1: E R_te = 0.5136 +/- 0.0045   formula = 0.5203   z = -1.50
p=60 n=15 gamma=4.0 sigma=0.0: E R_te = 0.7488 +/- 0.0039   formula = 0.7500   z = -0.31
PASS C8: excess risk = (1 - 1/gamma)||theta*||^2 + sigma^2 n/(p-n-1)  (asymptotic: sigma^2/(gamma-1))

C8b  why squared loss: it targets the conditional MEAN (L1 would target the median)
-----------------------------------------------------------------------------------
true slope = 3.0 | noise = Exponential(1)-1, mean = -0.0009, median = -0.3060
squared-loss fit : slope = 2.9798, intercept = -0.0007   <- targets E[y|x] = 3x + 0
L1 fit (IRLS)    : slope = 2.9714, intercept = -0.3063   <- targets median[y|x] = 3x + (-0.3060)
PASS C8b: squared loss = conditional mean; L1 = conditional median. The closed form exists only for the former

C9  kernel spectral filter: three-mode example
----------------------------------------------
mode 1: lambda= 10.00  a^2=1.00  eta*lambda*T= 10.000  (1-eta*lambda)^(2T)=0.000000  contribution=0.000000
mode 2: lambda=  1.00  a^2=0.25  eta*lambda*T=  1.000  (1-eta*lambda)^(2T)=0.133980  contribution=0.033495
mode 3: lambda=  0.01  a^2=0.04  eta*lambda*T=  0.010  (1-eta*lambda)^(2T)=0.980198  contribution=0.039208
total R_te(T=100) = 0.072703   (share of the slowest mode = 53.9%)

C10  early stopping vs ridge: rough correspondence, not equality
----------------------------------------------------------------
T=50  lambda=0.05: GD=0.6050 ridge(1/(eta T))=0.3333 ratio=1.82 | lambda=0.5: GD=0.0052 ridge(1/(eta T))=0.8333 ratio=0.01 | lambda=5.0: GD=0.0000 ridge(1/(eta T))=0.9804 ratio=0.00
T=200  lambda=0.05: GD=0.1340 ridge(1/(eta T))=0.6667 ratio=0.20 | lambda=0.5: GD=0.0000 ridge(1/(eta T))=0.9524 ratio=0.00 | lambda=5.0: GD=0.0000 ridge(1/(eta T))=0.9950 ratio=0.00
PASS C10: the two filters agree at the ends only; never quote them as equal

all checks printed above are reproduced by: python verification/exact_case_derivation.py
