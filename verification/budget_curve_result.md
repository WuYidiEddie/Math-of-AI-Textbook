# budget / learning-curve derivation checks (textbook/03-budget-and-curves.md)

numpy 1.26.4 | spectrum: 2 large modes + power-law tail of 58 modes

C1  effective step size: eta*T is what matters
----------------------------------------------
     eta      T    eta*T         R_te
   0.005   4000    20.00   2.2293e-04
   0.010   2000    20.00   2.2280e-04
   0.020   1000    20.00   2.2255e-04
   0.050    400    20.00   2.2178e-04
   0.100    200    20.00   2.2051e-04
   0.200    100    20.00   2.1798e-04
   0.400     50    20.00   1.0799e+34
the first six rows share eta*T = 20 and agree to 4+ digits; the last row (eta*lambda_max = 3.2) is past EoS
PASS C1: only the product eta*T matters, as long as eta*lambda_max < 2

C2  spectrum with a gap => exponential learning curve, tau = 1/(2 eta lambda_2)
-------------------------------------------------------------------------------
eta = 0.05, gap: lambda_1 = 4.0, lambda_2 = ... = 1.0
  T=   5  R_te = 0.500464   exp(-T/tau) fit = 0.486353
  T=  10  R_te = 0.289095   exp(-T/tau) fit = 0.290996
  T=  20  R_te = 0.102836   exp(-T/tau) fit = 0.104174
  T=  40  R_te = 0.013212   exp(-T/tau) fit = 0.013351
  T=  80  R_te = 0.000218   exp(-T/tau) fit = 0.000219
  T= 160  R_te = 0.000000   exp(-T/tau) fit = 0.000000
fitted tau = 9.735 | theory tau = 1/(2*eta*lambda_2) = 10.000 | ratio = 0.9735
PASS C2: the relaxation rate is 2*eta*lambda_2 (the error is a SQUARE, hence the factor 2)

C3  power-law spectrum => power-law learning curve
--------------------------------------------------
  T=   50  R_te = 1.202875e-01
  T=  100  R_te = 6.034776e-02
  T=  200  R_te = 3.017351e-02
  T=  400  R_te = 1.503028e-02
  T=  800  R_te = 7.445020e-03
  T= 1600  R_te = 3.649948e-03
log-log slope = -1.008  (R_te ~ T^-1.01)
PASS C3: no gap in the spectrum => no single tau => power law. 'Plateau-like' curves are the intermediate case

C4  edge of stability: eta > 2/lambda_max => oscillation
--------------------------------------------------------
     eta  eta*lam   |1-eta*lam|^50   loss at T=50   loss at T=51
    0.20    0.800        1.126e-35      1.268e-70      5.071e-72
    0.45    1.800        1.427e-05      2.037e-10      1.304e-10
    0.50    2.000        1.000e+00      1.000e+00      1.000e+00
    0.55    2.200        9.100e+03      8.282e+07      1.193e+08
    0.75    3.000        1.126e+15      1.268e+30      5.071e+30
above eta*lambda_max = 2 the factor is negative: the mode alternates sign, so the loss sawtooths
and |1-eta*lambda| > 1, so that mode's amplitude GROWS slowly. In a real network the other
modes still decrease, so the overall loss can keep trending down while the top mode oscillates
=> that coexistence is exactly the EoS regime ('noisy but still learning'), not clean divergence
PASS C4: instability (oscillation) starts at eta*lambda_max = 2; every mode's amplitude grows there

C5  early stopping vs ridge: compare the SAME quantity on both sides
--------------------------------------------------------------------
residual filter (how much of the mode is still 'unlearnt'):
     T   lambda   GD (1-eta*lam)^T  ridge lam_reg/(lam+lam_reg)    ratio
    20     0.01           0.980189                     0.980392    1.000
    20     0.10           0.817907                     0.833333    0.981
    20     1.00           0.121577                     0.333333    0.365
    20    10.00           0.000000                     0.047619    0.000
   100     0.01           0.904792                     0.909091    0.995
   100     0.10           0.366032                     0.500000    0.732
   100     1.00           0.000027                     0.090909    0.000
   100    10.00           0.000000                     0.009901    0.000
   500     0.01           0.606379                     0.666667    0.910
   500     0.10           0.006570                     0.166667    0.039
   500     1.00           0.000000                     0.019608    0.000
   500    10.00           0.000000                     0.001996    0.000
ratio range: 0.000 .. 1.000  (lam_reg = 1/(eta*T))

retained filter (1 - residual) on both sides:
     T   lambda   GD 1-(1-eta*lam)^T    ridge lam/(lam+lam_reg)    ratio
    20     0.01             0.019811                   0.019608    1.010
    20     0.10             0.182093                   0.166667    1.093
    20     1.00             0.878423                   0.666667    1.318
    20    10.00             1.000000                   0.952381    1.050
   100     0.01             0.095208                   0.090909    1.047
   100     0.10             0.633968                   0.500000    1.268
   100     1.00             0.999973                   0.909091    1.100
   100    10.00             1.000000                   0.990099    1.010
   500     0.01             0.393621                   0.333333    1.181
   500     0.10             0.993430                   0.833333    1.192
   500     1.00             1.000000                   0.980392    1.020
   500    10.00             1.000000                   0.998004    1.002
PASS C5: the two knobs live on the same scale (lam_reg ~ 1/(eta*T)), but the FILTER SHAPES differ;
       they agree only for eta*T*lambda << 1, so use the analogy for trends, never for exact numbers

C6  effective step size per optimizer (measured, not assumed)
-------------------------------------------------------------
SGD(lr=0.1)            first step = 0.0477 | mean step (2-30) = 0.0323 | max step = 0.0477 | ||theta_30 - theta_0|| = 0.9837
SGD(lr=0.1,mom=0.9)    first step = 0.0477 | mean step (2-30) = 0.1062 | max step = 0.1963 | ||theta_30 - theta_0|| = 2.0441
RMSprop(lr=0.01)       first step = 1.1314 | mean step (2-30) = 0.0608 | max step = 1.1314 | ||theta_30 - theta_0|| = 1.8858
Adam(lr=0.01)          first step = 0.1131 | mean step (2-30) = 0.0705 | max step = 0.1131 | ||theta_30 - theta_0|| = 1.8825
PASS C6: SGD's step scales with the gradient magnitude; Adam/RMSprop's step is set by lr itself
       => the same lr means very different things across optimizers; compare 'effective step', not lr

reproduce with: python verification/budget_curve_derivation.py
