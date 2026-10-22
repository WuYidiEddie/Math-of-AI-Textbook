# generalization / validation-statistics checks (textbook/04-generalization-and-validation.md)

numpy 1.26.4 | Gaussian design, isotropic test points, min-norm (ridgeless) interpolator

C1  double-descent expectation: E R_te = sigma^2 (max(p,n)-1) / (|p-n|-1)
-------------------------------------------------------------------------
p=  30 n= 100 (p/n= 0.30, |p-n|-1= 69): E R_te = 0.1298 +/- 0.0006   formula = 0.1291   ratio = 1.005   z = +1.14
p=  60 n= 100 (p/n= 0.60, |p-n|-1= 39): E R_te = 0.2305 +/- 0.0023   formula = 0.2285   ratio = 1.009   z = +0.89
p= 100 n=  30 (p/n= 3.33, |p-n|-1= 69): E R_te = 0.1293 +/- 0.0006   formula = 0.1291   ratio = 1.001   z = +0.31
p= 200 n=  30 (p/n= 6.67, |p-n|-1=169): E R_te = 0.1061 +/- 0.0002   formula = 0.1060   ratio = 1.001   z = +0.44
PASS C1: the SAME estimator (min-norm interpolator) gives E R_te = sigma^2 (max(p,n)-1)/(|p-n|-1)
       on both sides of the threshold; the formula is symmetric in (p, n) and peaks at p = n
       (asymptotic result: the empirical mean sits a few percent above it at finite n)

C1b  the shape of the curve: it RISES toward the threshold, then FALLS after it
-------------------------------------------------------------------------------
  formula, n=100, sigma=0.3:  p=1:0.0909  p=10:0.1001  p=50:0.1818  p=90:0.9900  p=98:8.9100  p=99:inf  p=101:inf  p=110:1.0900  p=200:0.1809  p=1000:0.1000
  -> from the left the curve ASCENDS toward p = n and from the right it DESCENDS:
     the right arm is the 'second descent' and comes from this (variance) formula itself;
     the descending left arm of the classic figure does NOT come from here, it comes from
     the approximation error A(capacity) shrinking as the model class grows (see chapter 04 S2.6)
  simulated p=   1, n=100: E R_te = 0.0908 +/- 0.0001   formula = 0.0909   ratio = 0.999
  simulated p=  50, n=100: E R_te = 0.1808 +/- 0.0013   formula = 0.1818   ratio = 0.994
  simulated p= 110, n=100: E R_te = 1.1048 +/- 0.0290   formula = 1.0900   ratio = 1.014
  simulated p= 200, n=100: E R_te = 0.1808 +/- 0.0010   formula = 0.1809   ratio = 1.000
PASS C1b: the curve is single-peaked at |p-n| = 1; on the p>n side it decreases (second descent),
        on the p<n side it increases (that arm is dominated by approximation, not by this formula)

C2  signal term: E ||theta_hat - theta*||^2 = (1 - 1/gamma)||theta*||^2 + sigma^2 n/(p-n-1)
-------------------------------------------------------------------------------------------
p=60 n=15 gamma=4.0 sigma=0.1: E||theta_hat - theta*||^2 = 0.7522 +/- 0.0038   formula = 0.7534   z = -0.32
p=60 n=30 gamma=2.0 sigma=0.1: E||theta_hat - theta*||^2 = 0.5112 +/- 0.0043   formula = 0.5103   z = +0.20
p=60 n=15 gamma=4.0 sigma=0.0: E||theta_hat - theta*||^2 = 0.7531 +/- 0.0038   formula = 0.7500   z = +0.81
PASS C2: the zero-space bias (1 - 1/gamma)||theta*||^2 grows with p/n; the noise term shrinks

C3  the threshold: the expectation DIVERGES while single draws stay finite
--------------------------------------------------------------------------
p=41 n=40 (|p-n|-1=0): median = 2.6807 | p90 = 17.0457 | p99 = 162.6947 | p99.9 = 783.0306 | mean = 12.5034   formula = diverges
p=50 n=40 (|p-n|-1=9): median = 0.4292 | p90 = 0.7391 | p99 = 1.4228 | p99.9 = 2.1237 | mean = 0.4848   formula = 0.4900
the median barely moves from p=41 to p=50 while the mean explodes: a heavy right tail
PASS C3: never quote the peak height of double descent from a single seed; the expectation diverges,
       single-sample values stay finite and are much smaller than the mean

C4  paired vs independent comparison, candidates scored on a SHARED test set
----------------------------------------------------------------------------
  candidate A (8 features, 200 steps): 10-seed mean test error = 0.01114 (sd 0.00048) | population risk = 0.01107
  candidate B (8 features, 210 steps): 10-seed mean test error = 0.01139 (sd 0.00046) | population risk = 0.01129
  candidate C (4 features, 200 steps): 10-seed mean test error = 0.27647 (sd 0.00969) | population risk = 0.28305
  A - B: diff = -0.000253 (population truth -0.000223) | SE independent = 0.000210 | SE paired = 0.000177 (ratio 1.19x) | |t| = 1.43 vs 2.262 -> knife-edge (not significant)
  A - C: diff = -0.265328 (population truth -0.271980) | SE independent = 0.003067 | SE paired = 0.003050 (ratio 1.01x) | |t| = 87.00 vs 2.262 -> SIGNIFICANT
PASS C4: the paired SE removes the seed noise shared by the two candidates and the shared test set
       removes the test-sampling noise, so the paired test is more sensitive than the
       independent formula; a real difference (C) is detected, a negligible one (B) is a knife-edge

C4b  how much test-set noise is cancelled by pairing (fixed models, many test sets)
  A vs B: sd(test error of A) = 0.00068 | of B = 0.00070 | sd of the paired difference on the SAME test set = 0.00036 | independent formula would give 0.00097 -> 2.72x
  A vs C: sd(test error of A) = 0.00068 | of C = 0.01774 | sd of the paired difference on the SAME test set = 0.01774 | independent formula would give 0.01776 -> 1.00x
PASS C4b: the two candidates are scored on the SAME test points, so the test-sampling error is
        common and cancels in the difference; the more similar the candidates, the more it
        cancels (the A-vs-C pair barely benefits because C's own test-set error dominates)

C5  validation-set reuse bias: E[max of K] - max ~ sigma sqrt(2 log K)
----------------------------------------------------------------------
K=   3: E[max] = 0.8449 | sqrt(2 log K) = 1.4823 (upper bound, gap -0.6374) | refined asymptotic = 0.5968
K=  10: E[max] = 1.5350 | sqrt(2 log K) = 2.1460 (upper bound, gap -0.6109) | refined asymptotic = 1.3619
K=  50: E[max] = 2.2488 | sqrt(2 log K) = 2.7971 (upper bound, gap -0.5484) | refined asymptotic = 2.1009
K= 200: E[max] = 2.7459 | sqrt(2 log K) = 3.2552 (upper bound, gap -0.5093) | refined asymptotic = 2.6104
PASS C5: sqrt(2 log K) is an UPPER bound on E[max of K iid N(0,1)]; the exact asymptotic is
       sqrt(2 log K) - (log log K + log 4 pi)/(2 sqrt(2 log K)), so the bound is conservative by ~0.5-0.6 for K <= 200

reproduce with: python verification/generalization_stats.py
