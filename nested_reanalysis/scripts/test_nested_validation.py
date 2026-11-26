import biosystems_lib as B
import numpy as np
from sklearn.kernel_ridge import KernelRidge

rng = np.random.default_rng(90210)
X = rng.binomial(1, 0.4, (60, 40)).astype(np.float32)
R = rng.normal(3, 0.8, (60, 35)).astype(np.float32)
y = X[:, 0] + R[:, 1] + rng.normal(size=60)
cl = np.repeat(np.arange(3), 20)
train, test = np.arange(40), np.arange(40, 60)
KG, KE, _, _ = B.kernels(X, R, train)
yfit, mean, sd = B.response(y, train)
Ktr = KG[np.ix_(train, train)]
Kte = KG[np.ix_(test, train)]
column = Ktr.mean(0)
Ktc = Ktr - column[None, :] - Ktr.mean(1)[:, None] + column.mean()
Kvc = Kte - column[None, :] - Kte.mean(1)[:, None] + column.mean()
reference = mean + sd * (KernelRidge(alpha=0.3, kernel="precomputed").fit(Ktc, yfit-yfit.mean()).predict(Kvc) + yfit.mean())
actual = B.ridge_grid(KG, y, train, test, np.array([0.3]))[:, 0]
np.testing.assert_allclose(actual, reference, atol=1e-9)
first, params1, details = B.nested_predict(X, R, y, cl, train, test, "lopo", 100)
changed = y.copy()
changed[test] = rng.normal(1000, 100, len(test))
second, params2, _ = B.nested_predict(X, R, changed, cl, train, test, "lopo", 100)
for feature in first:
    np.testing.assert_array_equal(first[feature], second[feature])
assert params1 == params2
assert details["inner_type"] == "grouped"
X2, R2 = X.copy(), R.copy()
X2[test] = 1-X2[test]
R2[test] += 100
KG2, KE2, _, _ = B.kernels(X2, R2, train)
np.testing.assert_array_equal(KG[np.ix_(train, train)], KG2[np.ix_(train, train)])
np.testing.assert_array_equal(KE[np.ix_(train, train)], KE2[np.ix_(train, train)])
assert np.array_equal(B.splits(cl, "lopo", 1)[0][0][1], np.arange(40, 60))
inner, _ = B.splits(cl[train], "lopo", 100, inner=True)
ti, vi = train[inner[0][0]], train[inner[0][1]]
KGi, KEi, markers, genes = B.kernels(X, R, ti)
Xi, Ri, yi = X.copy(), R.copy(), y.copy()
Xi[vi] = 1 - Xi[vi]
Ri[vi] += 100
Ri[vi, 0] = -10
yi[vi] = rng.normal(1000, 100, len(vi))
KGj, KEj, markers2, genes2 = B.kernels(Xi, Ri, ti)
np.testing.assert_array_equal(KGi[np.ix_(ti, ti)], KGj[np.ix_(ti, ti)])
np.testing.assert_array_equal(KEi[np.ix_(ti, ti)], KEj[np.ix_(ti, ti)])
assert (markers, genes) == (markers2, genes2)
for original, perturbed in zip(B.response(y, ti), B.response(yi, ti)):
    np.testing.assert_array_equal(original, perturbed)
print("PASS: numerical KRR agreement, outer-test-label invariance, inner-validation preprocessing isolation, training-kernel invariance and grouped partitions.")
