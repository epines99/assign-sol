def generate_cov(dim, corr):
    acc = []
    for i in range(dim):
        row = np.ones((1, dim)) * corr
        row[0][i] = 1
        acc.append(row)
    return np.concatenate(acc, axis=0)


def generate_multnorm(nobs, corr, nvar):
    mu = np.zeros(nvar)
    std = (np.abs(np.random.normal(loc=1, scale=.5, size=(nvar, 1)))) ** (1 / 2)
    # generate random normal distribution
    acc = []
    for i in range(nvar):
        acc.append(np.reshape(np.random.normal(mu[i], std[i], nobs), (nobs, -1)))

    normvars = np.concatenate(acc, axis=1)

    cov = generate_cov(nvar, corr)
    C = np.linalg.cholesky(cov)

    X = np.transpose(np.dot(C, np.transpose(normvars)))
    return X


def randomize_treatment(N, prob=0.5):
    return np.random.binomial(1, prob, N).reshape([N, 1])


def generate_data(tau, N, p, corr=0.5):
    """p is the number of covariates"""
    X = generate_multnorm(N, corr, p)
    T = randomize_treatment(N)
    global beta
    global err
    err = np.random.normal(0, 1, [N, 1])
    beta = np.random.normal(5, 5, [p, 1])

    Y = tau * T + X @ beta + err
    return Y, T, X


def randomized_experiment(tau, N, p, violate=False):
    Y, T, X = generate_data(tau, N, p)
    if violate == False:
        covars = np.concatenate([T, X], axis=1)
    # violate here meanse controlling covariates
    if violate:
        covars = np.concatenate([T], axis=1)
    mod = sm.OLS(Y, covars)
    res = mod.fit()
    tauhat = res.params[0]
    se_tauhat = res.HC1_se[0]
    return tauhat, se_tauhat


def get_bias_rmse_size(true_value, estimate: list, standard_error: list, cval=1.96):
    R = len(estimate)

    b = estimate - np.ones([R, 1]) * true_value
    bias = np.mean(b)
    rmse = np.sqrt(np.mean(b ** 2))
    tval = b / standard_error
    size = np.mean(1 * (np.abs(tval) > cval))
    return bias, rmse, size
