from sklearn import linear_model


def main():
    class_ = 'all'
    subsample = 20


    # get labels, features
    X = get_features()
    y = get_labels()

    # learn
    clf = linear_model.SGDRegressor(alpha=0.0001, epsilon=0.1, eta0=0.01, fit_intercept=True,
       l1_ratio=0.15, learning_rate='invscaling', loss='squared_loss',
       n_iter=5, penalty='l2', power_t=0.25, random_state=None,
       shuffle=False, verbose=0, warm_start=False)
    clf.fit(X, y)

    # plot


def get_features():
    pass


def get_labels():
    pass



if __name__ == "__main__":
    main()