#!/usr/bin/env python
import math
import random
import sys
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression, RidgeCV

OUT_LINEAR_REGRESSOR = "./data/linear_regressor"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("You have to provide the file with the paths of the images")
        print("for training and test.")
        print("{} <list_paths>".format(sys.argv[0]))
        sys.exit(1)

    with open(sys.argv[1], "r") as fpaths:
        lines = [(line.split()[0], line.split()[1]) for line in fpaths.read().splitlines()]

    X = []
    ind = 1
    for path, age in lines:
        tmp = []
        with open(path.replace(".jpg", "_fv"), "r") as f:
            l = [float(elem) for elem in (f.read().splitlines()[0]).split()]
            X.append(l)
            sys.stdout.write("\rloading: {}       ".format(ind))
            sys.stdout.flush()
        ind += 1

    print("\n")
    y = [float(elem[1]) for elem in lines]
    random.shuffle(X)
    split_index = int(len(X)*0.8)
    train = X[:split_index]
    train_y = y[:split_index]
    test = X[split_index:]
    test_y = y[split_index:]

    train = np.array(train)
    test = np.array(test)

    print("\nTraining Linear Regressor")
    #linear_regressor = LinearRegression(fit_intercept=False, normalize=True, copy_X=False)
    linear_regressor = RidgeCV()
    linear_regressor.fit(train, train_y)

    print("Finished Training.")
    print("Saving Trained model.")
    joblib.dump(linear_regressor, OUT_LINEAR_REGRESSOR)

    # Test
    linear_regressor = joblib.load(OUT_LINEAR_REGRESSOR)
    print("Testing trained model")
    mae = 0.0
    for i, elem in enumerate(test):
        age = linear_regressor.predict(elem)[0]
        mae += math.fabs(age - test_y[i])
        sys.stdout.write("\rMAE: {} - Pred: {} - Groundt: {}               ".format(mae / float(i+1), age, test_y[i]))
        sys.stdout.flush()
    print("\nFinished Testing.")
    sys.exit(0)
