import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from super_k.super_k import *

from scipy.optimize import minimize_scalar

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from tabulate import tabulate

from time import time


datasets = [
    "optdigits",
    ("USPS", 2),
    "satimage",
    "letter",
    "isolet",
    ]

test_sizes = {
    "optdigits": 1797,
    "USPS":      2007,
    "letter":    4000,
    "isolet":    1557,
    "default":   0.2
}

k_search_ranges = {
    "optdigits": range(5, 75, 5),
    "USPS": range(5, 75, 5),
    "satimage": range(5, 75, 5),
    "letter": range(5, 100, 5),
    "isolet": range(5, 75, 5),
    "default": range(5, 1000, 10)
}

superk_params = {
    "optdigits": {"k": 10},
    "USPS":      {"k": 40},
    "satimage":  {"k": 35},
    "letter":    {"k": 65},
    "isolet":    {"k": 5},
}


def tune_k(X, y, k_range):

    cache = {}

    def f(x):
        k = int(round(x))

        try:
            rate = cache[k]
        except:
            spk = SuperK(k)
            scores = cross_val_score(spk, X, y, cv=5)

            rate = scores.mean()
            cache[k] = rate

        error = (1.0 - rate) * 1e6 + k

        return error

    res = minimize_scalar(f, bounds=(min(k_range), max(k_range)), method='bounded', options={"disp": 3, "xatol": 1e-1})
    k = int(round(res.x))

    return k


def grid_search_k(X, y, k_range):

    params = {'k': k_range}
    spk = SuperK()
    clf = GridSearchCV(spk, params)
    clf.fit(X, y)
    best_k_index = clf.cv_results_["mean_test_score"].argmax()
    k = clf.cv_results_["param_k"][best_k_index]

    return k

def tvt_main():

    superk_accuracies = {}
    superk_train_times = {}
    superk_test_times = {}

    svmlinear_accuracies = {}
    svmlinear_train_times = {}
    svmlinear_test_times = {}

    svmrbf_accuracies = {}
    svmrbf_train_times = {}
    svmrbf_test_times = {}

    svmpoly_accuracies = {}
    svmpoly_train_times = {}
    svmpoly_test_times = {}

    knn_accuracies = {}
    knn_train_times = {}
    knn_test_times = {}



    for item in datasets:

        try:
            ds_name, ver = item
        except:
            ds_name = item
            ver = 1

        print("{} {}".format(ds_name, ver))
        dataset = fetch_openml(ds_name, version=ver)
        try:
            test_size = test_sizes[ds_name]
        except:
            test_size = test_sizes["default"]
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=test_size, shuffle=False)
        
        #### Super-k ####
        try:
            k = superk_params[ds_name]["k"]
        except:
            try:
                param_range = k_search_ranges[ds_name]
            except:
                param_range = k_search_ranges["default"]
            # k = tune_k(X_train, y_train, param_range)
            k = grid_search_k(X_train, y_train, param_range)

        spk = SuperK(k)
        t0 = time()
        for _ in range(10):
            spk.fit(X_train, y_train, verbose=True)
        t1 = time()
        for _ in range(10):
            accuracy = spk.score(X_test, y_test)
        t2 = time()

        superk_accuracies[ds_name] = accuracy
        superk_train_times[ds_name] = (t1 - t0) * 100
        superk_test_times[ds_name] = (t2 - t1) * 100

        print("{}\nSuper-k\n{}\nk: {}, n_genpts: {}\nTest score: {}\n{}".format("-" * 40, ds_name, k, spk.n_genpts, accuracy, "-" * 40))

        #### SVM Linear ####
        svm_linear = SVC(kernel="linear")
        t0 = time()
        for _ in range(10):
            svm_linear.fit(X_train, y_train)
        t1 = time()
        for _ in range(10):
            accuracy = svm_linear.score(X_test, y_test)
        t2 = time()

        svmlinear_accuracies[ds_name] = accuracy
        svmlinear_train_times[ds_name] = (t1 - t0) * 100
        svmlinear_test_times[ds_name] = (t2 - t1) * 100

        print("{}\nSVM Linear\n{}\nTest score: {}\n{}".format("-" * 40, ds_name, accuracy, "-" * 40))

        #### SVM RBF ####
        svm_rbf = SVC(kernel="rbf")
        t0 = time()
        for _ in range(10):
            svm_rbf.fit(X_train, y_train)
        t1 = time()
        for _ in range(10):
            accuracy = svm_rbf.score(X_test, y_test)
        t2 = time()

        svmrbf_accuracies[ds_name] = accuracy
        svmrbf_train_times[ds_name] = (t1 - t0) * 100
        svmrbf_test_times[ds_name] = (t2 - t1) * 100

        print("{}\nSVM RBF\n{}\nTest score: {}\n{}".format("-" * 40, ds_name, accuracy, "-" * 40))

        #### SVM Poly ####
        svm_poly = SVC(kernel="poly")
        t0 = time()
        for _ in range(10):
            svm_poly.fit(X_train, y_train)
        t1 = time()
        for _ in range(10):
            accuracy = svm_poly.score(X_test, y_test)
        t2 = time()

        svmpoly_accuracies[ds_name] = accuracy
        svmpoly_train_times[ds_name] = (t1 - t0) * 100
        svmpoly_test_times[ds_name] = (t2 - t1) * 100

        print("{}\nSVM Poly\n{}\nTest score: {}\n{}".format("-" * 40, ds_name, accuracy, "-" * 40))
        
        #### KNN ####
        knn = KNeighborsClassifier()
        t0 = time()
        for _ in range(10):
            knn.fit(X_train, y_train)
        t1 = time()
        for _ in range(10):
            accuracy = knn.score(X_test, y_test)
        t2 = time()

        knn_accuracies[ds_name] = accuracy
        knn_train_times[ds_name] = (t1 - t0) * 100
        knn_test_times[ds_name] = (t2 - t1) * 100

        print("{}\nKNN\n{}\nTest score: {}\n{}".format("-" * 40, ds_name, accuracy, "-" * 40))


    print("Super-k Results:\n{}\n{}\n{}\n".format(superk_accuracies, superk_train_times, superk_test_times))
    print("Linear SVM Results:\n{}\n{}\n{}\n".format(svmlinear_accuracies, svmlinear_train_times, svmlinear_test_times))
    print("SVM with RBF kernel results:\n{}\n{}\n{}\n".format(svmrbf_accuracies, svmrbf_train_times, svmrbf_test_times))
    print("SVM with Poly kernel results:\n{}\n{}\n{}\n".format(svmpoly_accuracies, svmpoly_train_times, svmpoly_test_times))
    print("KNN results:\n{}\n{}\n{}\n".format(knn_accuracies, knn_train_times, knn_test_times))

    accuracy_table = []

    accuracy_table.append([""] + ["{}".format(key) for key, _ in superk_accuracies.items()])
    accuracy_table.append(["Super-k"] + ["{:4.3f}".format(value) for _, value in superk_accuracies.items()])
    accuracy_table.append(["SVM Linear"] + ["{:4.3f}".format(value) for _, value in svmlinear_accuracies.items()])
    accuracy_table.append(["SVM RBF"] + ["{:4.3f}".format(value) for _, value in svmrbf_accuracies.items()])
    accuracy_table.append(["SVM Poly"] + ["{:4.3f}".format(value) for _, value in svmpoly_accuracies.items()])
    accuracy_table.append(["KNN"] + ["{:4.3f}".format(value) for _, value in knn_accuracies.items()])

    accuracy_latex = tabulate(accuracy_table, tablefmt="latex_raw")
    print(accuracy_latex)

    with open('accuracy_table.tex', "w") as f:
        f.write(accuracy_latex)

    train_times_table = []

    train_times_table.append([""] + ["{}".format(key) for key, _ in superk_train_times.items()])
    train_times_table.append(["Super-k"] + ["{:8.3f}".format(value) for _, value in superk_train_times.items()])
    train_times_table.append(["SVM Linear"] + ["{:8.3f}".format(value) for _, value in svmlinear_train_times.items()])
    train_times_table.append(["SVM RBF"] + ["{:8.3f}".format(value) for _, value in svmrbf_train_times.items()])
    train_times_table.append(["SVM Poly"] + ["{:8.3f}".format(value) for _, value in svmpoly_train_times.items()])
    train_times_table.append(["KNN"] + ["{:8.3f}".format(value) for _, value in knn_train_times.items()])

    train_times_latex = tabulate(train_times_table, tablefmt="latex_raw")
    print(train_times_latex)

    with open('train_times_table.tex', "w") as f:
        f.write(train_times_latex)

    test_times_table = []

    test_times_table.append([""] + ["{}".format(key) for key, _ in superk_test_times.items()])
    test_times_table.append(["Super-k"] + ["{:8.3f}".format(value) for _, value in superk_test_times.items()])
    test_times_table.append(["SVM Linear"] + ["{:8.3f}".format(value) for _, value in svmlinear_test_times.items()])
    test_times_table.append(["SVM RBF"] + ["{:8.3f}".format(value) for _, value in svmrbf_test_times.items()])
    test_times_table.append(["SVM Poly"] + ["{:8.3f}".format(value) for _, value in svmpoly_test_times.items()])
    test_times_table.append(["KNN"] + ["{:8.3f}".format(value) for _, value in knn_test_times.items()])

    test_times_latex = tabulate(test_times_table, tablefmt="latex_raw")
    print(test_times_latex)

    with open('test_times_table.tex', "w") as f:
        f.write(test_times_latex)

    

if __name__ == "__main__":
    tvt_main()

