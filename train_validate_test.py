import numpy as np
import matplotlib.pyplot as plt

from super_k.super_k import *

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from tabulate import tabulate
import timeit

N_REPEAT = 10
N_NUMBER = 1

TIME_SCALE = 1e-3 # milliseconds

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


def grid_search_k(X, y, k_range):

    params = {'k': k_range}
    spk = SuperK()
    clf = GridSearchCV(spk, params)
    clf.fit(X, y)
    best_k_index = clf.cv_results_["mean_test_score"].argmax()
    k = clf.cv_results_["param_k"][best_k_index]

    return k


def tvt_main():

    classifiers = ["superk", "svmlinear", "svmrbf", "svmpoly", "knn"]

    accuracies = {clfier : {} for clfier in classifiers} 
    train_times = {clfier : {} for clfier in classifiers} 
    test_times = {clfier : {} for clfier in classifiers} 

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
            k = grid_search_k(X_train, y_train, param_range)

        clf = SuperK(k)

        train_timer = timeit.Timer(lambda : clf.fit(X_train, y_train, verbose=True))
        train_results = train_timer.repeat(repeat=N_REPEAT, number=N_NUMBER)

        accuracy = None
        def f():
            nonlocal accuracy
            accuracy = clf.score(X_test, y_test)
        test_timer = timeit.Timer(f)
        test_results = test_timer.repeat(repeat=N_REPEAT, number=N_NUMBER)

        accuracies["superk"][ds_name] = accuracy
        train_times["superk"][ds_name] = (np.mean(train_results) / TIME_SCALE, np.std(train_results) / TIME_SCALE)
        test_times["superk"][ds_name] = (np.mean(test_results) / TIME_SCALE, np.std(test_results) / TIME_SCALE)

        print("{}\nSuper-k\n{}\nk: {}, n_genpts: {}\nTest score: {}\n{}".format("-" * 40, ds_name, k, clf.n_genpts, accuracy, "-" * 40))

        #### SVM Linear ####
        clf = SVC(kernel="linear")

        train_timer = timeit.Timer(lambda : clf.fit(X_train, y_train))
        train_results = train_timer.repeat(repeat=N_REPEAT, number=N_NUMBER)

        accuracy = None
        def f():
            nonlocal accuracy
            accuracy = clf.score(X_test, y_test)
        test_timer = timeit.Timer(f)
        test_results = test_timer.repeat(repeat=N_REPEAT, number=N_NUMBER)

        accuracies["svmlinear"][ds_name] = accuracy
        train_times["svmlinear"][ds_name] = (np.mean(train_results) / TIME_SCALE, np.std(train_results) / TIME_SCALE)
        test_times["svmlinear"][ds_name] = (np.mean(test_results) / TIME_SCALE, np.std(test_results) / TIME_SCALE)

        print("{}\nSVM Linear\n{}\nTest score: {}\n{}".format("-" * 40, ds_name, accuracy, "-" * 40))

        #### SVM RBF ####
        clf = SVC(kernel="rbf")

        train_timer = timeit.Timer(lambda : clf.fit(X_train, y_train))
        train_results = train_timer.repeat(repeat=N_REPEAT, number=N_NUMBER)

        accuracy = None
        def f():
            nonlocal accuracy
            accuracy = clf.score(X_test, y_test)
        test_timer = timeit.Timer(f)
        test_results = test_timer.repeat(repeat=N_REPEAT, number=N_NUMBER)

        accuracies["svmrbf"][ds_name] = accuracy
        train_times["svmrbf"][ds_name] = (np.mean(train_results) / TIME_SCALE, np.std(train_results) / TIME_SCALE)
        test_times["svmrbf"][ds_name] = (np.mean(test_results) / TIME_SCALE, np.std(test_results) / TIME_SCALE)

        print("{}\nSVM RBF\n{}\nTest score: {}\n{}".format("-" * 40, ds_name, accuracy, "-" * 40))

        #### SVM Poly ####
        clf = SVC(kernel="poly")

        train_timer = timeit.Timer(lambda : clf.fit(X_train, y_train))
        train_results = train_timer.repeat(repeat=N_REPEAT, number=N_NUMBER)

        accuracy = None
        def f():
            nonlocal accuracy
            accuracy = clf.score(X_test, y_test)
        test_timer = timeit.Timer(f)
        test_results = test_timer.repeat(repeat=N_REPEAT, number=N_NUMBER)

        accuracies["svmpoly"][ds_name] = accuracy
        train_times["svmpoly"][ds_name] = (np.mean(train_results) / TIME_SCALE, np.std(train_results) / TIME_SCALE)
        test_times["svmpoly"][ds_name] = (np.mean(test_results) / TIME_SCALE, np.std(test_results) / TIME_SCALE)

        print("{}\nSVM Poly\n{}\nTest score: {}\n{}".format("-" * 40, ds_name, accuracy, "-" * 40))
        
        #### KNN ####
        clf = KNeighborsClassifier()

        train_timer = timeit.Timer(lambda : clf.fit(X_train, y_train))
        train_results = train_timer.repeat(repeat=N_REPEAT, number=N_NUMBER)

        accuracy = None
        def f():
            nonlocal accuracy
            accuracy = clf.score(X_test, y_test)
        test_timer = timeit.Timer(f)
        test_results = test_timer.repeat(repeat=N_REPEAT, number=N_NUMBER)

        accuracies["knn"][ds_name] = accuracy
        train_times["knn"][ds_name] = (np.mean(train_results) / TIME_SCALE, np.std(train_results) / TIME_SCALE)
        test_times["knn"][ds_name] = (np.mean(test_results) / TIME_SCALE, np.std(test_results) / TIME_SCALE)

        print("{}\nKNN\n{}\nTest score: {}\n{}".format("-" * 40, ds_name, accuracy, "-" * 40))


    print("Super-k Results:\n{}\n{}\n{}\n".format(accuracies["superk"], train_times["superk"], test_times["superk"]))
    print("Linear SVM Results:\n{}\n{}\n{}\n".format(accuracies["svmlinear"], train_times["svmlinear"], test_times["svmlinear"]))
    print("SVM with RBF kernel results:\n{}\n{}\n{}\n".format(accuracies["svmrbf"], train_times["svmrbf"], test_times["svmrbf"]))
    print("SVM with Poly kernel results:\n{}\n{}\n{}\n".format(accuracies["svmpoly"], train_times["svmpoly"], test_times["svmpoly"]))
    print("KNN results:\n{}\n{}\n{}\n".format(accuracies["knn"], train_times["knn"], test_times["knn"]))

    accuracy_table = []

    accuracy_table.append([""] + ["{}".format(key) for key, _ in accuracies["superk"].items()])
    accuracy_table.append(["Super-k"] + ["{:.3f}".format(value) for _, value in accuracies["superk"].items()])
    accuracy_table.append(["SVM Linear"] + ["{:.3f}".format(value) for _, value in accuracies["svmlinear"].items()])
    accuracy_table.append(["SVM RBF"] + ["{:.3f}".format(value) for _, value in accuracies["svmrbf"].items()])
    accuracy_table.append(["SVM Poly"] + ["{:.3f}".format(value) for _, value in accuracies["svmpoly"].items()])
    accuracy_table.append(["KNN"] + ["{:.3f}".format(value) for _, value in accuracies["knn"].items()])

    accuracy_latex = tabulate(accuracy_table, tablefmt="latex_raw")
    print(accuracy_latex)

    with open('accuracy_table.tex', "w") as f:
        f.write(accuracy_latex)

    train_times_table = []

    train_times_table.append([""] + ["{}".format(key) for key, _ in train_times["superk"].items()])
    train_times_table.append(["Super-k"] + ["{:.1f}({:.1f})".format(*value) for _, value in train_times["superk"].items()])
    train_times_table.append(["SVM Linear"] + ["{:.1f}({:.1f})".format(*value) for _, value in train_times["svmlinear"].items()])
    train_times_table.append(["SVM RBF"] + ["{:.1f}({:.1f})".format(*value) for _, value in train_times["svmrbf"].items()])
    train_times_table.append(["SVM Poly"] + ["{:.1f}({:.1f})".format(*value) for _, value in train_times["svmpoly"].items()])
    train_times_table.append(["KNN"] + ["{:.1f}({:.1f})".format(*value) for _, value in train_times["knn"].items()])

    train_times_latex = tabulate(train_times_table, tablefmt="latex_raw")
    print(train_times_latex)

    with open('train_times_table.tex', "w") as f:
        f.write(train_times_latex)

    test_times_table = []

    test_times_table.append([""] + ["{}".format(key) for key, _ in test_times["superk"].items()])
    test_times_table.append(["Super-k"] + ["{:.1f}({:.1f})".format(*value) for _, value in test_times["superk"].items()])
    test_times_table.append(["SVM Linear"] + ["{:.1f}({:.1f})".format(*value) for _, value in test_times["svmlinear"].items()])
    test_times_table.append(["SVM RBF"] + ["{:.1f}({:.1f})".format(*value) for _, value in test_times["svmrbf"].items()])
    test_times_table.append(["SVM Poly"] + ["{:.1f}({:.1f})".format(*value) for _, value in test_times["svmpoly"].items()])
    test_times_table.append(["KNN"] + ["{:.1f}({:.1f})".format(*value) for _, value in test_times["knn"].items()])

    test_times_latex = tabulate(test_times_table, tablefmt="latex_raw")
    print(test_times_latex)

    with open('test_times_table.tex', "w") as f:
        f.write(test_times_latex)


if __name__ == "__main__":
    tvt_main()

