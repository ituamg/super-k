import numpy as np

from super_k.super_k import *

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
import timeit
import pickle
import time


N_REPEAT = 10
TIME_SCALE = 1e-3 # milliseconds
CV_FOLD = 5

classifiers = ["superk", "linearsvm", "svmlinear", "svmrbf", "svmpoly", "knn"]

datasets = [ # ds_name,     ds_version, test_size
              ("optdigits", 1,          1797),
              ("USPS",      2,          2007),
              ("satimage",  1,          1286),
              ("letter",    1,          4000),
              ("isolet",    1,          1557)
            ]

saved_params_file = "algo_params.pickle"

def tvt_main():

    print("\n{}\nStarting...\n{}".format("#" * 40, time.strftime("%a, %d %b %Y %H:%M:%S")))

    #### If exists load the parameters ####
    algo_params_changed = False
    try:
        with open(saved_params_file, 'rb') as f:
            algo_params = pickle.load(f)
    except:
        algo_params = {}

    for clfier in classifiers:
        if not clfier in algo_params:
            algo_params[clfier] = {}
        for ds_name, _, _ in datasets:
            if not ds_name in algo_params[clfier]:
                algo_params[clfier][ds_name] = {}

    accuracies = {clfier : {} for clfier in classifiers} 
    train_times = {clfier : {} for clfier in classifiers} 
    test_times = {clfier : {} for clfier in classifiers}

    def print_results(clfier, ds_name):
        print("{}\nAccuracy: {}\nTraining Time: {}\nInference Time: {}\n{}".format("-" * 40,
                                                                                    accuracies[clfier][ds_name], 
                                                                                    train_times[clfier][ds_name], 
                                                                                    test_times[clfier][ds_name], 
                                                                                    "-" * 40))

    for ds_name, ds_version, test_size in datasets:
        print("\n{}\nDatabase: {}, Version: {}\n{}".format("#" * 40, ds_name, ds_version, "#" * 40))

        #### Fetch and prepare the dataset ####
        dataset = fetch_openml(name=ds_name, version=ds_version)
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=test_size, shuffle=False)

        def train_test_save(clf, clfier):
            MINIMUM_DURATION = 1.0 # seconds
            
            def f_train():
                clf.fit(X_train, y_train)
            train_timer = timeit.Timer(f_train)

            # Initial run for autoranging
            t_init = train_timer.repeat(repeat=1, number=1).pop()
            # Determine number of runs for minimum duration
            n_train_runs = ceil(MINIMUM_DURATION / t_init)

            train_results = train_timer.repeat(repeat=N_REPEAT, number=n_train_runs)

            accuracy = None
            def f_test():
                nonlocal accuracy
                accuracy = clf.score(X_test, y_test)
            test_timer = timeit.Timer(f_test)

            # Initial run for autoranging
            t_init = test_timer.repeat(repeat=1, number=1).pop()
            # Determine number of runs for minimum duration
            n_test_runs = ceil(MINIMUM_DURATION / t_init)

            test_results = test_timer.repeat(repeat=N_REPEAT, number=n_test_runs)

            accuracies[clfier][ds_name] = accuracy
            train_times[clfier][ds_name] = (np.mean(train_results) / (n_train_runs * TIME_SCALE), 
                                            np.std(train_results) / (n_train_runs * TIME_SCALE))
            test_times[clfier][ds_name] = (np.mean(test_results) / (n_test_runs * TIME_SCALE),
                                           np.std(test_results) / (n_test_runs * TIME_SCALE))

        #### Super-k ####
        print("\n{:#^40}".format(" Super-k "))
        clfier = "superk"
        try:
            # Retrieve previously determined k
            k = algo_params[clfier][ds_name]["k"]
            print("Using previously determined k =", k)
        except:
            # Determine k
            print("Determining k")
            clf = GridSearchCV(estimator=SuperK(),
                               param_grid={"k" : np.geomspace(5, 500, 25, dtype=int)}, 
                               cv=CV_FOLD)
            clf.fit(X_train, y_train)
            k = clf.best_params_["k"]
            print("Using newly determined k =", k)
            algo_params[clfier][ds_name]["k"] = k
            algo_params_changed = True

        clf = SuperK(k=k)
        train_test_save(clf, clfier)
        print("n_genpts: {}".format(clf.n_genpts))
        print_results(clfier, ds_name)

        #### Linear SVM ####
        print("\n{:#^40}".format(" Linear SVM "))
        clfier = "linearsvm"
        try:
            # Retrieve previously determined C
            C = algo_params[clfier][ds_name]["C"]
            print("Using previously determined C =", C)
        except:
            # Determine C
            print("Determining C")
            clf = GridSearchCV(estimator=LinearSVC(dual=False, random_state=0, tol=1e-5),
                               param_grid={"C": np.logspace(-2, 1, 4)}, 
                               cv=CV_FOLD, n_jobs=-1)
            clf.fit(X_train, y_train)
            C = clf.best_params_["C"]
            print("Using newly determined C =", C)
            algo_params[clfier][ds_name]["C"] = C
            algo_params_changed = True

        clf = LinearSVC(dual=False, C=C, random_state=0, tol=1e-5)
        train_test_save(clf, clfier)
        print_results(clfier, ds_name)

        #### SVM Linear ####
        print("\n{:#^40}".format(" SVM Linear "))
        clfier = "svmlinear"
        try:
            # Retrieve previously determined C
            C = algo_params[clfier][ds_name]["C"]
            print("Using previously determined C =", C)
        except:
            # Determine C
            print("Determining C")
            clf = GridSearchCV(estimator=SVC(kernel="linear"),
                               param_grid={"C": np.logspace(-2, 1, 4)}, 
                               cv=CV_FOLD, n_jobs=-1)
            clf.fit(X_train, y_train)
            C = clf.best_params_["C"]
            print("Using newly determined C =", C)
            algo_params[clfier][ds_name]["C"] = C
            algo_params_changed = True

        clf = SVC(kernel="linear", C=C)
        train_test_save(clf, clfier)
        print_results(clfier, ds_name)

        #### SVM RBF ####
        print("\n{:#^40}".format(" SVM RBF "))
        clfier = "svmrbf"
        try:
            # Retrieve previously determined C and gamma
            C = algo_params[clfier][ds_name]["C"]
            gamma = algo_params[clfier][ds_name]["gamma"]
            print("Using previously determined C =", C, "and gamma =", gamma)
        except:
            # Determine C and gamma
            print("Determining C and gamma")
            clf = GridSearchCV(estimator=SVC(kernel="rbf"),
                               param_grid={"C" : np.logspace(-2, 1, 4), 
                                           "gamma" : np.logspace(-3, 0, 4)}, 
                               cv=CV_FOLD, n_jobs=-1)
            clf.fit(X_train, y_train)
            C = clf.best_params_["C"]
            gamma = clf.best_params_["gamma"]
            print("Using newly determined C =", C, "and gamma =", gamma)
            algo_params[clfier][ds_name]["C"] = C
            algo_params[clfier][ds_name]["gamma"] = gamma
            algo_params_changed = True

        clf = SVC(kernel="rbf", C=C, gamma=gamma)
        train_test_save(clf, clfier)
        print_results(clfier, ds_name)

        #### SVM Poly ####
        print("\n{:#^40}".format(" SVM Poly "))
        clfier = "svmpoly"
        try:
            # Retrieve previously determined C and gamma
            C = algo_params[clfier][ds_name]["C"]
            gamma = algo_params[clfier][ds_name]["gamma"]
            print("Using previously determined C =", C, "and gamma =", gamma)
        except:
            # Determine C and gamma
            print("Determining C and gamma")
            clf = GridSearchCV(estimator=SVC(kernel="poly"),
                               param_grid={"C" : np.logspace(-2, 1, 4), 
                                           "gamma" : np.logspace(-3, 0, 4)}, 
                               cv=CV_FOLD, n_jobs=-1)
            clf.fit(X_train, y_train)
            C = clf.best_params_["C"]
            gamma = clf.best_params_["gamma"]
            print("Using newly determined C =", C, "and gamma =", gamma)
            algo_params[clfier][ds_name]["C"] = C
            algo_params[clfier][ds_name]["gamma"] = gamma
            algo_params_changed = True

        clf = SVC(kernel="poly", C=C, gamma=gamma)
        train_test_save(clf, clfier)
        print_results(clfier, ds_name)
        
        #### KNN ####
        print("\n{:#^40}".format(" KNN "))
        clfier = "knn"
        try:
            # Retrieve previously determined k
            k = algo_params[clfier][ds_name]["k"]
            print("Using previously determined k =", k)
        except:
            # Determine k
            print("Determining k")
            clf = GridSearchCV(estimator=KNeighborsClassifier(algorithm="auto"),
                               param_grid={"n_neighbors" : range(1, 11, 2)}, 
                               cv=CV_FOLD, n_jobs=-1)
            clf.fit(X_train, y_train)
            k = clf.best_params_["n_neighbors"]
            print("Using newly determined k =", k)
            algo_params[clfier][ds_name]["k"] = k
            algo_params_changed = True

        clf = KNeighborsClassifier(algorithm="auto", n_neighbors=k, n_jobs=-1)
        train_test_save(clf, clfier)
        print_results(clfier, ds_name)

    # Save estimator parameters for later reuse
    if algo_params_changed:
        with open(saved_params_file, 'wb') as f:
            pickle.dump(algo_params, f, pickle.HIGHEST_PROTOCOL)

    #### Results ####
    print("\n{:#^40}".format(" Results "))

    row_names = {"superk" : "Super-k",
                 "linearsvm" : "Linear SVM",
                 "svmlinear" : "SVM Linear",
                 "svmrbf" : "SVM RBF",
                 "svmpoly" : "SVM Poly",
                 "knn" : "KNN"}

    table_header = [""] + ["{}".format(ds_name) for ds_name, _, _ in datasets]

    def accuracy_row(clfier):
        return [row_names[clfier]] + ["{:.3f}".format(value) for _, value in accuracies[clfier].items()]

    accuracy_table = [table_header] + [accuracy_row(clfier) for clfier in classifiers]
    accuracy_latex = tabulate(accuracy_table, tablefmt="latex_raw")
    print("\nAccuracy Table\n{}\n".format(accuracy_latex))

    with open('accuracy_table.tex', "w") as f:
        f.write(accuracy_latex)

    def train_row(clfier):
        return [row_names[clfier]] + ["{:.1f}{{\\small ({:.1f})}}".format(*value) for _, value in train_times[clfier].items()]

    train_times_table = [table_header] + [train_row(clfier) for clfier in classifiers]
    train_times_latex = tabulate(train_times_table, tablefmt="latex_raw")
    print("\nTraining Times\n{}\n".format(train_times_latex))

    with open('train_times_table.tex', "w") as f:
        f.write(train_times_latex)

    def test_row(clfier):
        return [row_names[clfier]] + ["{:.1f}{{\\small ({:.1f})}}".format(*value) for _, value in test_times[clfier].items()]

    test_times_table = [table_header] + [test_row(clfier) for clfier in classifiers]
    test_times_latex = tabulate(test_times_table, tablefmt="latex_raw")
    print("\nInference Times\n{}\n".format(test_times_latex))

    with open('test_times_table.tex', "w") as f:
        f.write(test_times_latex)

    print("Finished: {}".format(time.strftime("%a, %d %b %Y %H:%M:%S")))


if __name__ == "__main__":
    tvt_main()

