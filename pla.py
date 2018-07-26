#!/usr/bin/env python
# encoding: utf-8
from pandas import read_csv
import numpy as np
from tqdm import tqdm


def predict(x, w):
    yhat = w.T.dot(np.array(x))
    result = -1 if yhat == 0.0 else np.sign(yhat)
    return result


def check_error_pocket(data, w):
    error = 0
    error_samples = []
    for x,y in data:
        if predict(x, w) != float(y):
            error += 1
            error_samples.append((x, y))
    rand_index = np.random.choice(range(len(error_samples)))
    x, y = error_samples[rand_index]
    result = x, y, error
    return result


def pocket_pla(data, limit=50):
    w = np.zeros(len(data[0][0]))
    min_error = 1000000
    np.random.shuffle(data)
    count = 0
    while count < limit:
        for x, y in data:
            # try to correct the weight
            if predict(x, w) != float(y):
                # check if corrected weight performs better
                w_tmp = w + np.array(x) * y
                err = check_error(data, w_tmp)
                if err < min_error:
                    min_error = err
                    w = w_tmp
                count+=1
    return w


def check_error(data, w):
    count = 0
    for x, y in data:
        if predict(x, w) != float(y):
            count += 1
    return count


def pla(data, eta, shuffle):
    if shuffle:
        np.random.shuffle(data)
    w = np.zeros(len(data[0][0]))
    count = 1
    while 1:
        stop = False
        for x, y in data:
            # try to correct the weight
            if predict(x, w) != float(y):
                w = w + np.array(x) * y * eta
                count += 1
                if count == 50:
                    stop = True
                    break
        # if there are not any error point existed, terminate it.
        error_count = check_error(data, w)
        print("error count:%d and update count:%d"%(error_count, count))
        if error_count == 0 or stop:
            break
    return w


def evaluate(algorithm, epochs=2000):
    total_error = 0
    for i in range(epochs):
        print("===== Epochs %d =====" % (i + 1))
        # training
        data = read_raw_input('/home/vincent/Public/hw1_18_train.dat')
        weights = pocket_pla(data) if algorithm == "pocket" else pla(data, 1, True)

        # testing
        data = read_raw_input('/home/vincent/Public/hw1_18_test.dat')
        error = check_error(data, weights)
        total_error += (float(error) / len(data))
        print("error:%.3f" % (float(error) / len(data)))
    print("avg error:%.3f" % (total_error / epochs))


def read_raw_input(path):
    raw = read_csv(path, delim_whitespace=True)
    X = raw.iloc[:, :-1]
    X.insert(0, "bias", 1)
    Y = raw.iloc[:, -1:]
    return np.array(zip(X.values, Y.values))

if __name__ == '__main__':
    #evaluate("pocket")
    dataset = read_raw_input('/home/vincent/Public/hw1_15_train.dat')
    # question 15
    #pla(dataset, 1, True)

    # question 16 and 17
    #epochs = 2000
    #print("avg of update count %d" % (sum([pla(dataset, 0.5, True) for i in tqdm(range(epochs))])/float(epochs)))

    # question 18
    evaluate("pocket")