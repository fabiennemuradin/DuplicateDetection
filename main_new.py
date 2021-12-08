import json
import random
import re
import sys
import time
import numpy as np
from math import comb
from random import shuffle
from sympy import nextprime
from LSH import convert_binary_old, minhash, lsh, common_count
from data_loader import load

###################################################################################     
#################################### DATA #########################################
###################################################################################

def load(file_path):
    """
    Loads and cleans a JSON file of product occurences that are grouped by model ID (as in the TVs.JSON example).
    :param file_path: the file path to a JSON file
    :return: a cleaned list of the data (as if we do not know the model type), and a binary matrix with element (i, j)
    equal to one if item i and item j are duplicates.
    """

    # Load data into dictionary.
    with open(file_path, "r") as file:
        data = json.load(file)

    # Declare common value representations to be replaced by the last value of the list.
    inch = ["Inch", "inches", "\"", "-inch", "-Inch", " inch", " Inch", "inch"]
    hz = ["Hertz", "hertz", "Hz", "HZ", " hz", "-hz", " Hz", "hz"]
    lbs = ["pounds", " pounds", "lb.", "lbs.", "lb", " lbs", " lb", "lbs"]
    to_replace = [inch, hz, lbs]
    replacements = dict()
    for replace_list in to_replace:
        replacement = replace_list[-1]
        values = replace_list[0:-1]
        for value in values:
            replacements[value] = replacement

    # Clean data.
    clean_list = []
    for model in data:
        for occurence in data[model]:
            # Clean title.
            for value in replacements:
                occurence["title"] = occurence["title"].replace(value, replacements[value])

            # Clean features map.
            features = occurence["featuresMap"]
            for key in features:
                for value in replacements:
                    features[key] = features[key].replace(value, replacements[value])
            clean_list.append(occurence)

    # Normally we will not be given adjacent duplicates, so shuffle the item list.
    shuffle(clean_list)

    # Compute binary matrix of duplicates, where element (i, j) is one if i and j are duplicates, for i != j, and zero
    # otherwise. Note that this matrix will be symmetric.
    duplicates = np.zeros((len(clean_list), len(clean_list)))
    for i in range(len(clean_list)):
        model_i = clean_list[i]["modelID"]
        for j in range(i + 1, len(clean_list)):
            model_j = clean_list[j]["modelID"]
            if model_i == model_j:
                duplicates[i][j] = 1
                duplicates[j][i] = 1
    return clean_list, duplicates.astype(int)


###################################################################################     
#################################### LSH ##########################################
###################################################################################

def convert_binary_old(data):
    """
    Transforms a list of items to a binary vector product representation, using model words in the title and decimals in
    the feature values.
    NOTE. This is the old implementation by Hartveld et al. (2018), implemented for evaluation purposes.
    :param data: a list of items
    :return: a binary vector product representation
    """

    # For computational efficiency, we keep all model words as keys in a dictionary, where its value is the
    # corresponding row in the binary vector product representation.
    model_words = dict()
    binary_vec = []

    # Loop through all items to find model words.
    for i in range(len(data)):
        item = data[i]
        # Find model words in the title.
        # [a-zA-Z0-9]* matches any alphanumeric character (zero or more times).
        # (?:[0-9]+[^0-9, ]+) (incorrectly) matches any (numeric) - (non numeric) combination.
        # (?:[^0-9, ]+[0-9]+) (incorrectly) matches any (non numeric) - (numeric) combination.
        # [a-zA-Z0-9]* matches any alphanumeric character (zero or more times).
        mw_title = re.findall("([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9, ]+)|(?:[^0-9, ]+[0-9]+))[a-zA-Z0-9]*)", item["title"])
        item_mw = mw_title

        # Find model words in the key-value pairs.
        features = item["featuresMap"]
        for key in features:
            value = features[key]

            # Find decimals.
            # (?:(^[0-9]+(?:\.[0-9]+))[a-zA-Z]+$) matches any (numeric) - . - (numeric) - (non-numeric) combination (
            # i.e., decimals).
            # (^[0-9](?:\.[0-9]+)$)) matches any (numeric) - . - (numeric) combination (i.e., decimals).
            # [a-zA-Z0-9]+ matches any alphanumeric character (one or more times).
            mw_decimal = re.findall("(?:((?:^[0-9]+(?:\.[0-9]+))[a-zA-Z]+$)|(^[0-9](?:\.[0-9]+)$))", value)
            for decimal in mw_decimal:
                for group in decimal:
                    if group != "":
                        item_mw.append(group)

        # Loop through all identified model words and update the binary vector product representation.
        for mw in item_mw:
            if mw in model_words:
                # Set index for model word to one.
                row = model_words[mw]
                binary_vec[row][i] = 1
            else:
                # Add model word to the binary vector, and set index to one.
                binary_vec.append([0] * len(data))
                binary_vec[len(binary_vec) - 1][i] = 1

                # Add model word to the dictionary.
                model_words[mw] = len(binary_vec) - 1
    return binary_vec


def minhash(binary_vec, n):
    """
    Computes a MinHash signature matrix using n random hash functions, which will result in an n x c signature matrix,
    where c is the number of columns (items). These random hash functions are of the form (a + bx) mod k, where a and b
    are randomly generated integers, k is the smallest prime number that is larger than or equal to r (the original
    number of rows in the r x c binary vector). We use quick, vectorized, numpy operations to substantially reduce
    computation time.
    :param binary_vec: a binary vector product representation
    :param n: the number of rows in the new signature matrix
    :return: the signature matrix (a NumPy array)
    """

    random.seed(1)

    r = len(binary_vec)
    c = len(binary_vec[0])
    binary_vec = np.array(binary_vec)

    # Find k.
    k = nextprime(r - 1)

    # Generate n random hash functions.
    hash_params = np.empty((n, 2))
    for i in range(n):
        # Generate a, b, and k.
        a = random.randint(1, k - 1)
        b = random.randint(1, k - 1)
        hash_params[i, 0] = a
        hash_params[i, 1] = b

    # Initialize signature matrix to infinity for each element.
    signature = np.full((n, c), np.inf)

    # Loop through the binary vector representation matrix once, to compute the signature matrix.
    for row in range(1, r + 1):
        # Compute each of the n random hashes once for each row.
        e = np.ones(n)
        row_vec = np.full(n, row)
        x = np.stack((e, row_vec), axis=1)
        row_hash = np.sum(hash_params * x, axis=1) % k

        for i in range(n):
            # Update column j if and only if it contains a one and its current value is larger than the hash value for
            # the signature matrix row i.
            updates = np.where(binary_vec[row - 1] == 0, np.inf, row_hash[i])
            signature[i] = np.where(updates < signature[i], row_hash[i], signature[i])
    return signature.astype(int)


def lsh(signature, t):
    """
    Performs Locality Sensitive Hashing (LSH) based on a previously obtained MinHash matrix.
    :param signature: the MinHash signature matrix
    :param t: the approximate threshold value at which Pr[candidate] =~ 1/2
    :return: a binary matrix with a one if two elements are candidate pairs, and zero otherwise
    """

    n = len(signature)

    # Compute the approximate number of bands and rows from the threshold t, using that n = r * b, and t is
    # approximately (1/b)^(1/r).
    r_best = 1
    b_best = 1
    best = 1
    for r in range(1, n + 1):
        for b in range(1, n + 1):
            if r * b == n:
                # Valid pair.
                approximation = (1 / b) ** (1 / r)
                if abs(approximation - t) < abs(best - t):
                    best = approximation
                    r_best = r
                    b_best = b

    candidates = np.zeros((len(signature[0]), len(signature[0])))
    for band in range(b_best):
        buckets = dict()
        start_row = r_best * band  # Inclusive.
        end_row = r_best * (band + 1)  # Exclusive.
        strings = ["".join(signature[start_row:end_row, column].astype(str)) for column in range(len(signature[0]))]
        ints = [int(string) for string in strings]
        hashes = [integer % sys.maxsize for integer in ints]

        # Add all item hashes to the correct bucket.
        for item in range(len(hashes)):
            hash_value = hashes[item]
            if hash_value in buckets:

                # All items already in this bucket are possible duplicates of this item.
                for candidate in buckets[hash_value]:
                    candidates[item, candidate] = 1
                    candidates[candidate, item] = 1
                buckets[hash_value].append(item)
            else:
                buckets[hash_value] = [item]
    return candidates.astype(int)


def common_count(data):
    """
    Finds and reports the most common count features.
    :param data: a list of items
    :return:
    """
    feature_count = dict()

    # Loop through all items to identify common count features.
    for i in range(len(data)):
        item = data[i]
        features = item["featuresMap"]

        for key in features:
            value = features[key]

            count = re.match("^[0-9]+$", value)
            if count is not None:
                if key in feature_count:
                    feature_count[key] += 1
                else:
                    feature_count[key] = 1

    count_list = [(v, k) for k, v in feature_count.items()]
    count_list.sort(reverse=True)
    for feature in count_list:
        print(feature[1], feature[0])


###################################################################################     
#################################### MAIN #########################################
###################################################################################

def main():
    """
    Runs the whole MSMP++ procedure, and stores results in a csv file.
    :return:
    """

    identify_common_count = False
    run_lsh = True
    write_result = True

    thresholds = [x / 100 for x in range(5, 100, 5)]
    bootstraps = 5
    random.seed(0)

    file_path = "/Users/fabiennemuradin/python-workspace/CSvenv/TVs-all-merged.json"
    result_path = "/Users/fabiennemuradin/python-workspace/results/"

    start_time = time.time()

    data_list, duplicates = load(file_path)

    if identify_common_count:
        common_count(data_list)

    if run_lsh:
        if write_result:
            with open(result_path + "results_new.csv", 'w') as out:
                out.write(
                    "t,comparisons,pq,pc,f1,comparisons_alt,pq_alt,pc_alt,f1_alt,comparisons_old,pq_old,pc_old,"
                    "f1_old\n")

        for t in thresholds:
            print("t = ", t)

            # Initialize statistics, where results = [comparisons, pq, pc, f1].
            results_old = np.zeros(4)

            for run in range(bootstraps):
                data_sample, duplicates_sample = bootstrap(data_list, duplicates)
                comparisons_old_run, pq_old_run, pc_old_run, f1_old_run = do_lsh_old(data_sample, duplicates_sample, t)
                results_old += np.array([comparisons_old_run, pq_old_run, pc_old_run, f1_old_run])

            # Compute average statistics over all bootstraps.
            statistics_old = results_old / bootstraps

            if write_result:
                with open(result_path + "results_new.csv", 'a') as out:
                    out.write(str(t))
                    for stat in statistics_old:
                        out.write("," + str(stat))
                    out.write("\n")

    end_time = time.time()
    print("Run time:", end_time - start_time, "seconds")


def do_lsh_old(data_list, duplicates, t):
    """
    Bins items using MinHash and LSH, and computes and returns performance metrics based on the matrix of true
    duplicates.
    NOTE. This is the old implementation by Hartveld et al. (2018), implemented for evaluation purposes.
    :param data_list: a list of items
    :param duplicates: a binary matrix where item (i, j) is equal to one if items i and j are duplicates, and zero
    otherwise
    :param t: the threshold value
    :return: the fraction of comparisons, pair quality, pair completeness, and F_1 measure
    """

    binary_vec = convert_binary_old(data_list)
    n = round(round(0.5 * len(binary_vec)) / 100) * 100
    signature = minhash(binary_vec, n)
    candidates = lsh(signature, t)

    # Compute number of comparisons.
    comparisons = np.sum(candidates) / 2
    comparison_frac = comparisons / comb(len(data_list), 2)

    # Compute matrix of correctly binned duplicates, where element (i, j) is equal to one if item i and item j are
    # duplicates, and correctly classified as such by LSH.
    correct = np.where(duplicates + candidates == 2, 1, 0)
    n_correct = np.sum(correct) / 2

    # Compute Pair Quality (PQ)
    pq = n_correct / comparisons

    # Compute Pair Completeness (PC)
    pc = n_correct / (np.sum(duplicates) / 2)

    # Compute F_1 measure.
    f1 = 2 * pq * pc / (pq + pc)

    return comparison_frac, pq, pc, f1

def bootstrap(data_list, duplicates):
    """
    Creates a bootstrap by sampling n elements from the data with replacement, where n denotes the size of the original
    dataset.
    :param data_list: a list of data
    :param duplicates: a binary matrix where item (i, j) is equal to one if items i and j are duplicates, and zero
    otherwise
    :return: a bootstrap sample of the data and the corresponding duplicate matrix
    """

    # Compute indices to be included in the bootstrap.
    indices = [random.randint(x, len(data_list) - 1) for x in [0] * len(data_list)]

    # Collect samples.
    data_sample = [data_list[index] for index in indices]
    duplicates_sample = np.take(np.take(duplicates, indices, axis=0), indices, axis=1)
    return data_sample, duplicates_sample


if __name__ == '__main__':
    main()

