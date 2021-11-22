import numpy as np
from compressingencoder import CompressingEncoder
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
import random
import time


class DistributionCompressingEncoder:
    
    def __init__(self, bins):
        if bins > 100:
            raise Exception("Maximum number of bins is 100.")
        self.bins = bins
        
    def compress_list(self, l, k):

        l = l.reshape(len(l), 1)

        def fun(x):
            start = time.time()
            diffs = l - x
            out = np.sum(np.min(np.abs(diffs), axis=1))
            #print(f"fun value {out} ready after {time.time() - start}")
            return out

        def der(x):
            start = time.time()
            jac = np.zeros_like(x)

            # batch gradient
            s = random.sample(list(l), min(len(l), 100))

            for e in s:
                distances = [np.abs(e - xi) for xi in x]
                index_of_active_threshold = np.argmin(distances)
                jac[index_of_active_threshold] += (x[index_of_active_threshold] - e)
            #print(f"Derivative ready after {time.time() - start}")
            return jac

        # get optimal thresholds
        thresholds_init = np.array([np.quantile(l, (i + 1) / (k + 1)) for i in range(k)]) # start with equal distribution

        # define conditions
        bounds = Bounds([min(l)] * k, [max(l)] * k)
        M = [[(-1 if j == i else (1 if j == i + 1 else 0)) for j in range(k)] for i in range(k - 1)]
        linear_constraint = LinearConstraint(M, [0.01] * (k-1), [np.inf] * (k-1))

        res = minimize(fun, thresholds_init, method='SLSQP', jac=der, constraints=[linear_constraint],bounds=bounds, options={"maxiter": 10})
        thresholds = sorted(list(res["x"]))

        # get new list
        l_new = [thresholds[np.argmin([np.abs(e - t) for t in thresholds])] for e in l]
        return l_new

    def compress_distribution(self, dist):
        out = []
        k = self.bins
        for col  in dist.T:
            if len(np.unique(col)) > k:
                out.append(self.compress_list(col, k))
            else:
                out.append(col)
        return np.array(out).T

    def encode_label_vector(self, v):
        return CompressingEncoder.encode_label_vector(v)

    def decode_label_vector(self, descriptor):
        return CompressingEncoder.decode_label_vector(descriptor)
        
    def encode_distribution(self, arr):
        
        # compress the distribution (reduction to 10 values per column)
        arr_compressed = self.compress_distribution(arr[:,:-1])
        
        # now create the index info
        encoding_table = []
        compressed_dist = []
        for col in arr_compressed.T:
            values_in_col = sorted(list(np.unique(col)))
            new_col = [values_in_col.index(v) for v in col]
            encoding_table.append(values_in_col)
            compressed_dist.append(new_col)
        compressed_dist = np.array(compressed_dist).T
        
        # create the word describing the distribution
        R = list(np.ravel(compressed_dist))
        encoding_str = CompressingEncoder.encode_integer_list(R, 6 if self.bins <= 10 else 3)
        return {"p": encoding_str, "r": arr.shape[0], "c": arr.shape[1] - 1, "e": encoding_table}
    
    def decode_distribution(self, descriptor):
        
        # decode this thing
        encoding_str = descriptor["p"]
        rows = descriptor["r"]
        cols = descriptor["c"]
        encoding_table = descriptor["e"]

        # extract numbers
        nums = CompressingEncoder.decode_integer_list(encoding_str, 6 if self.bins <= 10 else 3, rows * cols)

        # reshape
        probs_encoded = np.array(nums).reshape((rows, cols))
        
        # recover "original" probabilities
        probs_decoded = []
        for i, col in enumerate(probs_encoded.T):
            probs_decoded.append([encoding_table[i][v] for v in col])
        probs_decoded = np.array(probs_decoded).T
        probs_decoded = np.column_stack([probs_decoded, 1 - np.sum(probs_decoded, axis=1)])
        
        # return recovered probabilities
        return probs_decoded