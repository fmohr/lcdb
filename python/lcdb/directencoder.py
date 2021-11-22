import io
import numpy as np

class DirectEncoder:

    def __init__(self, precision = None):
        if precision is not None and type(precision) != int:
            raise Exception("Precision must be an int or None!")
        self.precision = precision
        
    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def decode_integer_list(self, encoding_str, chunk_size, expected_length):
        nums = []
        r = int(6 / chunk_size)
        for i, sym in enumerate(encoding_str):
            code = str(ord(sym))
            if i < len(encoding_str) - 1:
                code = code.rjust(6, "0")
            else: # last symbol
                expected_fields = expected_length - len(nums)
                code = code.rjust(r * expected_fields, "0")
            probs = [int(c) for c in CompressingEncoder.chunks(code, r)]
            nums += probs
        return nums

    def encode_label_vector(self, v):
        labels = [str(u) for u in np.unique(v)]
        return labels, [labels.index(str(l)) for l in v]

    def decode_label_vector(self, descriptor):
        label_names, labels = descriptor[0], descriptor[1]
        return np.array([label_names[i] for i in labels])
        
    def encode_distribution(self, arr):
        encoded = arr[:,:-1].astype("float32")
        if self.precision is not None:
            encoded = np.round(encoded, self.precision)
            if self.precision <= 2:
                encoded = np.round(encoded * (10**self.precision)).astype(int)
        return (self.precision, encoded.tolist())
    
    def decode_distribution(self, encoded):
        if encoded is None:
            raise Exception("No (None) distribution given!")
        precision, probs = encoded[0], np.array(encoded[1])
        if precision is not None and precision <= 2:
            probs = probs.astype("float32") / (10**precision)
        probs = np.column_stack([probs, 1 - np.sum(probs, axis=1)])
        
        if precision is not None: # if probs were rounded to a certain precision at encoding time, make sure that the recovered numbers do not encode things that are not there
            probs = np.round(probs, precision)
        return probs