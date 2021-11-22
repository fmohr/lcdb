import numpy as np

class CompressingEncoder:

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    

    def encode_integer_list(self, l, chunk_size):
        C = list(self.chunks(l, chunk_size))
        chunks_as_strings = []
        for e in C:
            chunks_as_strings.append("".join([str(v).rjust(int(6 / chunk_size), "0") for v in e]))

        chunks_as_numbers = [int(c) for c in chunks_as_strings]
        encoding_str = "".join([str(chr(c)) for c in chunks_as_numbers])
        return encoding_str

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
            probs = [int(c) for c in self.chunks(code, r)]
            nums += probs
        return nums

    def encode_label_vector(self, v):
        labels = list(np.unique(v))
        v_int = [labels.index(l) for l in v]

        chunk_size = 6 if len(labels) < 10 else 3
        encoding = self.encode_integer_list(v_int, chunk_size)
        return {"l": labels, "v": encoding, "s": len(v), "c": chunk_size}

    def decode_label_vector(self, descriptor):
        labels = descriptor["l"]
        v_int = self.decode_integer_list(descriptor["v"], descriptor["c"], descriptor["s"])
        return [labels[i] for i in v_int]
        
    def encode_distribution(self, arr):

        # first eliminate the last columnd, round everything to two places, and form an integer
        M = np.round(100 * arr).astype(int)[:,:-1]

        # check which is the biggest integer that occurs least often in the array
        counter = {i: np.count_nonzero(M == i) for i in range(0, 101)}
        min_freq = min(counter.values())
        probs_with_min_freq = [k for k, v in counter.items() if v == min_freq]
        highest_prob_with_min_freq = max(probs_with_min_freq)

        # reduce all values higher or equal to highest probability by 1
        M[M > highest_prob_with_min_freq] -= 1

        # ravel the array and create chunks
        R = list(np.ravel(M))
        encoding_str = self.encode_integer_list(R, 3)
        
        # create dictionary
        return {"p": encoding_str, "r": arr.shape[0], "c": arr.shape[1] - 1, "t": highest_prob_with_min_freq}
    
    
    def decode_distribution(self, descriptor):

        # decode this thing
        encoding_str = descriptor["p"]
        rows = descriptor["r"]
        cols = descriptor["c"]
        threshold = descriptor["t"]

        # extract numbers
        nums = self.decode_integer_list(encoding_str, 3, rows * cols)

        # reshape
        probs = np.array(nums).reshape((rows, cols))
        probs[probs >= threshold] += 1
        probs = np.column_stack([probs, 100 - np.sum(probs, axis=1)]) / 100

        # return recovered probabilities
        return probs