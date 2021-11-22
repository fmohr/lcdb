import sys
import json
from lcdb import *

if __name__ == '__main__':
    
    print("Starting python script")
    
    # read params
    openmlid = int(sys.argv[1])
    algorithm = sys.argv[2]
    outer_seed = int(sys.argv[3])
    inner_seed_index = int(sys.argv[4])
    file = sys.argv[5]
    
    # get algorithm
    learner_params = {}
    if algorithm == "SVC_linear":
        learner_name = "sklearn.svm.LinearSVC"
    elif algorithm == "SVC_poly":
        learner_name = "sklearn.svm.SVC"
        learner_params = {"kernel": "poly"}
    elif algorithm == "SVC_rbf":
        learner_name = "sklearn.svm.SVC"
        learner_params = {"kernel": "rbf"}
    elif algorithm == "SVC_sigmoid":
        learner_name = "sklearn.svm.SVC"
        learner_params = {"kernel": "sigmoid"}
    else:
        learner_name = algorithm
    
    # compute curve
    num_seeds = 5
    outer_seeds = [outer_seed]
    inner_seeds = list(range(num_seeds * inner_seed_index, num_seeds * (inner_seed_index + 1)))
    encoder = DirectEncoder(2)
    print("Outer Seeds:", outer_seeds)
    print("Inner Seeds:", inner_seeds)
    out = compute_full_curve(learner_name, learner_params, openmlid, outer_seeds=outer_seeds, inner_seeds=inner_seeds, error="message", encoder=encoder, verbose=False, show_progress=True)
    with open(file, 'w') as outfile:
        json.dump(out, outfile)
