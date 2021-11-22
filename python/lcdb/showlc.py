import sys
import json
from lcdb import *
import sklearn.metrics

if __name__ == '__main__':
    
    print("Starting python script")
    
    # read params
    filename = sys.argv[1]
    metric = sys.argv[2]
    
    # load curve data
    with open(filename) as infile:
        curve_descriptor = json.load(infile)
    
    # now process curve
    encoder = DirectEncoder()
    curve = get_curve_by_metric(curve_descriptor, metric, encoder=encoder, error="message")
    
    # plot curve
    print(list(zip(curve[0], [np.round(np.mean(v), 3) for v in curve[2]])))
    plot_train_and_test_curve(curve)