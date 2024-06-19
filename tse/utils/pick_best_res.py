import json
import math
from sys import argv

if __name__ == "__main__":
    dir = argv[1]
    metric = argv[2]
    hyperparameters = argv[3:]

    with open(dir, "r") as f:
        data = json.load(f)


    best_metric = -math.inf
    best_config = None

    for d in data:
        if d[metric] >= best_metric:
            best_metric = d[metric]
            best_config = d
    
    print("@@@@ Best configuration setup @@@@", flush=True)
    for h in hyperparameters:
        print(f"{h}: {best_config[h]}", flush=True)
    print(f"{metric}: {best_metric}", flush=True)