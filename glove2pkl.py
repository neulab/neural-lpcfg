import pickle
import sys
import numpy as np
import tqdm

w2v = dict()

with open(sys.argv[1], "r") as f:
    for line in tqdm.tqdm(f):
        line = line.split()
        word, vector = ' '.join(line[0:-300]), line[-300:]
        vector = np.array(list(map(float, vector)))
        w2v[word] = vector
    
pickle.dump(w2v, open(sys.argv[2], "wb"))
        