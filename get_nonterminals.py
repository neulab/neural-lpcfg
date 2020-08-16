import sys
from cfg2dep import parse_non_binary_line
import tqdm
from collections import Counter

corpus = sys.argv[1]

nt_counter, pt_counter = Counter(), Counter()

def dfs(tree):
    if('subtrees' in tree):
        ret_nts, ret_pts = Counter(), Counter()
        for subtree in tree['subtrees']:
            nts, pts = dfs(subtree)
            ret_nts.update(nts)
            ret_pts.update(pts)
        ret_nts[tree['label']] += 1
        return ret_nts, ret_pts
    else:
        return Counter(), Counter({tree['label']:1})

with open(corpus, "r") as f:
    for line in tqdm.tqdm(f):
        tree = parse_non_binary_line(line.strip())
        nts, pts = dfs(tree)
        nt_counter.update(nts)
        pt_counter.update(pts)

coarse_nt_counter = Counter()

f = lambda x : x[:x.find('-')] if x.find('-') != -1 else x
g = lambda y : y[:y.find('=')] if y.find('=') != -1 else y

for i in nt_counter:
    coarse_nt_counter[f(g(i))] += nt_counter[i]

print(nt_counter)
print(len(coarse_nt_counter), coarse_nt_counter)
print(len(pt_counter), pt_counter)
