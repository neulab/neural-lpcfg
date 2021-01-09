import spacy
from collections import defaultdict
import sys
import tqdm
import pickle

left_head_count = defaultdict(int)
head_count = defaultdict(int)

def find_matching_parenthese(line, start_pos):
    count = 0
    for i in range(start_pos, len(line)):
        if line[i] == "(":
            count += 1
        elif line[i] == ")":
            count -= 1
        if count == 0:
            return i+1

def parse_line(line, offset=0):
    assert line[0] == "(" and line[-1] == ")"
    label = line.split(' ')[0][1:]
    content = ' '.join(line.split(' ')[1:])[:-1]
    is_leaf = label.split('-')[0] == "T"
    if is_leaf:
        return {'token': content, 'label':label}, 1
    else:
        break_pos = find_matching_parenthese(content, 0)
        left_subtree, left_count = parse_line(content[:break_pos], offset=offset)
        right_subtree, right_count = parse_line(content[break_pos+1:], offset=offset+left_count)
        return {'left':left_subtree, 'right':right_subtree, 'break_token_pos':offset+left_count, 'label':label}, left_count + right_count

def parse_non_binary_line(line):
    assert line[0] == "(" and line[-1] == ")"
    label = line.split(' ')[0][1:]
    content = ' '.join(line.split(' ')[1:])[:-1]
    is_leaf = content[0] != "("
    if is_leaf:
        return {'token': content, 'label':label}
    else:
        subtrees = []
        while(len(content)):
            break_pos = find_matching_parenthese(content, 0)
            subtree = parse_non_binary_line(content[:break_pos])
            content = content[break_pos+1:]
            subtrees.append(subtree)
        return {'subtrees':subtrees, 'label':label}

def get_sentence(tree):
    if 'left' in tree:
        return get_sentence(tree['left']) + get_sentence(tree['right'])
    else: 
        return [tree['token']]

def is_cut(idx, head_idx, pos):
    if (idx < pos and head_idx >= pos) or (head_idx < pos and idx >= pos):
        return True
    else:
        return False

def count_pointer(tree, dependency_head, root_pos):
    if 'break_token_pos' in tree:
        pos = tree['break_token_pos']
        nb_count = sum([1 if is_cut(i, j, pos) else 0 \
                          for i, j in dependency_head])
        if nb_count != 1:
            # print(tree, dependency_head, root_pos)
            return
        else:
            new_root = -1
            for idx, (i, j) in enumerate(dependency_head):
                if j == root_pos and is_cut(i, j, pos):
                    dependency_head[idx] = (i, i)
                    new_root = i
            # print(root_pos, pos)
            if root_pos < pos:
                left_head_count[tree['label']] += 1
                head_count[tree['label']] += 1
                left_head_count[(tree['label'], tree['left']['label'], tree['right']['label'])] += 1
                head_count[(tree['label'], tree['left']['label'], tree['right']['label'])] += 1
                count_pointer(tree['left'], list(filter(lambda x: x[0] < pos, dependency_head)), root_pos)
                count_pointer(tree['right'], list(filter(lambda x: x[0] >= pos, dependency_head)), new_root)
            else:
                head_count[tree['label']] += 1
                head_count[(tree['label'], tree['left']['label'], tree['right']['label'])] += 1
                count_pointer(tree['left'], list(filter(lambda x: x[0] < pos, dependency_head)), new_root)
                count_pointer(tree['right'], list(filter(lambda x: x[0] >= pos, dependency_head)), root_pos)
            
def tree2dependency(tree, dependency_head, offset=0):
    if('break_token_pos' in tree):
        left_root, left_count = tree2dependency(tree['left'], dependency_head, offset)
        right_root, right_count = tree2dependency(tree['right'], dependency_head, offset + left_count)
        lhc = left_head_count[(tree['label'], tree['left']['label'], tree['right']['label'])]
        hc = head_count[(tree['label'], tree['left']['label'], tree['right']['label'])]
        if hc == 0: # never seen this tuple, using back up plan
            lhc = left_head_count[tree['label']]
            hc = head_count[tree['label']]
        if lhc > hc - lhc: # left head
            dependency_head[right_root] = left_root
            return left_root, left_count + right_count
        else: # right head
            dependency_head[left_root] = right_root
            return right_root, left_count + right_count
    else:
        return offset, 1

def get_statistics():
    nlp = spacy.load("en_core_web_sm")
    for line in tqdm.tqdm(sys.stdin):
        tree, _ = parse_line(line.strip())
        sentence = get_sentence(tree)
        # if sentence[-1] == '.':
        #     sentence = sentence[:-1]
        doc = nlp(' '.join(sentence))
        dependency_head = [(idx, i.head.i) for idx, i in enumerate(doc)]
        root_pos = -1
        for i, j in dependency_head:
            if i == j:
                root_pos = i
        count_pointer(tree, dependency_head, root_pos)

def get_dependency():
    # left_head_count = defaultdict(int, {'NT-26': 16484, 'NT-10': 5763, 'NT-4': 6744, 'NT-25': 1012, 'NT-3': 3023, 'NT-9': 1237, 'NT-16': 1447, 'NT-8': 3903, 'NT-21': 15007, 'NT-24': 5629, 'NT-6': 14981, 'NT-27': 3693, 'NT-1': 3960, 'NT-18': 476, 'NT-23': 5842, 'NT-2': 3270, 'NT-12': 13585, 'NT-22': 3832, 'NT-19': 1941, 'NT-28': 1784, 'NT-13': 309, 'NT-5': 748, 'NT-20': 2106, 'NT-7': 4187, 'NT-14': 3205, 'NT-17': 1178, 'NT-30': 367, 'NT-29': 1674, 'NT-15': 17})
    # head_count = defaultdict(int, {'NT-12': 158660, 'NT-26': 25361, 'NT-20': 17788, 'NT-10': 6406, 'NT-4': 7328, 'NT-25': 28733, 'NT-3': 3055, 'NT-29': 17375, 'NT-9': 1279, 'NT-16': 1469, 'NT-8': 4007, 'NT-21': 17025, 'NT-24': 6196, 'NT-6': 15190, 'NT-27': 3903, 'NT-1': 12873, 'NT-18': 1951, 'NT-23': 5934, 'NT-2': 3294, 'NT-30': 1435, 'NT-22': 4044, 'NT-19': 1959, 'NT-28': 1801, 'NT-17': 7809, 'NT-13': 581, 'NT-7': 4725, 'NT-5': 1100, 'NT-14': 3219, 'NT-15': 17})
    nlp = spacy.load("en_core_web_sm")
    for line in tqdm.tqdm(sys.stdin):
        tree, size = parse_line(line.strip())
        dependency_head = [-1 for i in range(size)]
        root, _ = tree2dependency(tree, dependency_head)
        dependency_head[root] = root
        sentence = get_sentence(tree)
        doc = nlp(' '.join(sentence))
        if(len(doc) != len(dependency_head)):
            continue
        for i in doc:
            print("%d\t%s\t%s\t%d"%(i.i, i.text, i.tag_, dependency_head[i.i]))
        print()

if __name__ == "__main__":
    # get_statistics()
    left_head_count = pickle.load(open("left_head_count.pkl", "rb"))
    head_count = pickle.load(open("head_count.pkl", "rb"))
    get_dependency()
        
    