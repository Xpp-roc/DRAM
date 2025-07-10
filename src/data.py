import copy
from asyncio import Queue
from collections import Counter
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import sample
from queue import Queue
import random
from torch.nn.utils import clip_grad_norm_
import time

def construct_fact_dict(fact_rdf):
    fact_dict = {}
    for rdf in fact_rdf:
        fact = parse_rdf(rdf)
        h, r, t = fact
        if r not in fact_dict:
            fact_dict[r] = []
        fact_dict[r].append(rdf)

    return fact_dict  

def parse_rdf(rdf):
    """
        return: head, relation, tail
    """
    rdf_tail, rdf_rel, rdf_head= rdf
    return rdf_head, rdf_rel, rdf_tail


class Dictionary(object):
    def __init__(self):
        self.rel2idx_ = {}
        self.idx2rel_ = {}
        self.idx = 0
        
    def add_relation(self, rel):
        if rel not in self.rel2idx_.keys():
            self.rel2idx_[rel] = self.idx
            self.idx2rel_[self.idx] = rel
            self.idx += 1
        
    @property
    def rel2idx(self):
        return self.rel2idx_
    
    @property
    def idx2rel(self):
        return self.idx2rel_
    
    def __len__(self):
        return len(self.idx2rel_)


def load_entities(path):
    idx2ent, ent2idx = {}, {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            e = line.strip()
            ent2idx[e] = idx
            idx2ent[idx] = e
    return idx2ent, ent2idx      


class Dataset(object):
    def __init__(self, data_root, sparsity=1, inv=False):
        # Construct entity_list
        entity_path = data_root + 'entities.txt'
        self.idx2ent_, self.ent2idx_ = load_entities(entity_path)
        self.entities = set(self.idx2ent_.values())  # 添加实体集合
        # Construct rdict which contains relation2idx & idx2relation2
        relation_path = data_root + 'relations.txt'
        self.rdict = Dictionary()
        self.load_relation_dict(relation_path)
        # head relation
        self.head_rdict = Dictionary()
        self.head_rdict = copy.deepcopy(self.rdict)
        # load (h, r, t) tuples
        fact_path     = data_root + 'facts.txt'
        train_path    = data_root + 'train.txt'
        valid_path    = data_root + 'valid.txt'
        test_path     = data_root + 'test.txt'
        if inv :
            fact_path += '.inv'
        self.rdf_data_ = self.load_data_(fact_path, train_path, valid_path, test_path, sparsity)
        self.fact_rdf_, self.train_rdf_, self.valid_rdf_, self.test_rdf_ = self.rdf_data_
        # inverse
        if inv :
            # add inverse relation to rdict
            rel_list = list(self.rdict.rel2idx_.keys())
            for rel in rel_list:
                inv_rel = "inv_" + rel
                self.rdict.add_relation(inv_rel)                
                self.head_rdict.add_relation(inv_rel)                
        # add None 
        self.head_rdict.add_relation("None")

    def load_rdfs(self, path):
        rdf_list = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                tuples = line.strip().split('\t')
                rdf_list.append(tuples)
        return rdf_list
    
    def load_data_(self, fact_path, train_path, valid_path, test_path, sparsity):
        fact  = self.load_rdfs(fact_path)
        fact = sample(fact ,int(len(fact)*sparsity))
        train = self.load_rdfs(train_path)
        valid = self.load_rdfs(valid_path)
        test  = self.load_rdfs(test_path)
        return fact, train, valid, test
    
    def load_relation_dict(self, relation_path):
        """
        Read relation.txt to relation dictionary
        """
        with open(relation_path, encoding='utf-8') as f:
            rel_list = f.readlines()
            for r in rel_list:
                relation = r.strip()
                self.rdict.add_relation(relation)
                #self.head_dict.add_relation(relation)

    def getPathsFrom(self, entity):
        """
        获取从某个实体出发的所有路径（即所有 (头实体, 关系, 尾实体)）
        """
        paths = []
        if entity not in self.ent2idx_:
            return paths  # 该实体不在知识图谱中，返回空列表

        # 遍历所有事实数据，找出以该实体为起点的路径
        for h, r, t in self.fact_rdf_:
            if h == entity:
                paths.append((t, r))  # 记录路径：当前实体通过 r 连接到 t

        return paths
    def get_relation_dict(self):
        return self.rdict
    
    def get_head_relation_dict(self):
        return self.head_rdict

    @property
    def idx2ent(self):
        return self.idx2ent_

    @property
    def ent2idx(self):
        return self.ent2idx_

    @property
    def fact_rdf(self):
        return self.fact_rdf_
    
    @property
    def train_rdf(self):
        return self.train_rdf_
    
    @property
    def valid_rdf(self):
        return self.valid_rdf_
    
    @property
    def test_rdf(self):
        return self.test_rdf_

    def all_triples(self):
        """yield (h_idx, r_idx, t_idx) over fact+train+valid+test"""
        for h, r, t in self.fact_rdf_ + self.train_rdf_ + self.valid_rdf_ + self.test_rdf_:
            yield self.ent2idx_[h], self.rdict.rel2idx[r], self.ent2idx_[t]

    def test_triples(self):
        """yield (h_idx, r_idx, t_idx) over test set"""
        for h, r, t in self.test_rdf_:
            yield self.ent2idx_[h], self.rdict.rel2idx[r], self.ent2idx_[t]

def sample_anchor_rdf(rdf_data, num=1):
    if num < len(rdf_data):
        return sample(rdf_data, num)
    else:
        return rdf_data

def construct_descendant(rdf_data):
    # take entity as h, map it to its r, t
    entity2desced = {}
    for rdf_ in rdf_data:
        h_, r_, t_ = parse_rdf(rdf_)
        if h_ not in entity2desced.keys():
            entity2desced[h_] = [(r_, t_)]
        else:
            entity2desced[h_].append((r_, t_))
    return entity2desced


def connected(entity2desced, head, tail):
    if head in entity2desced:
        decedents = entity2desced[head]
        for d in decedents:
            d_relation_, d_tail_ = d
            if d_tail_ == tail:
                return d_relation_
        return False
    else:
        return False

    
def construct_rule_seq(rdf_data, anchor_rdf, entity2desced, max_path_len=2, PRINT=False):    
    len2seq = {}
    anchor_h, anchor_r, anchor_t = parse_rdf(anchor_rdf)
    # Search
    stack = [(anchor_h, anchor_r, anchor_t)]
    stack_print = ['{}-{}-{}'.format(anchor_h, anchor_r, anchor_t)]
    pre_path = anchor_h
    rule_seq, expended_node = [], []
    record = []
    while len(stack) > 0:
        cur_h, cur_r, cur_t = stack.pop(-1)
        cur_print = stack_print.pop(-1)
        deced_list = []
        
        if cur_t in entity2desced:
            deced_list = entity2desced[cur_t]  

        if len(cur_r.split('|')) < max_path_len and len(deced_list) > 0 and cur_t not in expended_node:
            for r_, t_ in deced_list:
                if t_ != cur_h and t_ != anchor_h:
                    stack.append((cur_t, cur_r+'|'+r_, t_))
                    stack_print.append(cur_print+'-{}-{}'.format(r_, t_))
        expended_node.append(cur_t)
        
        rule_head_rel = connected(entity2desced, anchor_h, cur_t)
        if rule_head_rel and cur_t != anchor_t:
            rule = cur_r + '-' + rule_head_rel  
            rule_seq.append(rule)
            if (cur_h, r_, t_) not in record:
                record.append((cur_h, r_, t_))
            if PRINT:
                print('rule body:\n{}'.format(cur_print))
                print('rule head:\n{}-{}-{}'.format(anchor_h, rule_head_rel, cur_t))
                print('rule:\n{}\n'.format(rule))
        elif rule_head_rel == False and random.random() > 0.9:
            rule = cur_r + '-' + "None"
            rule_seq.append(rule)
            if (cur_h, r_, t_) not in record:
                record.append((cur_h, r_, t_))
            if PRINT:
                print('rule body:\n{}'.format(cur_print))
                print('rule head:\n{}-{}-{}'.format(anchor_h, rule_head_rel, cur_t))
                print('rule:\n{}\n'.format(rule))
    return rule_seq, record


def body2idx(body_list, head_rdict):
    """
    Input a rule (string) and idx it
    """
    res = []
    for body in body_list:
        body_path = body.split('|')
        # indexs include body idx seq + notation + head idx
        indexs = []
        for rel in body_path:
            indexs.append(head_rdict.rel2idx[rel])
        res.append(indexs)
    return res

def inv_rel_idx(head_rdict):
    inv_rel_idx = []
    for i_ in range(len(head_rdict.idx2rel)):
        r_ = head_rdict.idx2rel[i_]
        if "inv_" in r_:
            inv_rel_idx.append(i_)
    return inv_rel_idx 
    
def idx2body(index, head_rdict):
    body = "|".join([head_rdict.idx2rel[idx] for idx in index])
    return body

def rule2idx(rule, head_rdict):
    """
    Input a rule (string) and idx it
    """
    body, head = rule.split('-')
    body_path = body.split('|')
    # indexs include body idx seq + notation + head idx
    indexs = []
    for rel in body_path+[-1, head]:
        indexs.append(head_rdict.rel2idx[rel] if rel != -1 else -1)
    return indexs


def idx2rule(index, head_rdict):
    body_idx = index[0:-2]
    body = "|".join([ head_rdict.idx2rel[b] for b in body_idx])
    rule = body + "-" + head_rdict.idx2rel[index[-1]] 
    return rule


def bidirectional_BFS(kb, entity1, entity2, max_depth=3):
    if entity1 not in kb.entities or entity2 not in kb.entities:
        return (False, None, None)

    forward_queue = Queue()
    forward_visited = {entity1: (None, None)}
    forward_queue.put(entity1)

    backward_queue = Queue()
    backward_visited = {entity2: (None, None)}
    backward_queue.put(entity2)

    meeting_entity = None

    while not forward_queue.empty() and not backward_queue.empty():
        meeting_entity = expand_BFS(kb, forward_queue, forward_visited, backward_visited)
        if meeting_entity:
            break
        meeting_entity = expand_BFS(kb, backward_queue, backward_visited, forward_visited)
        if meeting_entity:
            break

    if not meeting_entity:
        return (False, None, None)

    entity_list, path_list = reconstruct_bidirectional_path(forward_visited, backward_visited, entity1, entity2,
                                                            meeting_entity)
    return (True, entity_list, path_list)


def expand_BFS(kb, queue, visited, other_visited):
    if queue.empty():
        return None

    current_entity = queue.get()
    for next_entity, relation in kb.getPathsFrom(current_entity):

        if next_entity not in visited:
            visited[next_entity] = (current_entity, relation)
            queue.put(next_entity)

        if next_entity in other_visited:
            return next_entity

    return None


def reconstruct_bidirectional_path(forward_visited, backward_visited, entity1, entity2, meeting_entity):
    entity_list = []
    path_list = []
    cur = meeting_entity
    while cur:
        entity_list.append(cur)
        if forward_visited[cur][1]:
            path_list.append(forward_visited[cur][1])
        cur = forward_visited[cur][0]
    entity_list.reverse()
    path_list.reverse()

    cur = backward_visited[meeting_entity][0]
    while cur:
        entity_list.append(cur)
        path_list.append(backward_visited[cur][1])
        cur = backward_visited[cur][0]

    return entity_list, path_list


def optimized_sample_training_data(max_path_len, anchor_num,
                                   fact_rdf, entity2desced,
                                   head_rdict, kb):
    print("优化采样训练数据...")
    anchors_rdf = []
    relation_counts = Counter([rdf[1] for rdf in fact_rdf])
    total_relations = sum(relation_counts.values())
    relation_weights = {rel: c / total_relations
                        for rel, c in relation_counts.items()}

    fact_dict = construct_fact_dict(fact_rdf)
    for head in head_rdict.rel2idx:
        if head != "None" and "inv_" not in head:
            avail = fact_dict.get(head, [])
            if not avail:
                continue
            avail = np.asarray(avail, dtype=object)
            probs = np.asarray([relation_weights[r[1]] for r in avail])
            probs /= probs.sum()
            k = min(anchor_num, len(avail))
            anchors_rdf.extend(avail[
                np.random.choice(len(avail), k, replace=False, p=probs)
            ].tolist())

    total_anchors = len(anchors_rdf)
    print(f"选取的总锚点数量: {total_anchors}")
    anchors_with_path = 0
    closed_paths = 0
    train_rule, train_rule_dict = [], {}
    len2train_rule_idx = {}

    for anchor_rdf in anchors_rdf:
        h, _, t = anchor_rdf
        ok, ent_list, path_list = bidirectional_BFS(kb, h, t)

        if not ok:
            continue
        anchors_with_path += 1

        rule_seq, record = construct_rule_seq(
            fact_rdf, anchor_rdf, entity2desced,
            max_path_len, PRINT=False)

        closed_paths += len(record)

        for rule in rule_seq:
            idx = torch.LongTensor(rule2idx(rule, head_rdict))
            head_rel = head_rdict.idx2rel[idx[-1].item()]

            train_rule.append(rule)
            train_rule_dict.setdefault(head_rel, []).append(idx)

            L = len(idx) - 2          # body 长度
            len2train_rule_idx.setdefault(L, []).append(idx)
    anchors_without_path = total_anchors - anchors_with_path
    unique_rules = len(set(train_rule))

    print(f"  成功可达的锚点数  : {anchors_with_path}")
    print(f"  不可达锚点数    : {anchors_without_path}")
    print(f"  闭合路径实例数  : {closed_paths}")
    print(f"  唯一规则模式数  : {unique_rules}")
    print(f"  头关系种类数    : {len(train_rule_dict)}")

    for L, lst in len2train_rule_idx.items():
        print(f"  body_len = {L:<2d}  样本量 = {len(lst)}")

    return len2train_rule_idx


def enumerate_body(relation_num, rdict, body_len):
    import itertools
    all_body_idx = list(list(x) for x in itertools.product(range(relation_num), repeat=body_len))
    # transfer index to relation name
    idx2rel = rdict.idx2rel
    all_body = []
    for b_idx_ in all_body_idx:
        b_ = [idx2rel[x] for x in b_idx_]
        all_body.append(b_)
    return all_body_idx, all_body

