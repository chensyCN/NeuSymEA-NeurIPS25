import numpy as np
import scipy.sparse as sp
import scipy
import pdb
import tensorflow as tf
import os
import multiprocessing

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T


def get_matrix(triples,entity,rel):
        ent_size = max(entity)+1
        rel_size = (max(rel) + 1)
        print(ent_size,rel_size)
        adj_matrix = sp.lil_matrix((ent_size,ent_size))
        adj_features = sp.lil_matrix((ent_size,ent_size))
        radj = []
        rel_in = np.zeros((ent_size,rel_size))
        rel_out = np.zeros((ent_size,rel_size))
        
        for i in range(max(entity)+1):
            adj_features[i,i] = 1

        for h,r,t in triples:        
            adj_matrix[h,t] = 1; adj_matrix[t,h] = 1;
            adj_features[h,t] = 1; adj_features[t,h] = 1;
            radj.append([h,t,r]); radj.append([t,h,r+rel_size]); 
            rel_out[h][r] += 1; rel_in[t][r] += 1
            
        count = -1
        s = set()
        d = {}
        r_index,r_val = [],[]
        for h,t,r in sorted(radj,key=lambda x: x[0]*10e10+x[1]*10e5):
            if ' '.join([str(h),str(t)]) in s:
                r_index.append([count,r])
                r_val.append(1)
                d[count] += 1
            else:
                count += 1
                d[count] = 1
                s.add(' '.join([str(h),str(t)]))
                r_index.append([count,r])
                r_val.append(1)
        for i in range(len(r_index)):
            r_val[i] /= d[r_index[i][0]]
        
        rel_features = np.concatenate([rel_in,rel_out],axis=1)
        adj_features = normalize_adj(adj_features)
        rel_features = normalize_adj(sp.lil_matrix(rel_features))    
        return adj_matrix,r_index,r_val,adj_features,rel_features      


def load_data_for_dualamn(entity1, rel1, triples1, entity2, rel2, triples2, train_pair, dev_pair):
    
    adj_matrix,r_index,r_val,adj_features,rel_features = get_matrix(triples1+triples2,entity1.union(entity2),rel1.union(rel2))

    return np.array(train_pair),np.array(dev_pair),adj_matrix,np.array(r_index),np.array(r_val),adj_features,rel_features


def load_data_for_lightea(entity1, rel1, triples1, entity2, rel2, triples2, train_pair, dev_pair):

    triples = []
    for h, r, t in triples1:
        triples.append([h, t, 2 * r])
        triples.append([t, h, 2 * r + 1])
    for h, r, t in triples2:
        triples.append([h, t, 2 * r])
        triples.append([t, h, 2 * r + 1])
    triples = np.unique(triples, axis=0)
    
    node_size, rel_size = np.max(triples) + 1, np.max(triples[:, 2]) + 1

    ent_tuple,triples_idx = [],[]
    ent_ent_s,rel_ent_s,ent_rel_s = {},set(),set()
    last,index = (-1,-1), -1

    for i in range(node_size):
        ent_ent_s[(i,i)] = 0

    for h,t,r in triples:
        ent_ent_s[(h,h)] += 1
        ent_ent_s[(t,t)] += 1

        if (h,t) != last:
            last = (h,t)
            index += 1
            ent_tuple.append([h,t])
            ent_ent_s[(h,t)] = 0

        triples_idx.append([index,r])
        ent_ent_s[(h,t)] += 1
        rel_ent_s.add((r,h))
        ent_rel_s.add((t,r))

    ent_tuple = np.array(ent_tuple)
    triples_idx = np.unique(np.array(triples_idx),axis=0)

    ent_ent = np.unique(np.array(list(ent_ent_s.keys())),axis=0)
    ent_ent_val = np.array([ent_ent_s[(x,y)] for x,y in ent_ent]).astype("float32")
    rel_ent = np.unique(np.array(list(rel_ent_s)),axis=0)
    ent_rel = np.unique(np.array(list(ent_rel_s)),axis=0)

    train_pair = np.array(train_pair).astype(np.int64)
    dev_pair = np.array(dev_pair).astype(np.int64)


    return node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel, train_pair, dev_pair



