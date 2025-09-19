import pdb
import time
import numpy as np
import os
import sys
pj_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pj_path)
from config import Config
import tensorflow as tf
from tqdm import tqdm
import faiss
import pickle
import json
from tqdm import tqdm
import tensorflow.keras.backend as K

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")

 
def load_name_features(dataset,vector_path,mode = "word-level"):
    
    try:
        word_vecs = pickle.load(open("./word_vectors.pkl","rb"))
    except:
        word_vecs = {}
        with open(vector_path,encoding='UTF-8') as f:
            for line in tqdm(f.readlines()):
                line = line.split()
                word_vecs[line[0]] = [float(x) for x in line[1:]]
        pickle.dump(word_vecs,open("./word_vectors.pkl","wb"))

    if "EN" in dataset:
        ent_names = json.load(open("translated_ent_name/%s.json"%dataset[:-1].lower(),"r"))

    d = {}
    count = 0
    for _,name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word)-1):
                if word[idx:idx+2] not in d:
                    d[word[idx:idx+2]] = count
                    count += 1

    ent_vec = np.zeros((len(ent_names),300),"float32")
    char_vec = np.zeros((len(ent_names),len(d)),"float32")
    for i,name in tqdm(ent_names):
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
            for idx in range(len(word)-1):
                char_vec[i,d[word[idx:idx+2]]] += 1
        if k:
            ent_vec[i]/=k
        else:
            ent_vec[i] = np.random.random(300)-0.5

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = np.random.random(len(d))-0.5
    
    faiss.normalize_L2(ent_vec)
    faiss.normalize_L2(char_vec)

    if mode == "word-level":
        name_feature = ent_vec
    if mode == "char-level":
        name_feature = char_vec
    if mode == "hybrid-level": 
        name_feature = np.concatenate([ent_vec,char_vec],-1)
        
    return name_feature

def normalize_L2_np(features, epsilon=1e-12):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / (norms + epsilon)
    return normalized_features

def random_projection(x,out_dim):
    random_vec = K.l2_normalize(tf.random.normal((x.shape[-1],out_dim)),axis=-1)
    return K.dot(x,random_vec)

def batch_sparse_matmul(sparse_tensor,dense_tensor,batch_size = 128,save_mem = False):
    results = []
    for i in range(dense_tensor.shape[-1]//batch_size + 1):
        temp_result = tf.sparse.sparse_dense_matmul(sparse_tensor,dense_tensor[:, i*batch_size:(i+1)*batch_size])
        if save_mem:
            temp_result = temp_result.numpy()
        results.append(temp_result)
    return tf.concat(results, axis=-1)

def sparse_sinkhorn_sims(left,right,features,top_k=500,iteration=15,mode = "test"):

    features_l = features[left]
    features_r = features[right]
    t1 = time.time()
    faiss.normalize_L2(features_l); faiss.normalize_L2(features_r)
    t2 = time.time()
    print("Time to normalize features: %.2f seconds"%(t2-t1))

    res = faiss.StandardGpuResources()
    
    dim, measure = features_l.shape[1], faiss.METRIC_INNER_PRODUCT
    if mode == "test":
        param = 'Flat'
        index = faiss.index_factory(dim, param, measure)
    else:
        param = 'IVF256(RCQ2x5),PQ32'
        index = faiss.index_factory(dim, param, measure)
        index.nprobe = 16
    if len(gpus):
        index = faiss.index_cpu_to_gpu(res, Config.gpu+1, index)
    index.train(features_r)
    index.add(features_r)
    sims, index = index.search(features_l, top_k)
    
    row_sims = K.exp(sims.flatten()/0.02)
    index = K.flatten(index.astype("int32"))

    size = len(left)
    row_index = K.transpose(([K.arange(size*top_k)//top_k,index,K.arange(size*top_k)]))
    col_index = tf.gather(row_index,tf.argsort(row_index[:,1]))
    covert_idx = tf.argsort(col_index[:,2])

    for _ in range(iteration):
        row_sims = row_sims / tf.gather(indices=row_index[:,0],params = tf.math.segment_sum(row_sims,row_index[:,0]))
        col_sims = tf.gather(row_sims,col_index[:,2])
        col_sims = col_sims / tf.gather(indices=col_index[:,1],params = tf.math.segment_sum(col_sims,col_index[:,1]))
        row_sims = tf.gather(col_sims,covert_idx)
        
    return K.reshape(row_index[:,1],(-1,top_k)), K.reshape(row_sims,(-1,top_k))

def test(test_pair,features,top_k=500,iteration=15):
    left, right = test_pair[:,0], np.unique(test_pair[:,1])
    index,sims = sparse_sinkhorn_sims(left, right,features,top_k,iteration,"test")
    ranks = tf.argsort(-sims,-1).numpy()
    index = index.numpy()
    
    wrong_list,right_list = [],[]
    h1,h10,mrr = 0, 0, 0
    pos = np.zeros(np.max(right)+1)
    pos[right] = np.arange(len(right))
    for i in range(len(test_pair)):
        rank = np.where(pos[test_pair[i,1]] == index[i,ranks[i]])[0]
        if len(rank) != 0:
            if rank[0] == 0:
                h1 += 1
                right_list.append(test_pair[i])
            else:
                wrong_list.append((test_pair[i],right[index[i,ranks[i]][0]]))
            if rank[0] < 10:
                h10 += 1
            mrr += 1/(rank[0]+1) 
    print("Hits@1: %.3f Hits@10: %.3f MRR: %.3f\n"%(h1/len(test_pair),h10/len(test_pair),mrr/len(test_pair)))
    
    return right_list, wrong_list