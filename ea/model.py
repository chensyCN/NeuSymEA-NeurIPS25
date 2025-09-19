from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')
import GPUtil

import os
import sys
pj_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pj_path)
import keras
import numpy as np
import tensorflow as tf
from keras.layers import *
import keras.backend as K
from utils import *
from evaluate import evaluate
from dualamn_layers import *
from lightea_functions import *
from tqdm import trange


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class DualAmn:
    def __init__(self, data, node_hidden=128, rel_hidden=128, batch_size=1024, dropout_rate=0.3, lr=0.005, gamma=1, depth=2):
        self.node_hidden = node_hidden
        self.rel_hidden = rel_hidden
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.gamma = gamma
        self.depth = depth

        self.load_and_preprocess_data(data)
        self.model, self.get_emb = self.build_model()

        seed = 12306
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        # TensorFlow session configuration
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)

    def load_and_preprocess_data(self, data):
        entity1, rel1, triples1, entity2, rel2, triples2, train_pair, dev_pair = data
        self.train_pair, self.dev_pair, self.adj_matrix, self.r_index, self.r_val, self.adj_features, self.rel_features = load_data_for_dualamn(entity1, rel1, triples1, entity2, rel2, triples2, train_pair, dev_pair)
        self.adj_matrix = np.stack(self.adj_matrix.nonzero(), axis=1)
        self.rel_matrix, self.rel_val = np.stack(self.rel_features.nonzero(), axis=1), self.rel_features.data
        self.ent_matrix, self.ent_val = np.stack(self.adj_features.nonzero(), axis=1), self.adj_features.data
        self.node_size, self.rel_size = self.adj_features.shape[0], self.rel_features.shape[1]
        self.triple_size = len(self.adj_matrix)
        self.rest_set_1 = [e1 for e1, e2 in self.dev_pair]
        self.rest_set_2 = [e2 for e1, e2 in self.dev_pair]
        np.random.shuffle(self.rest_set_1)
        np.random.shuffle(self.rest_set_2)
        self.evaluater = evaluate(self.dev_pair)

    def update_data(self, train_pair):
        
        for e1, e2 in train_pair:
            if e1 in self.rest_set_1:
                self.rest_set_1.remove(e1)
            if e2 in self.rest_set_2:
                self.rest_set_2.remove(e2)

        original_train_pair = list(map(tuple, self.train_pair))
        self.train_pair = list(set(original_train_pair + train_pair))
        self.train_pair = np.array(self.train_pair)

    def reset_data(self, train_pair):
        self.train_pair = np.array(train_pair)
        self.rest_set_1 = [e1 for e1, e2 in self.dev_pair]
        self.rest_set_2 = [e2 for e1, e2 in self.dev_pair]
        for e1, e2 in train_pair:
            if e1 in self.rest_set_1:
                self.rest_set_1.remove(e1)
            if e2 in self.rest_set_2:
                self.rest_set_2.remove(e2)

    def build_model(self):
        # Insert the provided model building code here
        # return train_model, feature_model
        adj_input = Input(shape=(None,2))
        index_input = Input(shape=(None,2),dtype='int64')
        val_input = Input(shape = (None,))
        rel_adj = Input(shape=(None,2))
        ent_adj = Input(shape=(None,2))
        
        ent_emb = TokenEmbedding(self.node_size,self.node_hidden,trainable = True)(val_input) 
        rel_emb = TokenEmbedding(self.rel_size,self.node_hidden,trainable = True)(val_input)
        
        def avg(tensor,size):
            adj = K.cast(K.squeeze(tensor[0],axis = 0),dtype = "int64")   
            adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:,0],dtype = 'float32'), dense_shape=(self.node_size,size)) 
            adj = tf.sparse.softmax(adj) 
            return tf.sparse.sparse_dense_matmul(adj,tensor[1])
        
        opt = [rel_emb,adj_input,index_input,val_input]
        ent_feature = Lambda(avg,arguments={'size':self.node_size})([ent_adj,ent_emb])
        rel_feature = Lambda(avg,arguments={'size':self.rel_size})([rel_adj,rel_emb])
        
        e_encoder = NR_GraphAttention(self.node_size,activation="tanh",
                                        rel_size = self.rel_size,
                                        use_bias = True,
                                        depth = self.depth,
                                        triple_size = self.triple_size)
        
        r_encoder = NR_GraphAttention(self.node_size,activation="tanh",
                                        rel_size = self.rel_size,
                                        use_bias = True,
                                        depth = self.depth,
                                        triple_size = self.triple_size)
        
        out_feature = Concatenate(-1)([e_encoder([ent_feature]+opt),r_encoder([rel_feature]+opt)])
        out_feature = Dropout(self.dropout_rate)(out_feature)
        
        alignment_input = Input(shape=(None,2))
        
        def align_loss(tensor): 
            
            def squared_dist(x):
                A,B = x
                row_norms_A = tf.reduce_sum(input_tensor=tf.square(A), axis=1)
                row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.
                row_norms_B = tf.reduce_sum(input_tensor=tf.square(B), axis=1)
                row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.
                return row_norms_A + row_norms_B - 2 * tf.matmul(A, B,transpose_b=True) 
            
            emb = tensor[1]
            l,r = K.cast(tensor[0][0,:,0],'int32'),K.cast(tensor[0][0,:,1],'int32')
            l_emb,r_emb = K.gather(reference=emb,indices=l),K.gather(reference=emb,indices=r)
            
            pos_dis = K.sum(K.square(l_emb-r_emb),axis=-1,keepdims=True)
            r_neg_dis = squared_dist([r_emb,emb])
            l_neg_dis = squared_dist([l_emb,emb])
            
            l_loss = pos_dis - l_neg_dis + self.gamma
            l_loss = l_loss *(1 - K.one_hot(indices=l,num_classes=self.node_size) - K.one_hot(indices=r,num_classes=self.node_size))
            
            r_loss = pos_dis - r_neg_dis + self.gamma
            r_loss = r_loss *(1 - K.one_hot(indices=l,num_classes=self.node_size) - K.one_hot(indices=r,num_classes=self.node_size))
            
            r_loss = (r_loss - K.stop_gradient(K.mean(r_loss,axis=-1,keepdims=True))) / K.stop_gradient(K.std(r_loss,axis=-1,keepdims=True))
            l_loss = (l_loss - K.stop_gradient(K.mean(l_loss,axis=-1,keepdims=True))) / K.stop_gradient(K.std(l_loss,axis=-1,keepdims=True))
            
            lamb,tau = 30, 10
            l_loss = keras.backend.logsumexp(lamb*l_loss+tau,axis=-1)
            r_loss = keras.backend.logsumexp(lamb*r_loss+tau,axis=-1)
            return K.mean(l_loss + r_loss)

        loss = Lambda(align_loss)([alignment_input,out_feature])

        inputs = [adj_input,index_input,val_input,rel_adj,ent_adj]
        train_model = keras.Model(inputs = inputs + [alignment_input],outputs = loss)
        # train_model.compile(loss=lambda y_true,y_pred: y_pred,optimizer=keras.optimizers.rmsprop(self.lr))
        train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.lr))

        
        feature_model = keras.Model(inputs = inputs,outputs = out_feature)
        
        return train_model,feature_model

    def get_embedding(self, index_a, index_b, vec=None):
        # Insert the provided get_embedding function code here
        if vec is None:
            inputs = [self.adj_matrix,self.r_index,self.r_val,self.rel_matrix,self.ent_matrix]
            inputs = [np.expand_dims(item,axis=0) for item in inputs]
            vec = self.get_emb.predict_on_batch(inputs)
        Lvec = np.array([vec[e] for e in index_a])
        Rvec = np.array([vec[e] for e in index_b])
        Lvec = Lvec / (np.linalg.norm(Lvec,axis=-1,keepdims=True)+1e-5)
        Rvec = Rvec / (np.linalg.norm(Rvec,axis=-1,keepdims=True)+1e-5)
        return Lvec,Rvec

    def train(self, epoch=10):
        
        for i in trange(epoch):
            np.random.shuffle(self.train_pair)
            for pairs in [self.train_pair[i*self.batch_size:(i+1)*self.batch_size] for i in range(len(self.train_pair)//self.batch_size + 1)]:
                if len(pairs) == 0:
                    continue
                inputs = [self.adj_matrix, self.r_index, self.r_val, self.rel_matrix, self.ent_matrix, pairs]
                inputs = [np.expand_dims(item, axis=0) for item in inputs]
                self.model.train_on_batch(inputs, np.zeros((1, 1)))
            if i == epoch - 1:
                Lvec, Rvec = self.get_embedding(self.dev_pair[:, 0], self.dev_pair[:, 1])
                self.evaluater.test(Lvec, Rvec)
        new_pair = []
        print(f"number of rest_set_1: {len(self.rest_set_1)}, rest_set_2: {len(self.rest_set_2)}")
        Lvec,Rvec = self.get_embedding(self.rest_set_1, self.rest_set_2)
        A,B = self.evaluater.CSLS_cal(Lvec,Rvec,False)
        for i,j in enumerate(A):
            if  B[j] == i:
                new_pair.append((self.rest_set_1[j], self.rest_set_2[i]))

        dev_pair = list(map(tuple, self.dev_pair))
        correct_inferred = set(new_pair) & set(dev_pair)

        print(f"Correctly inferred {len(correct_inferred)} pairs.")
        print(f"Total inferred {len(new_pair)} pairs.")

        return new_pair

    def fine_tune(self, epoch=20, turn=5):
        """
        the original train function is used to train the baseline model, and the fine_tune function is used to fine-tune the model.
        """
        for _ in range(turn):
            for i in trange(epoch):
                np.random.shuffle(self.train_pair)
                for pairs in [self.train_pair[i*self.batch_size:(i+1)*self.batch_size] for i in range(len(self.train_pair)//self.batch_size + 1)]:
                    if len(pairs) == 0:
                        continue
                    inputs = [self.adj_matrix, self.r_index, self.r_val, self.rel_matrix, self.ent_matrix, pairs]
                    inputs = [np.expand_dims(item, axis=0) for item in inputs]
                    self.model.train_on_batch(inputs, np.zeros((1, 1)))
                if i == epoch - 1:
                    Lvec, Rvec = self.get_embedding(self.dev_pair[:, 0], self.dev_pair[:, 1])
                    self.evaluater.test(Lvec, Rvec)
                new_pair = []
            Lvec,Rvec = self.get_embedding(self.rest_set_1, self.rest_set_2)
            A,B = self.evaluater.CSLS_cal(Lvec,Rvec,False)
            for i,j in enumerate(A):
                if  B[j] == i:
                    new_pair.append((self.rest_set_1[j], self.rest_set_2[i]))
            
            self.update_data(new_pair)
            epoch = 5

 
class LightEA:
    
    def __init__(self, data, ent_dim=64, mini_dim=16, depth=2, top_k=500):
        self.ent_dim = ent_dim
        self.rel_dim = ent_dim//3
        self.mini_dim = mini_dim
        self.depth = depth
        self.top_k = top_k

        self.node_size, self.rel_size, self.ent_tuple, self.triples_idx, self.ent_ent, self.ent_ent_val, self.rel_ent, self.ent_rel, self.train_pair, self.test_pair = load_data_for_lightea(*data)
        self.candidates_x, self.candidates_y = set([x for x,y in self.test_pair]), set([y for x,y in self.test_pair])

    def get_features(self, train_pair,extra_feature = None):
    
        # pdb.set_trace()
        if extra_feature is not None:
            ent_feature = extra_feature
        else:
            random_vec = K.l2_normalize(tf.random.normal((len(train_pair),self.ent_dim)),axis=-1)
            ent_feature = tf.tensor_scatter_nd_update(tf.zeros((self.node_size,self.ent_dim)),train_pair.reshape((-1,1)),tf.repeat(random_vec,2,axis=0))
        rel_feature = tf.zeros((self.rel_size,ent_feature.shape[-1]))
        
        ent_ent_graph = tf.SparseTensor(indices=self.ent_ent,values=self.ent_ent_val,dense_shape=(self.node_size,self.node_size))
        rel_ent_graph = tf.SparseTensor(indices=self.rel_ent,values=K.ones(self.rel_ent.shape[0]),dense_shape=(self.rel_size,self.node_size))
        ent_rel_graph = tf.SparseTensor(indices=self.ent_rel,values=K.ones(self.ent_rel.shape[0]),dense_shape=(self.node_size,self.rel_size))
        
        ent_list,rel_list = [ent_feature],[rel_feature]
        for i in range(2):
            new_rel_feature = batch_sparse_matmul(rel_ent_graph,ent_feature)
            new_rel_feature = tf.nn.l2_normalize(new_rel_feature,axis=-1)
            
            new_ent_feature = batch_sparse_matmul(ent_ent_graph,ent_feature)
            new_ent_feature += batch_sparse_matmul(ent_rel_graph,rel_feature)
            new_ent_feature = tf.nn.l2_normalize(new_ent_feature,axis=-1)
            
            ent_feature = new_ent_feature; rel_feature = new_rel_feature
            ent_list.append(ent_feature); rel_list.append(rel_feature)
        
        ent_feature = K.l2_normalize(K.concatenate(ent_list,1),-1)
        rel_feature = K.l2_normalize(K.concatenate(rel_list,1),-1)
        rel_feature = random_projection(rel_feature,self.rel_dim)
        
        
        batch_size = ent_feature.shape[-1]//self.mini_dim
        sparse_graph = tf.SparseTensor(indices=self.triples_idx,values=K.ones(self.triples_idx.shape[0]),dense_shape=(np.max(self.triples_idx)+1,self.rel_size))
        adj_value = batch_sparse_matmul(sparse_graph,rel_feature)
        
        features_list = []
        for batch in range(self.rel_dim//batch_size + 1):
            temp_list = []
            for head in range(batch_size):
                if batch*batch_size+head>=self.rel_dim:
                    break
                sparse_graph = tf.SparseTensor(indices=self.ent_tuple,values=adj_value[:,batch*batch_size+head],dense_shape=(self.node_size,self.node_size))
                feature = batch_sparse_matmul(sparse_graph,random_projection(ent_feature,self.mini_dim))
                temp_list.append(feature)
            if len(temp_list):
                features_list.append(K.concatenate(temp_list,-1).numpy())
        features = np.concatenate(features_list,axis=-1)
        
        faiss.normalize_L2(features)
        # features = normalize_L2_np(features)
        if extra_feature is not None:
            features = np.concatenate([ent_feature,features],axis=-1)
        return features

    def train(self, epoch=None):
        # the parameter epochs is not used in the original code, we only include it here for compatibility with the baseline model
        with tf.device(f'/GPU:{Config.gpu}'):
            features = self.get_features(self.train_pair)
            left,right = list(self.candidates_x),list(self.candidates_y)
            index,sims = sparse_sinkhorn_sims(left,right,features,self.top_k)
            ranks = tf.argsort(-sims,-1).numpy()
            sims = sims.numpy(); index = index.numpy()
            new_pair = []
            x_list,y_list= list(self.candidates_x),list(self.candidates_y)
            for i in range(ranks.shape[0]):
                if sims[i,ranks[i,0]] > 0.5:
                    x = x_list[i]
                    y = y_list[index[i,ranks[i,0]]]
                    new_pair.append((x,y))

            dev_pair = list(map(tuple, self.test_pair))
            corrected_inferred = set(new_pair) & set(dev_pair)

            print(f"Correctly inferred {len(corrected_inferred)} pairs.")
            print(f"Total inferred {len(new_pair)} pairs.")

            right_list, wrong_list = test(self.test_pair,features,self.top_k)
            
            return new_pair

    def reset_data(self, train_pair):
        self.train_pair = np.array(train_pair)
        self.candidates_x = set([x for x,y in self.test_pair])
        self.candidates_y = set([y for x,y in self.test_pair])
        for x,y in train_pair:
            if x in self.candidates_x:
                self.candidates_x.remove(x)
            if y in self.candidates_y:
                self.candidates_y.remove(y)

    def fine_tune(self, epochs=3):
        with tf.device(f'/GPU:{Config.gpu}'):
            for epoch in range(epochs):
                features = self.get_features(self.train_pair)
                if epoch < epochs-1:
                    left,right = list(self.candidates_x),list(self.candidates_y)
                    index,sims = sparse_sinkhorn_sims(left,right,features,self.top_k)
                    ranks = tf.argsort(-sims,-1).numpy()
                    sims = sims.numpy(); index = index.numpy()

                    temp_pair = []
                    x_list,y_list= list(self.candidates_x),list(self.candidates_y)
                    for i in range(ranks.shape[0]):
                        if sims[i,ranks[i,0]] > 0.5:
                            x = x_list[i]
                            y = y_list[index[i,ranks[i,0]]]
                            temp_pair.append((x,y))

                    for x,y in temp_pair:
                        if x in self.candidates_x:
                            self.candidates_x.remove(x)
                        if y in self.candidates_y:
                            self.candidates_y.remove(y)

                    print("new generated pairs = %d"%(len(temp_pair)))
                    print("rest pairs = %d"%(len(self.candidates_x)))

                    if not len(temp_pair):
                        break
                    self.train_pair = np.concatenate([self.train_pair,np.array(temp_pair)])
                
                right_list, wrong_list = test(self.test_pair,features,self.top_k)


