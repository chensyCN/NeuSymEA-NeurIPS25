import gc
import sys
import os
import time
import random
import pdb
import numpy as np
import pickle

from tqdm import tqdm
from objects.KG import KG
import multiprocessing as mp
import json
pj_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pj_path)
from config import Config
from probabilisticReasoning import one_iteration_one_way, one_iteration_one_way_batch


sys.setrecursionlimit(1000000)


class KGs:
    def __init__(self, kg1: KG, kg2: KG, theta=0.1, iteration=3, workers=4, fusion_func=None, ground_truth_path=None):
        self.kg_l = kg1
        self.kg_r = kg2
        self.theta = theta
        self.iteration = iteration
        self.delta = 0.01
        self.epsilon = 1.01
        self.const = 10.0
        self.workers = workers
        self.fusion_func = fusion_func

        self.rel_ongoing_dict_l, self.rel_ongoing_dict_r = dict(), dict()
        self.rel_norm_dict_l, self.rel_norm_dict_r = dict(), dict()
        self.rel_align_dict_l, self.rel_align_dict_r = dict(), dict()

        self.sub_ent_match = None
        self.sup_ent_match = None
        self.sub_ent_prob = None
        self.sup_ent_prob = None

        self._iter_num = 0
        self.has_load = False
        self.util = KGsUtil(self, self.__get_counterpart_and_prob, self.__set_counterpart_and_prob)
        self.__init(ground_truth_path=ground_truth_path)

    def __init(self, ground_truth_path=None, ratio=0.3):
        if not self.kg_l.is_init():
            self.kg_l.init()
        if not self.kg_r.is_init():
            self.kg_r.init()

        kg_l_ent_num = len(self.kg_l.entity_set) + len(self.kg_l.literal_set)
        kg_r_ent_num = len(self.kg_r.entity_set) + len(self.kg_r.literal_set)
        self.sub_ent_match = [None for _ in range(kg_l_ent_num)]
        self.sub_ent_prob = [0.0 for _ in range(kg_l_ent_num)]
        self.sup_ent_match = [None for _ in range(kg_r_ent_num)]
        self.sup_ent_prob = [0.0 for _ in range(kg_r_ent_num)]

        self.gold_result = set() 
        with open(ground_truth_path, "r", encoding="utf8") as f:
            for line in f.readlines():
                params = str.strip(line).split("\t")
                ent_l, ent_r = params[0].strip(), params[1].strip()
                obj_l, obj_r = self.kg_l.entity_dict_by_name.get(ent_l), self.kg_r.entity_dict_by_name.get(
                    ent_r)
                if obj_l is None:
                    print("Exception: fail to load Entity (" + ent_l + ")")
                if obj_r is None:
                    print("Exception: fail to load Entity (" + ent_r + ")")
                if obj_l is None or obj_r is None:
                    continue
                self.gold_result.add((obj_l.id, obj_r.id))
        self.__split_data()
        self.annotated_alignments = set()
        for l_id, r_id in self.train_set:
            self.sub_ent_match[l_id], self.sub_ent_prob[l_id] = r_id, 1.0
            self.sup_ent_match[r_id], self.sup_ent_prob[r_id] = l_id, 1.0
            self.annotated_alignments.add((l_id, r_id))

    def __split_data(self):
        """
        split the gold result into training set and test set
        """
        gold_list = list(self.gold_result)
        random.shuffle(gold_list)
        train_num = int(len(gold_list) * Config.train_ratio)
        self.train_set = set(gold_list[:train_num])
        self.test_set = set(gold_list[train_num:])

    def __get_counterpart_and_prob(self, ent):
        source = ent.affiliation is self.kg_l
        counterpart_id = self.sub_ent_match[ent.id] if source else self.sup_ent_match[ent.id]
        if counterpart_id is None:
            return None, 0.0
        else:
            counterpart = self.kg_r.ent_lite_list_by_id[counterpart_id] if source \
                else self.kg_l.ent_lite_list_by_id[counterpart_id]
            return counterpart, self.sub_ent_prob[ent.id] if source else self.sup_ent_prob[ent.id]

    def __set_counterpart_and_prob(self, ent_l, ent_r, prob, force=False):
        source = ent_l.affiliation is self.kg_l
        l_id, r_id = ent_l.id, ent_r.id
        curr_prob = self.sub_ent_prob[l_id] if source else self.sup_ent_prob[l_id]
        if not force and prob < curr_prob:
            return False
        if source:
            self.sub_ent_match[l_id], self.sub_ent_prob[l_id] = r_id, prob
        else:
            self.sup_ent_match[l_id], self.sup_ent_prob[l_id] = r_id, prob
        return True

#####################################################################################################
# Functions for probabilistic reasoning
#####################################################################################################

    def inject_ea_inferred_pairs(self, pairs, ent_bias, filter=False, reinject=False):
        injected_pair = 0
        for (l, r) in pairs:
            r = r - ent_bias
            # for the reinjection, the inferred paris will override the previous ea inferred paris
            if reinject:
                if (l, r) in self.annotated_alignments:
                    continue
            else:
                if self.sub_ent_match[l] and self.sub_ent_prob[l] >= Config.delta_1 and self.sub_ent_match[r] and self.sub_ent_prob[r] >= Config.delta_1:
                    continue
            self.sub_ent_match[l], self.sub_ent_prob[l] = r, Config.delta_0
            self.sup_ent_match[r], self.sup_ent_prob[r] = l, Config.delta_0
            injected_pair += 1
            if filter:
                continue
            else:
                self.annotated_alignments.add((l, r))
        if filter:
            print(f"Injected {injected_pair} pairs out of {len(pairs)} pairs, but not include in the annotated alignments")
        else:
            print(f"Injected {injected_pair} pairs out of {len(pairs)} pairs")

    def set_fusion_func(self, func):
        self.fusion_func = func

    def set_iteration(self, iteration):
        self.iteration = iteration

    def set_worker_num(self, worker_num):
        self.workers = worker_num

    def warm_up(self):
        start_time = time.time()
        for i in range(self.iteration):
            self._iter_num = i
            print(str(i + 1) + "-th iteration......")
            self.__run_per_iteration()
            gc.collect()
        end_time = time.time()
        print("Warm up completed!")
        print("Total time: " + str(end_time - start_time))

    def run(self):
        start_time = time.time()
        self.util.test(gold_result=self.gold_result, threshold=[0.1 * i for i in range(10)])
        for i in tqdm(range(self.iteration), desc="Probabilistic Inference"):
            self._iter_num = i
            self.__run_per_iteration()
            self.util.test(gold_result=self.gold_result, threshold=[0.1 * i for i in range(10)])
            gc.collect()
        print("Probabilistic Inference Completed!")
        end_time = time.time()
        print("Total time: " + str(end_time - start_time))

    def test(self):
        self.util.test(gold_result=self.gold_result, threshold=[0.1 * i for i in range(10)])

    def __run_per_iteration(self):
        self.__run_per_iteration_one_way(self.kg_l)
        self.__ent_bipartite_matching()
        self.__run_per_iteration_one_way(self.kg_r, ent_align=False)
        return

    def __run_per_iteration_one_way(self, kg: KG, ent_align=True):
        """
        This function is used to run the one-way probabilistic inference.
        We enable multi-processing to speed up the inference.
        Each process is responsible for a batch of entities, 
        so as to avoid the overhead of process initialization and communication.
        """
        kg_other = self.kg_l if kg is self.kg_r else self.kg_r
        ent_list = self.__generate_list(kg)
        
        # Divide entity list into batches
        # BATCH_SIZE = 1000  # Can be adjusted according to actual situation
        BATCH_SIZE = len(ent_list) // self.workers
        ent_batches = [ent_list[i:i + BATCH_SIZE] for i in range(0, len(ent_list), BATCH_SIZE)]
        
        mgr = mp.Manager()
        # Use larger queue buffer
        ent_queue = mgr.Queue(max(len(ent_batches) * 2, 10))
        rel_ongoing_dict_queue = mgr.Queue()
        rel_norm_dict_queue = mgr.Queue()
        ent_match_tuple_queue = mgr.Queue()
        
        # Put batches into queue
        for batch in ent_batches:
            ent_queue.put(batch)

        kg_r_fact_dict_by_head = kg_other.fact_dict_by_head
        kg_l_fact_dict_by_tail = kg.fact_dict_by_tail
        kg_l_func, kg_r_func = kg.functionality_dict, kg_other.functionality_dict

        rel_align_dict_l, rel_align_dict_r = self.rel_align_dict_l, self.rel_align_dict_r

        if kg is self.kg_l:
            ent_match, ent_prob = self.sub_ent_match, self.sub_ent_prob
            is_literal_list_r = self.kg_r.is_literal_list
        else:
            ent_match, ent_prob = self.sup_ent_match, self.sup_ent_prob
            rel_align_dict_l, rel_align_dict_r = rel_align_dict_r, rel_align_dict_l
            is_literal_list_r = self.kg_l.is_literal_list

        init = not self.has_load and self._iter_num <= 1
        tasks = []
        kg_l_ent_embeds, kg_r_ent_embeds = kg.ent_embeddings, kg_other.ent_embeddings
        
        # Start worker processes
        for _ in range(self.workers):
            task = mp.Process(target=one_iteration_one_way_batch, args=(ent_queue, kg_r_fact_dict_by_head,
                                                                    kg_l_fact_dict_by_tail,
                                                                    kg_l_func, kg_r_func,
                                                                    ent_match, ent_prob,
                                                                    is_literal_list_r,
                                                                    rel_align_dict_l, rel_align_dict_r,
                                                                    rel_ongoing_dict_queue, rel_norm_dict_queue,
                                                                    ent_match_tuple_queue,
                                                                    kg_l_ent_embeds, kg_r_ent_embeds,
                                                                    self.fusion_func,
                                                                    self.theta, self.epsilon, self.delta, init,
                                                                    ent_align))
            task.start()
            tasks.append(task)

        try:
            # Wait for all processes to complete
            for task in tasks:
                task.join()

            # Process all results
            self.__clear_ent_match_and_prob(ent_match, ent_prob)
            while not ent_match_tuple_queue.empty():
                ent_match_tuple = ent_match_tuple_queue.get()
                self.__merge_ent_align_result(ent_match, ent_prob, ent_match_tuple[0], ent_match_tuple[1])

            # Clean up and update dictionaries
            rel_ongoing_dict = self.rel_ongoing_dict_l if kg is self.kg_l else self.rel_ongoing_dict_r
            rel_norm_dict = self.rel_norm_dict_l if kg is self.kg_l else self.rel_norm_dict_r
            rel_align_dict = self.rel_align_dict_l if kg is self.kg_l else self.rel_align_dict_r

            rel_ongoing_dict.clear()
            rel_norm_dict.clear()
            rel_align_dict.clear()

            while not rel_ongoing_dict_queue.empty():
                self.__merge_rel_ongoing_dict(rel_ongoing_dict, rel_ongoing_dict_queue.get())

            while not rel_norm_dict_queue.empty():
                self.__merge_rel_norm_dict(rel_norm_dict, rel_norm_dict_queue.get())

            self.__update_rel_align_dict(rel_align_dict, rel_ongoing_dict, rel_norm_dict)
                
        except Exception as e:
            print(f"Error in main process: {str(e)}")
            
        finally:
            # Ensure all processes are cleaned up
            for task in tasks:
                if task.is_alive():
                    task.terminate()
                    task.join(timeout=1)

    def __process_queued_results(self, ent_match, ent_prob, 
                               rel_ongoing_dict_queue, rel_norm_dict_queue,
                               ent_match_tuple_queue, batch_size=50):
        """Batch process results in queue"""
        try:
            # Process entity matching results
            ent_match_tuples = []
            while len(ent_match_tuples) < batch_size:
                try:
                    ent_match_tuples.append(ent_match_tuple_queue.get_nowait())
                except Exception:
                    break
                    
            for ent_match_tuple in ent_match_tuples:
                self.__merge_ent_align_result(ent_match, ent_prob, 
                                            ent_match_tuple[0], ent_match_tuple[1])

            # Process relation dictionary results
            rel_ongoing_dicts = []
            while len(rel_ongoing_dicts) < batch_size:
                try:
                    rel_ongoing_dicts.append(rel_ongoing_dict_queue.get_nowait())
                except Exception:
                    break
                    
            for rel_dict in rel_ongoing_dicts:
                self.__merge_rel_ongoing_dict(self.rel_ongoing_dict_l, rel_dict)

            # Process normalization dictionary results
            rel_norm_dicts = []
            while len(rel_norm_dicts) < batch_size:
                try:
                    rel_norm_dicts.append(rel_norm_dict_queue.get_nowait())
                except Exception:
                    break
                    
            for norm_dict in rel_norm_dicts:
                self.__merge_rel_norm_dict(self.rel_norm_dict_l, norm_dict)
                
        except Exception as e:
            print(f"Warning: Error processing queued results: {str(e)}")

    @staticmethod
    def __generate_list(kg: KG):
        ent_list = kg.ent_id_list
        random.shuffle(ent_list)
        return ent_list

    @staticmethod
    def __merge_rel_ongoing_dict(rel_dict_l, rel_dict_r):
        for (rel, rel_counterpart_dict) in rel_dict_r.items():
            if not rel_dict_l.__contains__(rel):
                rel_dict_l[rel] = rel_counterpart_dict
            else:
                for (rel_counterpart, prob) in rel_counterpart_dict.items():
                    if not rel_dict_l[rel].__contains__(rel_counterpart):
                        rel_dict_l[rel][rel_counterpart] = prob
                    else:
                        rel_dict_l[rel][rel_counterpart] += prob

    @staticmethod
    def __merge_rel_norm_dict(norm_dict_l, norm_dict_r):
        for (rel, norm) in norm_dict_r.items():
            if not norm_dict_l.__contains__(rel):
                norm_dict_l[rel] = norm
            else:
                norm_dict_l[rel] += norm

    @staticmethod
    def __update_rel_align_dict(rel_align_dict, rel_ongoing_dict, rel_norm_dict, const=10.0):
        for (rel, counterpart_dict) in rel_ongoing_dict.items():
            norm = rel_norm_dict.get(rel, 1.0)
            if not rel_align_dict.__contains__(rel):
                rel_align_dict[rel] = dict()
            rel_align_dict[rel].clear()
            for (counterpart, score) in counterpart_dict.items():
                prob = score / (const + norm)
                rel_align_dict[rel][counterpart] = prob

    def __ent_bipartite_matching(self):
        for ent_l in self.kg_l.entity_set:
            ent_id = ent_l.id
            counterpart_id, prob = self.sub_ent_match[ent_id], self.sub_ent_prob[ent_id]
            if counterpart_id is None:
                continue
            counterpart_prob = self.sup_ent_prob[counterpart_id]
            if counterpart_prob < prob:
                self.sup_ent_match[counterpart_id] = ent_id
                self.sup_ent_prob[counterpart_id] = prob
        for ent_l in self.kg_l.entity_set:
            ent_id = ent_l.id
            sub_counterpart_id = self.sub_ent_match[ent_id]
            if sub_counterpart_id is None:
                continue
            sup_counterpart_id = self.sup_ent_match[sub_counterpart_id]
            if sup_counterpart_id is None:
                continue
            if sup_counterpart_id != ent_id:
                self.sub_ent_match[ent_id], self.sub_ent_prob[ent_id] = None, 0.0

    @staticmethod
    def __merge_ent_align_result(ent_match_l, ent_prob_l, ent_match_r, ent_prob_r):
        assert len(ent_match_l) == len(ent_match_r)
        for i in range(len(ent_prob_l)):
            if ent_prob_l[i] < ent_prob_r[i]:
                ent_prob_l[i] = ent_prob_r[i]
                ent_match_l[i] = ent_match_r[i]

    @staticmethod
    def __clear_ent_match_and_prob(ent_match, ent_prob):
        for i in range(len(ent_match)):
            ent_match[i] = None
            ent_prob[i] = 0.0


class KGsUtil:
    def __init__(self, kgs, get_counterpart_and_prob, set_counterpart_and_prob):
        self.kgs = kgs
        self.__get_counterpart_and_prob = get_counterpart_and_prob
        self.__set_counterpart_and_prob = set_counterpart_and_prob
        self.ent_links_candidate = list()

    def reset_ent_align_result(self):
        for ent in self.kgs.kg_l.entity_set:
            idx = ent.id
            self.kgs.sub_ent_match[idx], self.kgs.sub_ent_prob[idx] = None, 0.0
        for ent in self.kgs.kg_r.entity_set:
            idx = ent.id
            self.kgs.sup_ent_match[idx], self.kgs.sup_ent_prob[idx] = None, 0.0
        emb_l, emb_r = self.kgs.kg_l.ent_embeddings, self.kgs.kg_r.ent_embeddings
        matrix = np.matmul(emb_l, emb_r.T)
        max_indices = np.argmax(matrix, axis=1)
        print(max_indices)
        for i in range(len(max_indices)):
            counterpart_id = max_indices[i]
            self.kgs.sub_ent_match[i], self.kgs.sub_ent_prob[i] = counterpart_id, 0.2
            self.kgs.sup_ent_match[counterpart_id], self.kgs.sup_ent_prob[counterpart_id] = i, 0.2

    def test(self, gold_result, threshold):

        threshold_list = []
        if isinstance(threshold, float) or isinstance(threshold, int):
            threshold_list.append(float(threshold))
        else:
            threshold_list = threshold

        for threshold_item in threshold_list:
            ent_align_result = set()
            for ent_id in self.kgs.kg_l.ent_id_list:
                counterpart_id = self.kgs.sub_ent_match[ent_id]
                if counterpart_id is not None:
                    prob = self.kgs.sub_ent_prob[ent_id]
                    if prob < threshold_item:
                        continue
                    ent_align_result.add((ent_id, counterpart_id))
            if Config.print_during_exp['paris']:
                self.print_metrics(gold_result, ent_align_result)

    def print_metrics(self, gold_result, predictions):
        correct_num = len(gold_result & predictions)
        predict_num = len(predictions)
        total_num = len(gold_result)

        if predict_num == 0:
            print("Exception: no satisfied alignment result")
            return

        if total_num == 0:
            print("Exception: no satisfied instance for testing")
        else:
            precision, recall = correct_num / predict_num, correct_num / total_num

            if precision <= 0.0 or recall <= 0.0:
                print("Precision: " + format(precision, ".4f") +
                      "\tRecall: " + format(recall, ".4f") + "\tF1-Score: 0.0")
            else:
                f1_score = 2.0 * precision * recall / (precision + recall)
                print("Precision: " + format(precision, ".4f") +
                      "\tRecall: " + format(recall, ".4f") + "\tF1-Score: " + format(f1_score, ".4f"))

    def generate_input_for_emb_model(self, filter=False, augment=False):

        entity1 = set([ent.id for ent in self.kgs.kg_l.entity_set])
        rel1 = set([rel.id for rel in self.kgs.kg_l.relation_set])
        triples1 = [(h.id, rel.id, t.id) for (h, rel, t) in self.kgs.kg_l.relation_tuple_list]

        entity2 = set([ent.id for ent in self.kgs.kg_r.entity_set])
        rel2 = set([rel.id for rel in self.kgs.kg_r.relation_set])
        triples2 = [(h.id, rel.id, t.id) for (h, rel, t) in self.kgs.kg_r.relation_tuple_list]

        # train_pair = self.kgs.refined_alignments
        train_pair = set()
        for ent in self.kgs.kg_l.entity_set:
            counterpart, prob = self.__get_counterpart_and_prob(ent)
            if (filter and (prob < Config.delta_1)) or counterpart is None:
                continue
            train_pair.add((ent.id, counterpart.id))
        train_pair = train_pair.intersection(self.kgs.annotated_alignments)
        print(f"Number of refined train pairs: {len(train_pair)} from {len(self.kgs.annotated_alignments)} annotated pairs")
        # all labels with high confidence are inferred
        if augment:
            for ent in self.kgs.kg_l.entity_set:
                counterpart, prob = self.__get_counterpart_and_prob(ent)
                if prob < Config.delta_1 or counterpart is None:
                    continue
                train_pair.add((ent.id, counterpart.id))
        print(f"Number of overall refined and inferred train pairs: {len(train_pair)}")
        self.print_metrics(self.kgs.gold_result, train_pair)
        # the recall of annotated true pairs
        annotated_true = self.kgs.annotated_alignments.intersection(self.kgs.gold_result)
        recalled_true = train_pair.intersection(self.kgs.gold_result)
        print(f"recall of annotated true pairs: {len(recalled_true)}/{len(annotated_true)}")
        print(f"precision of annotated true pairs: {len(recalled_true)}/{len(train_pair)}")
        dev_pair = self.kgs.test_set

        # make sure that the entity/relation id in the two KGs have no intersection by adding a bias value
        ent_bias = max(entity1) + 1
        rel_bias = max(rel1) + 1
        entity2 = set([ent + ent_bias for ent in entity2])
        rel2 = set([rel + rel_bias for rel in rel2])
        triples2 = [(h + ent_bias, r + rel_bias, t + ent_bias) for (h, r, t) in triples2]
        train_pair = [(l, r+ent_bias) for (l, r) in train_pair]
        dev_pair = [(l, r+ent_bias) for (l, r) in dev_pair]
        
        return (entity1, rel1, triples1, entity2, rel2, triples2, train_pair, dev_pair), (ent_bias, rel_bias)

    def generate_input_for_emb_model_active_only(self):
        """
        this function is used to generate input for the embedding model,
        the training set is the self.annotation result
        """

        entity1 = set([ent.id for ent in self.kgs.kg_l.entity_set])
        rel1 = set([rel.id for rel in self.kgs.kg_l.relation_set])
        triples1 = [(h.id, rel.id, t.id) for (h, rel, t) in self.kgs.kg_l.relation_tuple_list]

        entity2 = set([ent.id for ent in self.kgs.kg_r.entity_set])
        rel2 = set([rel.id for rel in self.kgs.kg_r.relation_set])
        triples2 = [(h.id, rel.id, t.id) for (h, rel, t) in self.kgs.kg_r.relation_tuple_list]

        train_pair = self.kgs.annotated_alignments
        dev_pair = self.kgs.test_set

        # make sure that the entity/relation id in the two KGs have no intersection by adding a bias value
        ent_bias = max(entity1) + 1
        rel_bias = max(rel1) + 1
        entity2 = set([ent + ent_bias for ent in entity2])
        rel2 = set([rel + rel_bias for rel in rel2])
        triples2 = [(h + ent_bias, r + rel_bias, t + ent_bias) for (h, r, t) in triples2]
        train_pair = [(l, r+ent_bias) for (l, r) in train_pair]
        dev_pair = [(l, r+ent_bias) for (l, r) in dev_pair]
        print(f"precision of annotated true pairs: {len(set(train_pair).intersection(set(dev_pair)))}/{len(train_pair)}")

        return (entity1, rel1, triples1, entity2, rel2, triples2, train_pair, dev_pair), (ent_bias, rel_bias)

    def get_mappings(self):
        ent_dict, lite_dict, attr_dict, rel_dict = dict(), dict(), dict(), dict()
        for obj in (self.kgs.kg_l.entity_set | self.kgs.kg_l.literal_set):
            counterpart, prob = self.__get_counterpart_and_prob(obj)
            if counterpart is not None:
                if obj.is_literal():
                    lite_dict[(obj, counterpart)] = [prob]
                else:
                    ent_dict[(obj, counterpart)] = [prob]

        for (rel_id, rel_counterpart_id_dict) in self.kgs.rel_align_dict_l.items():
            rel = self.kgs.kg_l.rel_attr_list_by_id[rel_id]
            dictionary = attr_dict if rel.is_attribute() else rel_dict
            for (rel_counterpart_id, prob) in rel_counterpart_id_dict.items():
                if prob > self.kgs.theta:
                    rel_counterpart = self.kgs.kg_r.rel_attr_list_by_id[rel_counterpart_id]
                    dictionary[(rel, rel_counterpart)] = [prob, 0.0]

        for (rel_id, rel_counterpart_id_dict) in self.kgs.rel_align_dict_r.items():
            rel = self.kgs.kg_r.rel_attr_list_by_id[rel_id]
            dictionary = attr_dict if rel.is_attribute() else rel_dict
            for (rel_counterpart_id, prob) in rel_counterpart_id_dict.items():
                if prob > self.kgs.theta:
                    rel_counterpart = self.kgs.kg_l.rel_attr_list_by_id[rel_counterpart_id]
                    if not dictionary.__contains__((rel_counterpart, rel)):
                        dictionary[(rel_counterpart, rel)] = [0.0, 0.0]
                    dictionary[(rel_counterpart, rel)][-1] = prob
        
        return ent_dict, lite_dict, attr_dict, rel_dict

    def save_params(self, path="output/EA_Params"):
        base, _ = os.path.split(path)
        if not os.path.exists(base):
            os.makedirs(base)
        with open(path, "w", encoding="utf8") as f:
            for obj in (self.kgs.kg_l.entity_set | self.kgs.kg_l.literal_set):
                counterpart, prob = self.__get_counterpart_and_prob(obj)
                if counterpart is not None:
                    f.write("\t".join(["L", obj.name, counterpart.name, str(prob)]) + "\n")
            for obj in (self.kgs.kg_r.entity_set | self.kgs.kg_r.literal_set):
                counterpart, prob = self.__get_counterpart_and_prob(obj)
                if counterpart is not None:
                    f.write("\t".join(["R", obj.name, counterpart.name, str(prob)]) + "\n")
            for (rel_id, rel_counterpart_id_dict) in self.kgs.rel_align_dict_l.items():
                rel = self.kgs.kg_l.rel_attr_list_by_id[rel_id]
                for (rel_counterpart_id, prob) in rel_counterpart_id_dict.items():
                    if prob > 0.0:
                        rel_counterpart = self.kgs.kg_r.rel_attr_list_by_id[rel_counterpart_id]
                        prefix = "L"
                        f.write("\t".join([prefix, rel.name, rel_counterpart.name, str(prob)]) + "\n")
            for (rel_id, rel_counterpart_id_dict) in self.kgs.rel_align_dict_r.items():
                rel = self.kgs.kg_r.rel_attr_list_by_id[rel_id]
                for (rel_counterpart_id, prob) in rel_counterpart_id_dict.items():
                    if prob > 0.0:
                        rel_counterpart = self.kgs.kg_l.rel_attr_list_by_id[rel_counterpart_id]
                        prefix = "R"
                        f.write("\t".join([prefix, rel.name, rel_counterpart.name, str(prob)]) + "\n")
        
        # save the train and annotated data
        with open(path + "_train", "w", encoding="utf8") as f:
            for (l, r) in list(self.kgs.train_set):
                f.write(f"{l}\t{r}\n")
        with open(path + "_annotated", "w", encoding="utf8") as f:
            for (l, r) in list(self.kgs.annotated_alignments):
                f.write(f"{l}\t{r}\n")
        return

    def load_params(self, path="output/EA_Params", init=True):
        self.kgs.has_load = init

        # clear kgs
        self.kgs.train_set.clear()
        self.kgs.annotated_alignments.clear()
        self.kgs.sub_ent_match = [None for _ in range(len(self.kgs.sub_ent_match))]
        self.kgs.sup_ent_match = [None for _ in range(len(self.kgs.sup_ent_match))]
        self.kgs.sub_ent_prob = [0.0 for _ in range(len(self.kgs.sub_ent_prob))]
        self.kgs.sup_ent_prob = [0.0 for _ in range(len(self.kgs.sup_ent_prob))]

        def get_obj_by_name(kg_l, kg_r, name1, name2):
            obj1, obj2 = kg_l.literal_dict_by_name.get(name1), kg_r.literal_dict_by_name.get(name2)
            if obj1 is None or obj2 is None:
                obj1, obj2 = kg_l.entity_dict_by_name.get(name1), kg_r.entity_dict_by_name.get(name2)
            if obj1 is None or obj2 is None:
                obj1, obj2 = kg_l.entity_dict_by_name.get(name1), kg_r.entity_dict_by_name.get(name2)
            if obj1 is None or obj2 is None:
                obj1, obj2 = kg_l.relation_dict_by_name.get(name1), kg_r.relation_dict_by_name.get(name2)
            if obj1 is None or obj2 is None:
                obj1, obj2 = kg_l.attribute_dict_by_name.get(name1), kg_r.attribute_dict_by_name.get(name2)
            return obj1, obj2

        with open(path, "r", encoding="utf8") as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                params = line.strip().split("\t")
                assert len(params) == 4
                prefix, name_l, name_r, prob = params[0].strip(), params[1].strip(), params[2].strip(), float(
                    params[3].strip())
                if prefix == "L":
                    obj_l, obj_r = get_obj_by_name(self.kgs.kg_l, self.kgs.kg_r, name_l, name_r)
                else:
                    obj_l, obj_r = get_obj_by_name(self.kgs.kg_r, self.kgs.kg_l, name_l, name_r)
                assert (obj_l is not None and obj_r is not None)
                if obj_l.is_entity():
                    idx_l = obj_l.id
                    if prefix == "L":
                        self.kgs.sub_ent_match[idx_l], self.kgs.sub_ent_prob[idx_l] = obj_r.id, prob
                    else:
                        self.kgs.sup_ent_match[idx_l], self.kgs.sup_ent_prob[idx_l] = obj_r.id, prob
                else:
                    if prefix == "L":
                        self.__params_loader_helper(self.kgs.rel_align_dict_l, obj_l.id, obj_r.id, prob)
                    else:
                        self.__params_loader_helper(self.kgs.rel_align_dict_r, obj_l.id, obj_r.id, prob)
        

        # load checkpoint
        with open(path + "_train", "r", encoding="utf8") as f:
            for line in f.readlines():
                params = line.strip().split("\t")
                assert len(params) == 2
                l, r = int(params[0].strip()), int(params[1].strip())
                self.kgs.train_set.add((l, r))
        with open(path + "_annotated", "r", encoding="utf8") as f:
            for line in f.readlines():
                params = line.strip().split("\t")
                assert len(params) == 2
                l, r = int(params[0].strip()), int(params[1].strip())
                self.kgs.annotated_alignments.add((l, r))
        
        return

    @staticmethod
    def __params_loader_helper(dict_by_key: dict, key1, key2, value):
        if not dict_by_key.__contains__(key1):
            dict_by_key[key1] = dict()
        dict_by_key[key1][key2] = value
