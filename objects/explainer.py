
from collections import defaultdict
import pdb
import time
import networkx as nx
import os
import sys
import numpy as np
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base)

from config import Config
from objects.BiMap import BiMap


class Explainer:    

    def __init__(self, KGs):

        self.kg1 = None
        self.kg2 = None
        self.kg1_attr = None
        self.kg2_attr = None

        self.kg1_functionality = None
        self.kg2_functionality = None

        self.path2ReachableNodes_kg1 = None
        self.path2ReachableNodes_kg2 = None

        self._load_dataset(KGs)

#===================================================================================================
# Functions for loading dataset
#===================================================================================================
    @staticmethod
    def _get_nx_kg(KG):
        nxG = nx.DiGraph()
        rel2head = defaultdict(set)
        head2rel = defaultdict(set)

        for h, r, t in KG.relation_tuple_list:
            h, r, t = h.name, r.name, t.name
            nxG.add_edge(h, t, relation=r)
            rel2head[r].add(h)
            head2rel[h].add(r)

        return nxG, dict(rel2head), dict(head2rel)

    @staticmethod
    def _get_attr_dict(KG):
        attr_dict = defaultdict(dict)
        for attr_tuple in KG.attribute_tuple_list:
            if len(attr_tuple) != 3:
                continue
            ent, attr, val = attr_tuple[0].name, attr_tuple[1].name, attr_tuple[2].name
            attr_dict[ent][attr] = val
        return attr_dict

    @staticmethod
    def _get_kg_statistics(KG):
        kg_functionality = dict()
        for rel in KG.relation_set:
            kg_functionality[rel.name] = rel.functionality
        return kg_functionality

    @staticmethod
    def _other_kg(kg):
        if kg == "kg1":
            return "kg2"
        elif kg == "kg2":
            return "kg1"
        else:
            raise ValueError(f"kg should be kg1 or kg2, but got {kg}")

    def _get_kg(self, kg):
        if kg == "kg1":
            return self.kg1
        elif kg == "kg2":
            return self.kg2

    def update_mappings(self, KGs):
        ent_dict, lite_dict, attr_dict, rel_dict = KGs.util.get_mappings()

        ent_mappings = []
        num_ent_mappings = 0
        for k, v in ent_dict.items():
            ent, counterpart = k
            prob = v[0]
            # for soft anchor mode, uncomment the following code if you want to use soft anchor mode
            # if prob > 0.9:
            #     ent_mappings.append((ent.name, counterpart.name, prob))
            
            # for hard anchor mode, uncomment the following code if you want to use hard anchor mode
            if prob >= 1.0 and num_ent_mappings <= 4500:
                num_ent_mappings += 1
                ent_mappings.append((ent.name, counterpart.name, prob))

        rel_mappings = []
        for k, v in rel_dict.items():
            rel, counterpart = k
            prob_l2r, prob_r2l = v[0], v[1]
            # if prob_l2r > 0.1 and prob_r2l > 0.1:
            rel_mappings.append((rel.name, counterpart.name, prob_l2r, prob_r2l))
        # pdb.set_trace()
        anchor_mappings = BiMap(ent_mappings, have_scores=True)
        rel_mappings = BiMap(rel_mappings, is_relation=True, have_scores=True)
        
        return anchor_mappings, rel_mappings

    def _load_dataset(self, KGs):
        self.KGs = KGs
        kg_l, kg_r = KGs.kg_l, KGs.kg_r
        self.kg1, self.kg1_rel2head, self.kg1_head2rel = self._get_nx_kg(kg_l)
        self.kg2, self.kg2_rel2head, self.kg2_head2rel = self._get_nx_kg(kg_r)
        self.kg1_attr = self._get_attr_dict(kg_l)
        self.kg2_attr = self._get_attr_dict(kg_r)
        self.kg1_functionality = self._get_kg_statistics(kg_l)
        self.kg2_functionality = self._get_kg_statistics(kg_r)
        self.anchor_mappings, self.rel_mappings = self.update_mappings(KGs)
        # self.gold_mappings = BiMap(gold_mappings, have_scores=False)

#===================================================================================================
# Functions for finding paths to reachable alignments of a pair of entities
#===================================================================================================

    @staticmethod
    def path_importance(r_list, kg_functionality):
        # path: list of triples, the second element of each triple is the relation
        # avg_out_degree: dict, key is relation, value is average out degree of that relation
        # return the importance of the path by multiplying the inverse of average out degree of each relation

        importance = 1
        for r in r_list:
            importance *= kg_functionality[r]
        return importance

    def path_sim(self, path1, path2):
        if len(path1) != len(path2):
            return 0.0
        else:
            rel_pair_list = [(path1[i], path2[i]) for i in range(len(path1))]
            return self.rel_mappings.get_path_similarity_score(rel_pair_list)

    def get_nodes_within_distance_on_the_fly(self, kg, node):
        if kg == "kg1":
            KG = self.kg1
            kg_functionality = self.kg1_functionality
        elif kg == "kg2":
            KG = self.kg2
            kg_functionality = self.kg2_functionality

        path2ReachableNodes = dict()

        # Perform a breadth-first search from 'node', limiting to 'max_dist' levels deep
        bfs_predecessors = dict(nx.bfs_predecessors(KG, node, depth_limit=Config.max_dist))
        # The nodes in the BFS tree are the nodes within 'max_dist' of 'node'
        path2ReachableNodes[node] = dict()
        for reachable_node in bfs_predecessors.keys():
            path = [reachable_node]
            while path[-1] != node:
                path.append(bfs_predecessors[path[-1]])
            
            # compute path importance and store it in the dict
            r_list = [KG[path[i-1]][path[i]]['relation'] for i in range(1, len(path))]
            path.reverse()
            path_importance = self.path_importance(r_list, kg_functionality)
            triples_path = [(path[i], KG[path[i-1]][path[i]]['relation'].replace("-(INV)", ""), path[i-1]) if KG[path[i-1]][path[i]]['relation'].endswith('-(INV)') else (path[i-1], KG[path[i-1]][path[i]]['relation'], path[i]) for i in range(1, len(path))]
            path2ReachableNodes[node][reachable_node] = (triples_path, path_importance, r_list)

        if kg == "kg1":
            self.path2ReachableNodes_kg1 = path2ReachableNodes
        elif kg == "kg2":
            self.path2ReachableNodes_kg2 = path2ReachableNodes

    def get_aligned_neighbors_counterpart(self, kg, node):
        if kg == "kg1":
            KG = self.kg1
        elif kg == "kg2":
            KG = self.kg2
        
        bfs_predecessors = dict(nx.bfs_predecessors(KG, node, depth_limit=Config.max_dist))
        neighbors = set(bfs_predecessors.keys())
        counterparts = self.anchor_mappings.getCoupleNodes(neighbors, kg)

        return counterparts

    def find_reachable_alignments(self, node1, node2):
        """
        ent_links: links showing aligned entities, represented as BiMap

        this function find reachable aligned entities by following steps:
        1. get reachable nodes of node1 in kg1 using p2RN2, called RN_node1;
            and reachable nodes of node2 in kg2 using p2RN2, called RN_node2;
        2. find the counterpart entities of RN_node1 by a "filter and map" step, called RN_node1_counterpart;
        3. get the intersection of RN_node2 and RN_node1_counterpart, return it and its counterpart in kg1
            called `reachable_alignments`, represented as a list of tuples.
        """
        # if Config.parse_path_on_the_fly:
        self.get_nodes_within_distance_on_the_fly(kg="kg1", node=node1)
        self.get_nodes_within_distance_on_the_fly(kg="kg2", node=node2)
        RN_node1 = list(self.path2ReachableNodes_kg1[node1].keys())
        RN_node2 = set(self.path2ReachableNodes_kg2[node2].keys())
        RN_node1_counterpart = self.anchor_mappings.getCoupleNodes(RN_node1, sourceGraph='kg1')
        reachable_aligned_node_in_kg2 =list(set(RN_node1_counterpart) & RN_node2)
        reachable_aligned_node_in_kg1 = self.anchor_mappings.getCoupleNodes(reachable_aligned_node_in_kg2, sourceGraph='kg2', check_existence=False)
        reachable_alignments = list(zip(reachable_aligned_node_in_kg1, reachable_aligned_node_in_kg2))
        return reachable_alignments

    def parse_paths_to_alignments(self, node1, node2, reachable_alignments, bottom_k=False):
        """
        reachable_alignments: a list of tuples, each tuple is a pair of reachable aligned entities
            in kg1 and kg2, respectively.

        this function parse the paths between reachable aligned entities to alignments, paths are sorted by the confidence

        confidence = (path1-importance+path2-importance)*path_sim/2
        """

        paths_to_alignments = []

        # List to store index along with alignment data
        indexed_paths_to_alignments = []     
        for index, alignment in enumerate(reachable_alignments):
            path1_and_importance = self.path2ReachableNodes_kg1[node1][alignment[0]]
            path2_and_importance = self.path2ReachableNodes_kg2[node2][alignment[1]]
            path_pair_similarity = self.path_sim(path1_and_importance[2], path2_and_importance[2])
            # Store index along with alignments
            indexed_paths_to_alignments.append((index, (path1_and_importance, path2_and_importance, path_pair_similarity))) 

        if bottom_k:  # arguement is used in case study, to compare the top-k and bottom-k path's
            indexed_paths_to_alignments = sorted(indexed_paths_to_alignments, key=lambda x: (x[1][0][1] + x[1][1][1])*x[1][2])
        else:
            indexed_paths_to_alignments = sorted(indexed_paths_to_alignments, key=lambda x: (x[1][0][1] + x[1][1][1])*x[1][2], reverse=True)

        # Extract top-k indices for corresponding elements in reachable_alignments
        top_k_indices = [x[0] for x in indexed_paths_to_alignments[:Config.maximum_common_aligned_neighbors]]
        
        # Using top-k indices to select corresponding elements in reachable_alignments
        top_k_reachable_alignments = [reachable_alignments[i] for i in top_k_indices]

        # Select the top-k paths_to_alignments as well
        paths_to_alignments = [x[1] for x in indexed_paths_to_alignments[:Config.maximum_common_aligned_neighbors]]
        # print("\nNumber of reachable alignments:", len(reachable_alignments))
        
        # if the highest confidence is less than 0.5, then return None, used for case study
        if len(paths_to_alignments) == 0:
            return None, None
        if (paths_to_alignments[0][0][1] + paths_to_alignments[0][1][1])/2*paths_to_alignments[0][2] < Config.conf_thres:
            return None, None
        print("\nImportance of top-k paths:", [(x[0][1] + x[1][1])/2*x[2] for x in paths_to_alignments])

        paths_to_alignments = [(x[0][0], x[1][0]) for x in paths_to_alignments]

        # return top_k_reachable_alignments, paths_to_alignments
        top_k_reachable_alignments, paths_to_alignments = self.simplify_paths(top_k_reachable_alignments, paths_to_alignments)
        
        return top_k_reachable_alignments, paths_to_alignments

    def simplify_paths(self, reachable_alignments, paths_to_alignments):
        """
        simplify the representations of alignments, by replacing the names of relations and entities, with their values.
        """
        reachable_alignments = [(self.KGs.kg_l.entity_dict_by_name[h].value, self.KGs.kg_r.entity_dict_by_name[t].value) for h, t in reachable_alignments]
        simplified_paths_to_alignments = []
        for path_to_alignment in paths_to_alignments:
            path1, path2 = path_to_alignment
            path1 = [(self.KGs.kg_l.entity_dict_by_name[h].value, self.KGs.kg_l.relation_dict_by_name[r].value, self.KGs.kg_l.entity_dict_by_name[t].value) for h, r, t in path1]
            path2 = [(self.KGs.kg_r.entity_dict_by_name[h].value, self.KGs.kg_r.relation_dict_by_name[r].value, self.KGs.kg_r.entity_dict_by_name[t].value) for h, r, t in path2]
            simplified_paths_to_alignments.append((path1, path2))

        return reachable_alignments, simplified_paths_to_alignments

    def is_inferred(self, node1, node2):
        """
        check whether it is an inferred alignment, if not ,return False
        """
        return self.anchor_mappings.AtoB.get(node1)== node2

    def explain(self, node1, node2, bottom_k=False):
        reachable_alignments = self.find_reachable_alignments(node1, node2)
        reachable_alignments, paths_to_alignments = self.parse_paths_to_alignments(node1, node2, reachable_alignments, bottom_k=bottom_k)
        if reachable_alignments is None:
            return
        # print explanations
        print(f"\nExplanation for {node1} and {node2}")
        for index, (reachable_alignment, paths_to_alignment) in enumerate(zip(reachable_alignments, paths_to_alignments)):
            print(f"\nExplanation {index+1}:")
            print(f"Reachable alignment: {reachable_alignment}")
            print(f"Paths to alignment:")
            print(f"Path 1: {paths_to_alignment[0]}")
            print(f"Path 2: {paths_to_alignment[1]}")

    def count_supporting_alignments(self, node1, node2):
        """
        count the number of supporting alignments with score larger than Config.conf_thres
        """

        # for positve pairs, if it is not an inferred alignment, then return None
        if not self.is_inferred(node1, node2):
            return None

        reachable_alignments = self.find_reachable_alignments(node1, node2)

        num_supports = 0

        for index, alignment in enumerate(reachable_alignments):
            path1_and_importance = self.path2ReachableNodes_kg1[node1][alignment[0]]
            path2_and_importance = self.path2ReachableNodes_kg2[node2][alignment[1]]
            path_pair_similarity = self.path_sim(path1_and_importance[2], path2_and_importance[2])

            if (path1_and_importance[1]+path2_and_importance[1])/2*path_pair_similarity > Config.conf_thres:
                num_supports += 1
        
        return num_supports

    def avg_confidence_of_topk_supports(self, node1, node2, is_positive=False):

        if not self.is_inferred(node1, node2) and is_positive:
            return None, None

        reachable_alignments = self.find_reachable_alignments(node1, node2)
        confidences = []
        for index, alignment, in enumerate(reachable_alignments):
            
            path1_and_importance = self.path2ReachableNodes_kg1[node1][alignment[0]]
            path2_and_importance = self.path2ReachableNodes_kg2[node2][alignment[1]]
            path_pair_similarity = self.path_sim(path1_and_importance[2], path2_and_importance[2])
            confidence = (path1_and_importance[1]+path2_and_importance[1])/2*path_pair_similarity
            if confidence >= Config.conf_thres:
                confidences.append(confidence)
        

        confidences = np.array(confidences)
        sorted_indices = np.argsort(confidences)[::-1]
        topk_confidences = confidences[sorted_indices[:Config.maximum_common_aligned_neighbors]]

        avg_conf = np.mean(topk_confidences) if len(topk_confidences) > 0 else 0.0
        max_conf = np.max(topk_confidences) if len(topk_confidences) > 0 else 0.0

        return avg_conf, max_conf
            
