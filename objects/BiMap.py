import pdb
import random
from collections import defaultdict

random.seed(0)

class BiMap:
    
    def __init__(self, links, is_relation=False, have_scores=False):

        self.is_relation = is_relation
        self.have_scores = have_scores

        # for predicted ent/rel mapping
        if have_scores:
            self.score_dict = defaultdict(float)
            if is_relation:
                for rel1, rel2, score1, score2 in links:
                    self.score_dict[(rel1, rel2)] = (float(score1) + float(score2)) / 2
            else:
                for ent1, ent2, score in links:
                    self.score_dict[(ent1, ent2)] = float(score)
            
            # eleminate the scores to be compatibale with following code
            links = [link[:2] for link in links]

        if is_relation:
            self.AtoB, self.BtoA = defaultdict(set), defaultdict(set)
            for rel1, rel2 in links:
                self.AtoB[rel1].add(rel2)
                self.BtoA[rel2].add(rel1)
        else:
            self.AtoB = dict(links)
            self.BtoA = {v: k for k, v in self.AtoB.items()}

    def sample_aligned_ent_pair(self):
        if not self.have_scores:
            ent_A_list = list(self.AtoB.keys())
            sampled_A_ent = random.choice(ent_A_list)
            sampled_B_ent = self.AtoB[sampled_A_ent]
            return (sampled_A_ent, sampled_B_ent)

    def sample_unaligned_ent_pair(self):
        if not self.have_scores:
            ent_A_list = list(self.AtoB.keys())
            ent_B_list = list(self.BtoA.keys())
            sampled_A_ent = random.choice(ent_A_list)
            sampled_B_ent = random.choice(ent_B_list)
            while sampled_B_ent == self.AtoB[sampled_A_ent]:
                sampled_B_ent = random.choice(ent_B_list)
            return (sampled_A_ent, sampled_B_ent)

    def getB(self, nodeA):
        if not self.have_scores:
            return self.AtoB.get(nodeA)

    def getA(self, nodeB):
        if not self.have_scores:
           return self.BtoA.get(nodeB)

    def getCoupleNodes(self, keys, sourceGraph, check_existence=True):
        if sourceGraph == 'kg1':
            keys = [key for key in keys if key in self.AtoB] if check_existence else keys
            return [self.AtoB[key] for key in keys]
        elif sourceGraph == 'kg2':
            keys = [key for key in keys if key in self.BtoA] if check_existence else keys
            return [self.BtoA[key] for key in keys]

    def getAlignedNodes(self, sourceGraph):
        if sourceGraph == 'kg1':
            return self.AtoB.keys()
        elif sourceGraph == 'kg2':
            return self.BtoA.keys()

    def get_path_similarity_score(self, rel_pair_list):
        """
        rel_pairs: a list of relation pairs, e.g. [(rel1, rel2), (rel3, rel4), ...]
        """
        # path similarity scores are measured by the average scores of alignment prob of each corresponding relation pair
        scores = []
        # pdb.set_trace()
        for rel_pair in rel_pair_list:
            if rel_pair in self.score_dict:
                scores.append(self.score_dict[rel_pair])
            else:
                return 0.0
        return sum(scores) / len(scores)


    def __repr__(self) -> str:
        return f' A bi-directional map to represent the aligned entities.\n "is_relation" is {self.is_relation}.\n "have_scores" is {self.have_scores}.\n'