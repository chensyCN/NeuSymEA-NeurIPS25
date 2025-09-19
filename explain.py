import os
import pdb
import pickle
import concurrent.futures
import argparse
import random
import time
from tqdm import tqdm


from objects.KG import KG
from objects.KGs import KGs
from objects.explainer import Explainer
from config import Config


argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default="fr_en")
argparser.add_argument('--iter', type=int, default=3)
argparser.add_argument('--train_ratio', type=float, default=0.1, help="the ratio of training data")
args = argparser.parse_args()

Config.train_ratio = args.train_ratio
Config.print_during_exp['paris'] = True



def construct_kg(path_r, path_a=None, sep='\t', name=None):
    kg = KG(name=name)
    if path_a is not None:
        with open(path_r, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                params = str.strip(line).split(sep=sep)
                if len(params) != 3:
                    print(line)
                    continue
                h, r, t = params[0].strip(), params[1].strip(), params[2].strip()
                kg.insert_relation_tuple(h, r, t)

        with open(path_a, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                params = str.strip(line).split(sep=sep)
                if len(params) != 3:
                    print(line)
                    continue
                # assert len(params) == 3
                e, a, v = params[0].strip(), params[1].strip(), params[2].strip()
                kg.insert_attribute_tuple(e, a, v)
    else:
        with open(path_r, "r", encoding="utf-8") as f:
            prev_line = ""
            for line in f.readlines():
                params = line.strip().split(sep)
                if len(params) != 3 or len(prev_line) == 0:
                    prev_line += "\n" if len(line.strip()) == 0 else line.strip()
                    continue
                prev_params = prev_line.strip().split(sep)
                e, a, v = prev_params[0].strip(), prev_params[1].strip(), prev_params[2].strip()
                prev_line = "".join(line)
                if len(e) == 0 or len(a) == 0 or len(v) == 0:
                    print("Exception: " + e)
                    continue
                if v.__contains__("http"):
                    kg.insert_relation_tuple(e, a, v)
                else:
                    kg.insert_attribute_tuple(e, a, v)
    kg.init()
    kg.print_kg_info()
    return kg


def construct_kgs(dataset_dir, name="KGs", load_chk=None):
    path_r_1 = os.path.join(dataset_dir, "rel_triples_1")
    path_a_1 = os.path.join(dataset_dir, "attr_triples_1")

    path_r_2 = os.path.join(dataset_dir, "rel_triples_2")
    path_a_2 = os.path.join(dataset_dir, "attr_triples_2")

    kg1 = construct_kg(path_r_1, path_a_1, name=str(name + "-KG1"))
    kg2 = construct_kg(path_r_2, path_a_2, name=str(name + "-KG2"))
    kgs = KGs(kg1=kg1, kg2=kg2, ground_truth_path=os.path.join(dataset_path, "ent_links"))
    # load the previously saved PRASE model
    if load_chk is not None:
        kgs.util.load_params(load_chk)
    

    return kgs


def test(kgs, links):
    # configurations for the explainer:
    Config.max_dist = 3
    Config.maximum_common_aligned_neighbors = 2
    Config.conf_thres = 0.0 # this means filtering out the interpretations with confidence less than 0.0
    ex = Explainer(kgs)

    num_evaluated = 5 # the number of links you want to generate explanations for
    for line in random.sample(links[-6000:], num_evaluated):
        n1, n2 = line.strip().split("\t")
        ex.explain(n1, n2)


def num_align_to_max_dist(kgs, links, threshold=0.0, path=None):
    """
    to evaluate how many supporting alignments with confidence larger than the threshold, with respect to the neighbor hops
    """
    Config.conf_thres = threshold
    num_evaluated = 500

    ex =  Explainer(kgs)

    supports_dict = {}
    for max_dist in [1, 2, 3, 4, 5]:
        print(f"\n max_dist: {max_dist}")
        Config.max_dist = max_dist
        
        t1 = time.time()
        supports = []
        for line in tqdm(links[:num_evaluated], desc="computing supports"):
            n1, n2 = line.strip().split("\t")
            num_supports = ex.count_supporting_alignments(n1, n2)
            if num_supports == None:
                continue
            supports.append(num_supports)
            # print(f"num_supports:{num_supports}")
        avg_supports= sum(supports) / len(supports)
        std_supports = (sum([(x - avg_supports) ** 2 for x in supports]) / len(supports)) ** 0.5
        print(f" avg supports: {avg_supports}, std supports: {std_supports}")
        supports_dict[max_dist] = (avg_supports, std_supports)
        print(f"Time elapsed for max_dist {max_dist}: {time.time() - t1}")
    
    print(f"supports: {supports_dict}")

    with open(path[:-6] + "/avg_supports" + f"_thres_{threshold}", "w", encoding="utf8") as f:
        for dist, support in supports_dict.items():
            f.write(f"{dist}\t{support[0]}\t{support[1]}\n")


def avg_confidence(kgs, links, path):
    Config.max_dist = 3
    Config.maximum_common_aligned_neighbors = 3
    Config.conf_thres = 0.0
    num_evaluated = 500

    ex = Explainer(kgs)

    avg_confidences, max_confidences = [], []

    for line in tqdm(links[-num_evaluated:], desc="computing avg confidences"):
        n1, n2 = line.strip().split("\t")
        avg_conf, max_conf = ex.avg_confidence_of_topk_supports(n1, n2, is_positive=True)
        if avg_conf == None:
            continue
        avg_confidences.append(avg_conf)
        max_confidences.append(max_conf)
    
    print(f"avg confs for aligned pairs: {avg_confidences}")
    with open(path[:-6] + "/avg_conf", "w", encoding="utf8") as f:
        for conf in avg_confidences:
            f.write(str(conf) + "\n")
    print(f"max confs for aligned pairs: {max_confidences}")
    with open(path[:-6] + "/max_conf", "w", encoding="utf8") as f:
        for conf in max_confidences:
            f.write(str(conf) + "\n")

    # for negative samples
    avg_confidences, max_confidences = [], []

    for i in tqdm(range(num_evaluated), desc="computing avg confidences"):
        n1 = random.choice(links).strip().split("\t")[0]
        n2 = random.choice(links).strip().split("\t")[1]

        avg_conf, max_conf = ex.avg_confidence_of_topk_supports(n1, n2)
        avg_confidences.append(avg_conf)
        max_confidences.append(max_conf)
    
    print(f"avg confs for misaligned pairs: {avg_confidences}")
    with open(path[:-6] + "/avg_conf_neg", "w", encoding="utf8") as f:
        for conf in avg_confidences:
            f.write(str(conf) + "\n")
    
    print(f"max confs for misaligned pairs: {max_confidences}")
    with open(path[:-6] + "/max_conf_neg", "w", encoding="utf8") as f:
        for conf in max_confidences:
            f.write(str(conf) + "\n")


if __name__ == '__main__':

    print(f"\nExp config:\n {Config()}\n")

    base, _ = os.path.split(os.path.abspath(__file__))
    dataset_name = args.dataset
    dataset_path = os.path.join(os.path.join(base, "data"), dataset_name)

    # print("Construct KGs...")
    kgs = construct_kgs(dataset_dir=dataset_path, name=dataset_name, load_chk=None)

    path = os.path.join("output", f"{dataset_name}", "params")
    # if the path does not exist, then raise an error and tell the user to train the model first
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist. Please train the model first.")

    kgs.util.load_params(path)
    print(f"loaded params from {path}")

    # load links from the ground truth file
    with open(os.path.join(dataset_path, "ent_links"), "r", encoding="utf-8") as f:
        links = f.readlines()

    test(kgs, links)
    # avg_confidence(kgs, links, path)
    # for thres in [0.2, 0.4, 0.6, 0.8]:
    #     num_align_to_max_dist(kgs, links, thres, path)





