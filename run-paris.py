import os
import pdb
import pickle
import argparse
import cProfile
import time
import pstats
from pstats import SortKey


from objects.KG import KG
from objects.KGs import KGs
from ea.model import DualAmn
from config import Config


argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default="fr_en_augmented")
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


if __name__ == '__main__':

    print(f"\nExp config:\n {Config()}\n")

    base, _ = os.path.split(os.path.abspath(__file__))
    dataset_name = args.dataset
    dataset_path = os.path.join(os.path.join(base, "data"), dataset_name)
    
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset exists: {os.path.exists(dataset_path)}")
    print(f"Files in dataset directory:")
    for f in os.listdir(dataset_path):
        print(f"  - {f}")

    print("\nConstruct KGs...")
    kgs = construct_kgs(dataset_dir=dataset_path, name=dataset_name, load_chk=None)

    # num_workers = max(1, os.cpu_count() - 1)
    # num_workers = os.cpu_count() - 2 # 2个核心用于系统
    num_workers = 40
    kgs.set_worker_num(num_workers)
    
    kgs.set_iteration(10)

    
    # 开始性能分析
    print("\nStarting performance profiling...")
    start_time = time.time()
    
    kgs.run()
    
    end_time = time.time()
    
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    