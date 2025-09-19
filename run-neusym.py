import os
import pdb
import pickle
import argparse


from objects.KG import KG
from objects.KGs import KGs
from ea.model import DualAmn, LightEA
from config import Config


argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default="D_W_15K")
argparser.add_argument('--iter', type=int, default=5)
argparser.add_argument('--delta_1', type=float, default=0.9)
argparser.add_argument('--train_ratio', type=float, default=0.1, help="the ratio of training data")
argparser.add_argument('--ea_model', type=str, default="dualamn", help="the EA model to use")
argparser.add_argument('--gpu', type=int, default=1)
args = argparser.parse_args()

Config.train_ratio = args.train_ratio
Config.ea_model = args.ea_model
Config.gpu = args.gpu
if Config.ea_model != "lightea":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(Config.gpu)
# hyperparameters to be tuned:
Config.delta_1 = args.delta_1
iter = args.iter


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


def get_ea_model(data):
    if Config.ea_model == "dualamn":
        return DualAmn(data)
    elif Config.ea_model == "lightea":
        return LightEA(data)


def align(kgs):

    iter = args.iter

    # mutual interactive mode
    for i in range(iter):
        print(f"iter: {i}")
        data, bias = kgs.util.generate_input_for_emb_model()
        if i == 0:
            ea_model = get_ea_model(data)
        else:
            train_pair = data[-2]
            ea_model.reset_data(train_pair)
        new_pairs = ea_model.train(epoch=20)
        kgs.inject_ea_inferred_pairs(new_pairs, bias[0], filter=False, reinject=True)
        kgs.set_iteration(10)
        kgs.run()
    
    kgs.run()
    # self interactive mode
    data, bias = kgs.util.generate_input_for_emb_model()
    ea_model = get_ea_model(data)
    ea_model.fine_tune()


if __name__ == '__main__':

    print(f"\nExp config:\n {Config()}\n")

    base, _ = os.path.split(os.path.abspath(__file__))
    dataset_name = args.dataset
    dataset_path = os.path.join(os.path.join(base, "data"), dataset_name)

    print("Construct KGs...")
    kgs = construct_kgs(dataset_dir=dataset_path, name=dataset_name, load_chk=None)


    num_workers = os.cpu_count()
    kgs.set_worker_num(num_workers)
    
    kgs.set_iteration(20)

    align(kgs=kgs)