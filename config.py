
class Config(object):
    # for data processing
    simplify_url = True
    remove_ID_attr = False

    # for dataset split
    train_ratio = 0.1

    # for probabilistic reasoning
    delta_0 = 0.5
    delta_1 = 0.9

    # for ea base model
    ea_model = "dualamn" # select from dualamn, lightea

    ## print control during exps
    print_during_exp = {
        'paris': False
    }

    @classmethod
    def __repr__(cls):
        return '\n'.join(f'{k}: {v}' for k, v in cls.__dict__.items() if not k.startswith('__') and k != 'gpt_api_key')