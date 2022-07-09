import sys
# python enjoy_sample_factory.py --algo=APPO --env=polamp_env --experiment=example2
sys.path.insert(0, "sample-factory/")
sys.path.insert(0, "../")
from enjoy_polamp import enjoy
from training_sample_factory import register_custom_components, custom_parse_args

import json
from policy_gradient.curriculum_train import generateDataSet


with open("configs/train_configs.json", 'r') as f:
    train_config = json.load(f)

with open("configs/environment_configs.json", 'r') as f:
    our_env_config = json.load(f)
    # print(our_env_config)

with open("configs/reward_weight_configs.json", 'r') as f:
    reward_config = json.load(f)

with open("configs/car_configs.json", 'r') as f:
    car_config = json.load(f)

'''
with open("../configs/train_configs.json", 'r') as f:
    train_config = json.load(f)

with open("../configs/environment_configs.json", 'r') as f:
    our_env_config = json.load(f)
    # print(our_env_config)

with open("../configs/reward_weight_configs.json", 'r') as f:
    reward_config = json.load(f)

with open("../configs/car_configs.json", 'r') as f:
    car_config = json.load(f)
'''

def main():
    """Script entry point."""
    register_custom_components()
    cfg = custom_parse_args(evaluation=True)
    # print(f"config {cfg}")
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())