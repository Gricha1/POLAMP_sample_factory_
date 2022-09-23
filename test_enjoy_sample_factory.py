import sys
import json
sys.path.insert(0, "sample-factory/")
sys.path.insert(0, "../")
#from enjoy_polamp import enjoy
from test_enjoy_polamp import enjoy
#from training_sample_factory import register_custom_components, custom_parse_args
from test_training_sample_factory import register_custom_components, custom_parse_args

with open("configs/environment_configs.json", 'r') as f:
    our_env_config = json.load(f)

with open("configs/reward_weight_configs.json", 'r') as f:
    reward_config = json.load(f)

with open("configs/car_configs.json", 'r') as f:
    car_config = json.load(f)

def main():
    """Script entry point."""
    register_custom_components()
    cfg = custom_parse_args(evaluation=True)    
    status = enjoy(cfg)

    return status

if __name__ == '__main__':
    sys.exit(main())