import logging
import yaml

import determined as det
from determined.pytorch import deepspeed as det_ds

from model_def import DCGANTrial


def main(config_file: str):
    with det_ds.init() as train_context:
        with open(config_file,"r") as f:
            experiment_config = yaml.load(f,Loader=yaml.SafeLoader)
        trial = DCGANTrial(train_context, experiment_config)
        trainer = det_ds.Trainer(trial, train_context)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    main("mnist.yaml")
