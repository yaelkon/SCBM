import hydra
import os
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from datasets.mnist_sum import from_concept_to_population_idx
from models.models import create_model
from models.losses import create_loss
from utils.data import get_data, get_concept_groups
from utils.metrics import Custom_Metrics, Population_Metrics
from utils.utils import reset_random_seeds
from utils.training import (
        validate_one_epoch_cbm,
        validate_one_epoch_scbm,
)


def test(config: DictConfig):
    
    gen = reset_random_seeds(config.seed)
    
    # Setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # Additional info when using cuda
    if device.type == "cuda":
        print("Using", torch.cuda.get_device_name(0))
    else:
        print("No GPU available")


    # Wandb
    os.environ["WANDB_CACHE_DIR"] = os.path.join(
        Path(__file__).absolute().parent, "wandb", ".cache", "wandb"
    )  # S.t. on slurm, artifacts are logged to the right place
    print("Cache dir:", os.environ["WANDB_CACHE_DIR"])
    wandb.init(
        project=config.logging.project,
        reinit=True,
        # entity=config.logging.entity,
        config=OmegaConf.to_container(config, resolve=True),
        mode=config.logging.mode,
        name=config.logging.experiment_name,
        tags=[config.model.tag],
    )
    if config.logging.mode in ["online", "disabled"]:
        wandb.run.name = wandb.run.name.split("-")[-1] + "-" + config.experiment_name
    elif config.logging.mode == "offline":
        wandb.run.name = config.experiment_name
    else:
        raise ValueError("wandb needs to be set to online, offline or disabled.")

    # ---------------------------------
    #       Prepare data and model
    # ---------------------------------
    _, _, test_loader = get_data(
        config,
        config.data,
        gen,
    )

    # Get concept names for plotting
    concept_names_graph = get_concept_groups(config.data)

    # Initialize model and training objects
    model = create_model(config)
    model = model.to(device)
    model.load_state_dict(torch.load(config.model.pretrained_model_path, weights_only=False))

    loss_fn = create_loss(config)

    metrics = Custom_Metrics(config.data.num_concepts, device).to(device)

    population_metrics = None
    if config.data.get("subpopulations", None) is not None:
        population_metrics = Population_Metrics(
            n_concepts=config.data.num_concepts,
            n_populations=len(config.data.subpopulations),
            device=device,
        ).to(device)

    if config.model.model == "cbm":
        validate_one_epoch = validate_one_epoch_cbm

    else:
        validate_one_epoch = validate_one_epoch_scbm

    validate_one_epoch(
        test_loader,
        model,
        metrics,
        0,
        config,
        loss_fn,
        device,
        test=True,
        concept_names_graph=concept_names_graph,
        population_metrics=population_metrics,
    )
    

@hydra.main(version_base=None, config_path="configs", config_name="config_test")
def main(config: DictConfig):
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    test(config)


if __name__ == "__main__":
    main()
