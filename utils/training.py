"""
Utility functions for training.
"""
import torch
import wandb

from tqdm import tqdm
from utils.plotting import compute_and_plot_heatmap, SubpoopulationPlotter


def train_one_epoch_scbm(
    train_loader, model, optimizer, mode, metrics, epoch, config, loss_fn, device
):
    """
    Train the Stochastic Concept Bottleneck Model (SCBM) for one epoch.

    This function trains the SCBM for one epoch using the provided training data loader, model, optimizer, and loss function.
    It supports different training modes and updates the model parameters accordingly. The function also computes and logs
    various metrics during the training process.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The SCBM model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        mode (str): The training mode. Supported modes are:
                    - "j": Joint training of the model.
                    - "c": Training the concept head only.
                    - "t": Training the classifier head only.
        metrics (object): An object to track and compute metrics during training.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.

    Returns:
        None

    Notes:
        - Depending on the training mode, certain parts of the model are set to evaluation mode.
        - The function iterates over the training data, performs forward and backward passes, and updates the model parameters.
        - Metrics are computed and logged at the end of each epoch.
    """

    model.train()
    metrics.reset()

    if (
        config.model.training_mode == "sequential"
        or config.model.training_mode == "independent"
    ):
        if mode == "c":
            model.head.eval()
        elif mode == "t":
            model.encoder.eval()

    for k, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    ):
        batch_features, target_true = batch["features"].to(device), batch["labels"].to(
            device
        )
        concepts_true = batch["concepts"].to(device)

        # Forward pass
        output = model(
            batch_features, epoch, c_true=concepts_true
        )
        concepts_mcmc_probs, triang_cov, target_pred_logits = output["c_mcmc_prob"], output["c_triang_cov"], output["y_pred_logits"]
        # Backward pass depends on the training mode of the model
        optimizer.zero_grad()

        # Compute the loss
        loss_dict = loss_fn(
            concepts_mcmc_probs,
            concepts_true,
            target_pred_logits,
            target_true,
            triang_cov,
        )
        target_loss, concepts_loss, prec_loss, total_loss = loss_dict["target_loss"], loss_dict["concepts_loss"], loss_dict["prec_loss"], loss_dict["total_loss"]
        if mode == "j":
            total_loss.backward()
        elif mode == "c":
            (concepts_loss + prec_loss).backward()
        else:
            target_loss.backward()
        optimizer.step()  # perform an update

        # Store predictions
        concepts_pred_probs = concepts_mcmc_probs.mean(-1)
        metrics.update(
            target_loss,
            concepts_loss,
            total_loss,
            target_true,
            target_pred_logits,
            concepts_true,
            concepts_pred_probs,
            prec_loss=prec_loss,
        )

    # Calculate and log metrics
    metrics_dict = metrics.compute()
    wandb.log({f"train/{k}": v for k, v in metrics_dict.items()})
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics.reset()
    return


def train_one_epoch_cbm(
    train_loader, model, optimizer, mode, metrics, epoch, config, loss_fn, device
):
    """
    Train a baseline method for one epoch.

    This function trains the CEM/AR/CBM for one epoch using the provided training data loader, model, optimizer, and loss function.
    It supports different training modes and updates the model parameters accordingly. The function also computes and logs
    various metrics during the training process.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The SCBM model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        mode (str): The training mode. Supported modes are:
                    - "j": Joint training of the model.
                    - "c": Training the concept head only.
                    - "t": Training the classifier head only.
        metrics (object): An object to track and compute metrics during training.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.

    Returns:
        None

    Notes:
        - Depending on the training mode, certain parts of the model are set to evaluation mode.
        - The function iterates over the training data, performs forward and backward passes, and updates the model parameters.
        - Metrics are computed and logged at the end of each epoch.
    """

    model.train()
    metrics.reset()

    if config.model.training_mode in ("sequential", "independent"):
        if mode == "c":
            model.head.eval()
        elif mode == "t":
            model.encoder.eval()

    for k, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    ):
        batch_features, target_true = batch["features"].to(device), batch["labels"].to(
            device
        )
        concepts_true = batch["concepts"].to(device)

        # Forward pass
        if config.model.training_mode == "independent" and mode == "t":
            output = model(batch_features, epoch, concepts_true)
        elif config.model.concept_learning == "autoregressive" and mode == "c":
            output = model(batch_features, epoch, concepts_train_ar=concepts_true)
        else:
            output = model(batch_features, epoch)
        concepts_pred_probs, target_pred_logits = output["c_prob"], output["y_pred_logits"]
        
        # Backward pass depends on the training mode of the model
        optimizer.zero_grad()
        # Compute the loss
        target_loss, concepts_loss, total_loss = loss_fn(
            concepts_pred_probs, concepts_true, target_pred_logits, target_true
        )

        if mode == "j":
            total_loss.backward()
        elif mode == "c":
            concepts_loss.backward()
        else:
            target_loss.backward()
        optimizer.step()  # perform an update

        # Store predictions
        metrics.update(
            target_loss,
            concepts_loss,
            total_loss,
            target_true,
            target_pred_logits,
            concepts_true,
            concepts_pred_probs,
        )

    # Calculate and log metrics
    metrics_dict = metrics.compute()
    wandb.log({f"train/{k}": v for k, v in metrics_dict.items()})
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics.reset()
    return


def validate_one_epoch_scbm(
    loader,
    model,
    metrics,
    epoch,
    config,
    loss_fn,
    device,
    test=False,
    concept_names_graph=None,
    population_metrics=None,
):
    """
    Validate the Stochastic Concept Bottleneck Model (SCBM) for one epoch.

    This function evaluates the SCBM for one epoch using the provided data loader, model, and loss function.
    It computes and logs various metrics during the validation process. It also generates
    and plots a heatmap of the learned concept correlation matrix on the final test set.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation or test data.
        model (torch.nn.Module): The SCBM model to be validated.
        metrics (object): An object to track and compute metrics during validation.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and validation settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.
        test (bool, optional): Flag indicating whether this is the final evaluation on the test set. Default is False.
        concept_names_graph (list, optional): List of concept names for plotting the heatmap.
                                              Default is None for which range(n_concepts) is used.

    Returns:
        None

    Notes:
        - The function sets the model to evaluation mode and disables gradient computation.
        - It iterates over the validation data, performs forward passes, and computes the losses.
        - Metrics are computed and logged at the end of the validation epoch.
        - During testing, the function generates and plots a heatmap of the concept correlation matrix.
    """
    model.eval()
    with torch.no_grad():

        for k, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", position=0, leave=True)
        ):
            batch_features, target_true = batch["features"].to(device), batch[
                "labels"
            ].to(device)
            concepts_true = batch["concepts"].to(device)

            output = model(
                batch_features, epoch, validation=True, c_true=concepts_true
            )
            concepts_mcmc_probs, triang_cov, target_pred_logits = output["c_mcmc_prob"], output["c_triang_cov"], output["y_pred_logits"]
            # Compute covariance matrix of concepts
            cov = torch.matmul(triang_cov, torch.transpose(triang_cov, dim0=1, dim1=2))

            if test and k % (len(loader) // 10) == 0:
                try:
                    corr = (cov[0] / cov[0].diag().sqrt()).transpose(
                        dim0=0, dim1=1
                    ) / cov[0].diag().sqrt()
                    matrix = corr.cpu().numpy()

                    compute_and_plot_heatmap(
                        matrix, concepts_true, concept_names_graph, config
                    )

                except:
                    pass

            losses_dict = loss_fn(
                concepts_mcmc_probs,
                concepts_true,
                target_pred_logits,
                target_true,
                triang_cov,
            )
            target_loss, concepts_loss, prec_loss, total_loss = losses_dict["target_loss"], losses_dict["concepts_loss"], losses_dict["prec_loss"], losses_dict["total_loss"]

            # Store predictions
            concepts_pred_probs = concepts_mcmc_probs.mean(-1)
            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                target_pred_logits,
                concepts_true,
                concepts_pred_probs,
                prec_loss=prec_loss,
            )
            if population_metrics is not None:
                population_metrics.update(
                    target_loss=losses_dict["per_sample_target_loss"],
                    concepts_loss=losses_dict["per_concept_loss"],
                    y_true=target_true,
                    y_pred_logits=target_pred_logits,
                    c_true=concepts_true,
                    c_pred_probs=concepts_pred_probs,
                    c_mcmc_pred_probs=concepts_mcmc_probs,
                    cov_mat=cov,
                )
    # Calculate and log metrics
    metrics_dict = metrics.compute(validation=True, config=config)
    if population_metrics is not None:
        # Compute and plot subpopulation statistics
        population_metrics_dict = population_metrics.compute(validation=True)

        subpop_plotter = SubpoopulationPlotter(
            population_metrics=population_metrics_dict,
            subpopulations_str2idx=loader.dataset.subpopulations_dict,
            log_scale=True,
            save_path=config.experiment_dir if not config.logging.debug_mode else "",
        )

        subpop_plotter.plot(plot_uncertainty=True)

    if not test:
        wandb.log({f"validation/{k}": v for k, v in metrics_dict.items()})
        prints = f"Epoch {epoch}, Validation: "
    else:
        wandb.log({f"test/{k}": v for k, v in metrics_dict.items()})
        prints = f"Test: "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()
    metrics.reset()
    if population_metrics is not None:
        population_metrics.reset()
    return


def validate_one_epoch_cbm(
    loader,
    model,
    metrics,
    epoch,
    config,
    loss_fn,
    device,
    test=False,
    concept_names_graph=None,
    population_metrics=None,
):
    """
    Validate a baseline method for one epoch.

    This function evaluates the CEM/AR/CBM for one epoch using the provided data loader, model, and loss function.
    It computes and logs various metrics during the validation process.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation or test data.
        model (torch.nn.Module): The model to be validated.
        metrics (object): An object to track and compute metrics during validation.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and validation settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.
        test (bool, optional): Flag indicating whether this is the final evaluation on the test set. Default is False.

    Returns:
        None

    Notes:
        - The function sets the model to evaluation mode and disables gradient computation.
        - It iterates over the validation data, performs forward passes, and computes the losses.
        - Metrics are computed and logged at the end of the validation epoch.
    """
    model.eval()

    with torch.no_grad():
        for k, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", position=0, leave=True)
        ):
            batch_features, target_true = batch["features"].to(device), batch[
                "labels"
            ].to(device)
            concepts_true = batch["concepts"].to(device)

            output = model(
                batch_features, epoch, validation=True
            )
            concepts_pred_probs, target_pred_logits = output["c_prob"], output["y_pred_logits"]
            
            if config.model.concept_learning == "autoregressive":
                concepts_pred_probs = torch.mean(
                    concepts_pred_probs, dim=-1
                )  # Calculating the metrics on the average probabilities from MCMC

            losses_dict = loss_fn(
                concepts_pred_probs, concepts_true, target_pred_logits, target_true
            )
            target_loss, concepts_loss, total_loss = losses_dict["target_loss"], losses_dict["concepts_loss"], losses_dict["total_loss"]

            # Store predictions
            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                target_pred_logits,
                concepts_true,
                concepts_pred_probs,
            )

            if population_metrics is not None:
                population_metrics.update(
                    target_loss=losses_dict["per_sample_target_loss"],
                    concepts_loss=losses_dict["per_concept_loss"],
                    y_true=target_true,
                    y_pred_logits=target_pred_logits,
                    c_true=concepts_true,
                    c_pred_probs=concepts_pred_probs,
                    # c_mcmc_pred_probs=concepts_mcmc_probs,
                    # cov_mat=cov,
                )

    # Calculate and log metrics
    metrics_dict = metrics.compute(validation=True, config=config)
    if population_metrics is not None:
        population_metrics_dict = population_metrics.compute(validation=True)
        plot_subpopulations_stats(
            population_metrics=population_metrics_dict,
            save_path=config.logging.save_dir,
            plot_uncertainty=False
        )
    if not test:
        wandb.log({f"validation/{k}": v for k, v in metrics_dict.items()})
        prints = f"Epoch {epoch}, Validation: "
    else:
        wandb.log({f"test/{k}": v for k, v in metrics_dict.items()})
        prints = f"Test: "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()
    metrics.reset()
    if population_metrics is not None:
        population_metrics.reset()
    return


def create_optimizer(config, model):
    """
    Parse the configuration file and return a optimizer object to update the model parameters.
    """
    assert config.optimizer in [
        "sgd",
        "adam",
    ], "Only SGD and Adam optimizers are available!"

    optim_params = [
        {
            "params": filter(lambda p: p.requires_grad, model.parameters()),
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    ]

    if config.optimizer == "sgd":
        return torch.optim.SGD(optim_params)
    elif config.optimizer == "adam":
        return torch.optim.Adam(optim_params)


def freeze_module(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    m.train()
    for param in m.parameters():
        param.requires_grad = True
