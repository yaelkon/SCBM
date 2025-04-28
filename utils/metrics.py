"""
Utility functions for computing metrics.
"""
from typing import Optional

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score, jaccard_score
from torchmetrics import AveragePrecision, CalibrationError, Metric

from utils.utils import torch_to_numpy

class Population_Metrics(Metric):

    def __init__(
            self, 
            n_populations, 
            n_concepts,
            device,
            ):
        
        super().__init__()
        self.n_concepts = n_concepts
        self.n_populations = n_populations
        # self.population_func = populationIdxTrans_func
        # self.dataset_name = dataset_name

        self.add_state('n_samples_per_population', default=torch.tensor(
            np.zeros(n_populations), device=device,
        ))

        self.add_state('concept_losses_per_population', default=torch.tensor(
            np.zeros((n_populations, n_concepts)), device=device,
        ))

        self.add_state('target_losses_per_population', default=torch.tensor(
            np.zeros((n_populations,)), device=device,
        ))

        self.add_state('total_losses_per_population', default=torch.tensor(
            np.zeros((n_populations,)), device=device,
        ))

        self.add_state("uncertainty_per_population", default=torch.tensor(
            np.zeros((n_populations, n_concepts)), device=device,
        ))

        self.add_state("std_per_population", default=torch.tensor(
            np.zeros((n_populations, n_concepts)), device=device,
        ))

        for i in range(n_populations):
            self.add_state(f"y_true_per_population_{i}", default=[])
            self.add_state(f"y_probs_per_population_{i}", default=[])
            self.add_state(f"c_true_per_population_{i}", default=[])
            self.add_state(f"c_probs_per_population_{i}", default=[])
        

    def update(
        self,
        target_loss: torch.Tensor,
        concepts_loss: torch.Tensor,
        # total_loss: torch.Tensor,
        y_true: torch.Tensor,
        y_pred_logits: torch.Tensor,
        c_true: torch.Tensor,
        c_pred_probs: torch.Tensor,
        subpopulation_indices: torch.Tensor,
        c_mcmc_pred_probs: Optional[torch.Tensor] = None,
        cov_mat: Optional[torch.Tensor] = None,
    ): 
        # population_indices = self.population_func(c_true)
        unique_populations, n_samples_per_populations = subpopulation_indices.unique(return_counts=True)
        self.n_samples_per_population[unique_populations] += n_samples_per_populations.to(self.n_samples_per_population.device)
        
        y_probs = torch.softmax(y_pred_logits, dim=1)
        concepts_std = None
        if cov_mat is not None:
            concepts_std = torch.sqrt(torch.diagonal(cov_mat, dim1=1, dim2=2))
        for p in range(self.n_populations):
            if p not in unique_populations:
                continue

            mask = subpopulation_indices == p
            masked_c_loss = concepts_loss[mask].sum(dim=0)
            masked_t_loss = target_loss[mask].sum()
            self.concept_losses_per_population[p] += masked_c_loss
            self.target_losses_per_population[p] += masked_t_loss
            self.total_losses_per_population[p] += (masked_t_loss + (masked_c_loss.sum() / self.n_concepts))

            if c_mcmc_pred_probs is not None:
                concept_uncertainty = c_mcmc_pred_probs[mask].std(dim=-1).sum(0)
                self.uncertainty_per_population[p] += concept_uncertainty
            if concepts_std is not None:
                self.std_per_population[p] += concepts_std[mask].sum(dim=0)

            getattr(self, f"y_true_per_population_{p}").extend(y_true[mask].tolist())
            getattr(self, f"y_probs_per_population_{p}").extend(y_probs[mask].tolist())
            getattr(self, f"c_true_per_population_{p}").extend(c_true[mask].tolist())
            getattr(self, f"c_probs_per_population_{p}").extend(c_pred_probs[mask].tolist())

    def compute(self, validation=True):
        metrics = {}
        per_population_task_accuracy = []
        per_population_concept_accuracy = []
        per_population_complete_concept_accuracy = []
        per_population_concept_uncertainty = []
        per_population_concept_std = []
        per_population_target_loss = []
        per_population_concept_loss = []
        per_population_total_loss = []

        for p in range(self.n_populations):
            if self.n_samples_per_population[p] == 0:
                continue

            y_true = np.array(getattr(self, f"y_true_per_population_{p}"))
            y_probs = np.array(getattr(self, f"y_probs_per_population_{p}"))
            c_true = np.array(getattr(self, f"c_true_per_population_{p}"))
            c_probs = np.array(getattr(self, f"c_probs_per_population_{p}"))

            y_pred = np.argmax(y_probs, axis=-1)
            c_pred = c_probs > 0.5

            target_acc = (y_true == y_pred).sum() / self.n_samples_per_population[p]
            target_acc = torch_to_numpy(target_acc)
            concept_acc = (c_true == c_pred).sum(0) / int(self.n_samples_per_population[p])
            complete_concept_acc = ((c_true == c_pred).sum(1) == self.n_concepts).sum() / self.n_samples_per_population[p]
            complete_concept_acc = torch_to_numpy(complete_concept_acc)

            per_population_task_accuracy.append(target_acc)
            per_population_concept_accuracy.append(concept_acc)
            per_population_complete_concept_accuracy.append(complete_concept_acc)

            target_loss = torch_to_numpy(self.target_losses_per_population[p] / self.n_samples_per_population[p])
            concept_loss = torch_to_numpy(self.concept_losses_per_population[p] / self.n_samples_per_population[p])
            total_loss = torch_to_numpy(self.total_losses_per_population[p] / self.n_samples_per_population[p])

            per_population_target_loss.append(target_loss)
            per_population_concept_loss.append(concept_loss)
            per_population_total_loss.append(total_loss)

            if self.uncertainty_per_population.sum() != 0:
                concept_uncertainty = torch_to_numpy(self.uncertainty_per_population[p] / self.n_samples_per_population[p])
                per_population_concept_uncertainty.append(concept_uncertainty)

            if self.std_per_population.sum() != 0:
                concept_std = torch_to_numpy(self.std_per_population[p] / self.n_samples_per_population[p])
                per_population_concept_std.append(concept_std)
        
        metrics["target_loss"] = np.stack(per_population_target_loss)
        metrics["concept_loss"] = np.stack(per_population_concept_loss)
        metrics["total_loss"] = np.stack(per_population_total_loss)
        metrics["task_accuracy"] = np.stack(per_population_task_accuracy)
        metrics["concept_accuracy"] = np.stack(per_population_concept_accuracy)
        metrics["complete_concept_accuracy"] = np.stack(per_population_complete_concept_accuracy)
        metrics["n_samples_per_population"] = torch_to_numpy(self.n_samples_per_population)
        if self.uncertainty_per_population.sum() != 0:
            metrics["concept_uncertainty"] = np.stack(per_population_concept_uncertainty)
        if self.std_per_population.sum() != 0:
            metrics["concept_std"] = np.stack(per_population_concept_std)

        return metrics

class Custom_Metrics(Metric):
    """
    Custom metrics class for tracking and computing various metrics during training and validation.

    This class extends the PyTorch Metric class and provides methods to update and compute metrics such as
    target loss, concept loss, total loss, accuracy, and Jaccard index for both target and concepts.
    It is being updated for each batch. At the end of each epoch, the compute function is called to compute
    the final metrics and return them as a dictionary.

    Args:
        n_concepts (int): The number of concepts in the model.
        device (torch.device): The device to run the computations on.

    Attributes:
        n_concepts (int): The number of concepts in the model.
        target_loss (torch.Tensor): The accumulated target loss.
        concepts_loss (torch.Tensor): The accumulated concepts loss.
        total_loss (torch.Tensor): The accumulated total loss.
        y_true (list): List of true target labels.
        y_pred_logits (list): List of predicted target logits.
        c_true (list): List of true concept labels.
        c_pred_probs (list): List of predicted concept probabilities.
        batch_features (list): List of batch features.
        cov_norm (torch.Tensor): The accumulated covariance norm.
        n_samples (torch.Tensor): The number of samples processed.
        prec_loss (torch.Tensor): The accumulated precision loss.
    """

    def __init__(self, n_concepts, device):
        super().__init__()
        self.n_concepts = n_concepts
        self.add_state("target_loss", default=torch.tensor(0.0, device=device))
        self.add_state("concepts_loss", default=torch.tensor(0.0, device=device))
        self.add_state("total_loss", default=torch.tensor(0.0, device=device))
        self.add_state("y_true", default=[])
        self.add_state("y_pred_logits", default=[])
        self.add_state("c_true", default=[])
        (
            self.add_state("c_pred_probs", default=[]),
            self.add_state("concepts_input", default=[]),
        ),
        self.add_state("batch_features", default=[])
        self.add_state("cov_norm", default=torch.tensor(0.0, device=device))
        self.add_state(
            "n_samples", default=torch.tensor(0, dtype=torch.int, device=device)
        )
        self.add_state("prec_loss", default=torch.tensor(0.0, device=device))

    def update(
        self,
        target_loss: torch.Tensor,
        concepts_loss: torch.Tensor,
        total_loss: torch.Tensor,
        y_true: torch.Tensor,
        y_pred_logits: torch.Tensor,
        c_true: torch.Tensor,
        c_pred_probs: torch.Tensor,
        cov_norm: torch.Tensor = None,
        prec_loss: torch.Tensor = None,
    ):
        assert c_true.shape == c_pred_probs.shape

        n_samples = y_true.size(0)
        # self.ce = nn.CrossEntropyLoss()
        # self.bce = nn.BCELoss()
        self.n_samples += n_samples
        self.target_loss += target_loss * n_samples
        self.concepts_loss += concepts_loss * n_samples
        self.total_loss += total_loss * n_samples
        self.y_true.append(y_true)
        self.y_pred_logits.append(y_pred_logits.detach())
        self.c_true.append(c_true)
        self.c_pred_probs.append(c_pred_probs.detach())
        if cov_norm:
            self.cov_norm += cov_norm * n_samples
        if prec_loss:
            self.prec_loss += prec_loss * n_samples

    def compute(self, validation=False, config=None):
        y_true = torch.cat(self.y_true, dim=0).cpu()
        c_true = torch.cat(self.c_true, dim=0).cpu()
        c_pred_probs = torch.cat(self.c_pred_probs, dim=0).cpu()
        y_pred_logits = torch.cat(self.y_pred_logits, dim=0).cpu()
        c_true = c_true.cpu().numpy()
        c_pred_probs = c_pred_probs.cpu().numpy()
        c_pred = c_pred_probs > 0.5
        if y_pred_logits.size(1) == 1:
            y_pred_probs = nn.Sigmoid()(y_pred_logits.squeeze())
            y_pred = y_pred_probs > 0.5
        else:
            y_pred_probs = nn.Softmax(dim=1)(y_pred_logits)
            y_pred = y_pred_logits.argmax(dim=-1)

        target_acc = (y_true == y_pred).sum() / self.n_samples
        concept_acc = (c_true == c_pred).sum() / (self.n_samples * self.n_concepts)
        complete_concept_acc = (
            (c_true == c_pred).sum(1) == self.n_concepts
        ).sum() / self.n_samples
        target_jaccard = jaccard_score(y_true, y_pred, average="micro")
        concept_jaccard = jaccard_score(c_true, c_pred, average="micro")
        metrics = dict(
            {
                "target_loss": self.target_loss / self.n_samples,
                "prec_loss": self.prec_loss / self.n_samples,
                "concepts_loss": self.concepts_loss / self.n_samples,
                "total_loss": self.total_loss / self.n_samples,
                "y_accuracy": target_acc,
                "c_accuracy": concept_acc,
                "complete_c_accuracy": complete_concept_acc,
                "target_jaccard": target_jaccard,
                "concept_jaccard": concept_jaccard,
            }
        )

        if self.cov_norm != 0:
            metrics = metrics | {"covariance_norm": self.cov_norm / self.n_samples}

        if validation is True:
            c_pred_probs_list = []
            for j in range(self.n_concepts):
                c_pred_probs_list.append(
                    np.hstack(
                        (
                            np.expand_dims(1 - c_pred_probs[:, j], 1),
                            np.expand_dims(c_pred_probs[:, j], 1),
                        )
                    )
                )

            y_metrics = calc_target_metrics(
                y_true.numpy(), y_pred_probs.numpy(), config.data
            )
            c_metrics, c_metrics_per_concept = calc_concept_metrics(
                c_true, c_pred_probs_list, config.data
            )
            metrics = (
                metrics
                | {f"y_{k}": v for k, v in y_metrics.items()}
                | {f"c_{k}": v for k, v in c_metrics.items()}
            )  # | c_metrics_per_concept # Update dict

        return metrics
    

def _roc_auc_score_with_missing(labels, scores):
    # Computes OVR macro-averaged AUROC under missing classes
    aurocs = np.zeros((scores.shape[1],))
    weights = np.zeros((scores.shape[1],))
    for c in range(scores.shape[1]):
        if len(labels[labels == c]) > 0:
            labels_tmp = (labels == c) * 1.0
            aurocs[c] = roc_auc_score(
                labels_tmp, scores[:, c], average="weighted", multi_class="ovr"
            )
            weights[c] = len(labels[labels == c])
        else:
            aurocs[c] = np.NaN
            weights[c] = np.NaN

    # Computing weighted average
    mask = ~np.isnan(aurocs)
    weighted_sum = np.sum(aurocs[mask] * weights[mask])
    average = weighted_sum / len(labels)
    # Regular "macro"
    # average = np.nanmean(aurocs)
    return average


def calc_target_metrics(ys, scores_pred, config, n_decimals=4, n_bins_cal=10):
    """

    :param ys:
    :param scores_pred:
    :param config:
    :return:
    """
    # AUROC
    if config.num_classes == 2:
        auroc = roc_auc_score(ys, scores_pred)
    elif config.num_classes > 2:
        auroc = _roc_auc_score_with_missing(ys, scores_pred)

    # AUPR
    aupr = 0.0
    if config.num_classes == 2:
        aupr = average_precision_score(ys, scores_pred)
    elif config.num_classes > 2:
        ap = AveragePrecision(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        aupr = float(
            ap(torch.tensor(scores_pred), torch.tensor(ys.squeeze()).type(torch.int64))
            .cpu()
            .numpy()
        )

    # Brier score
    if config.num_classes == 2:
        brier = brier_score(ys, np.squeeze(scores_pred))
    else:
        brier = brier_score(ys, scores_pred)

    # ECE
    if config.num_classes == 2:
        ece_fct = CalibrationError(task="binary", n_bins=n_bins_cal, norm="l1")
        tl_ece_fct = CalibrationError(task="binary", n_bins=n_bins_cal, norm="l2")
        ece = float(
            ece_fct(
                torch.tensor(np.squeeze(scores_pred)),
                torch.tensor(ys.squeeze()).type(torch.int64),
            )
            .cpu()
            .numpy()
        )
        tl_ece = float(
            tl_ece_fct(
                torch.tensor(np.squeeze(scores_pred)),
                torch.tensor(ys.squeeze()).type(torch.int64),
            )
            .cpu()
            .numpy()
        )

    else:
        ece_fct = CalibrationError(
            task="multiclass",
            n_bins=n_bins_cal,
            norm="l1",
            num_classes=config.num_classes,
        )
        tl_ece_fct = CalibrationError(
            task="multiclass",
            n_bins=n_bins_cal,
            norm="l2",
            num_classes=config.num_classes,
        )
        ece = float(
            ece_fct(
                torch.tensor(scores_pred), torch.tensor(ys.squeeze()).type(torch.int64)
            )
            .cpu()
            .numpy()
        )
        tl_ece = float(
            tl_ece_fct(
                torch.tensor(scores_pred), torch.tensor(ys.squeeze()).type(torch.int64)
            )
            .cpu()
            .numpy()
        )

    return {
        "AUROC": np.round(auroc, n_decimals),
        "AUPR": np.round(aupr, n_decimals),
        "Brier": np.round(brier, n_decimals),
        "ECE": np.round(ece, n_decimals),
        "TL-ECE": np.round(tl_ece, n_decimals),
    }


def calc_concept_metrics(cs, concepts_pred_probs, config, n_decimals=4, n_bins_cal=10):
    num_concepts = cs.shape[1]

    metrics_per_concept = []

    for j in range(num_concepts):
        # AUROC
        auroc = 0.0
        if len(np.unique(cs[:, j])) == 2:
            auroc = roc_auc_score(
                cs[:, j],
                concepts_pred_probs[j][:, 1],
                average="macro",
                multi_class="ovr",
            )
        elif len(np.unique(cs[:, j])) > 2:
            auroc = roc_auc_score(
                cs[:, j], concepts_pred_probs[j], average="macro", multi_class="ovr"
            )

        # AUPR
        aupr = 0.0
        if len(np.unique(cs[:, j])) == 2:
            aupr = average_precision_score(
                cs[:, j], concepts_pred_probs[j][:, 1], average="macro"
            )
        elif len(np.unique(cs[:, j])) > 2:
            ap = AveragePrecision(
                task="multiclass", num_classes=config.num_classes, average="macro"
            )
            aupr = float(
                ap(torch.tensor(concepts_pred_probs[j]), torch.tensor(cs[:, j]))
                .cpu()
                .numpy()
            )

        # Brier score
        if len(np.unique(cs[:, j])) == 2:
            brier = brier_score(cs[:, j], concepts_pred_probs[j][:, 1])
        else:
            brier = brier_score(cs[:, j], concepts_pred_probs[j])

        # ECE
        ece_fct = CalibrationError(task="binary", n_bins=n_bins_cal, norm="l1")
        tl_ece_fct = CalibrationError(task="binary", n_bins=n_bins_cal, norm="l2")
        if len(concepts_pred_probs[j].shape) == 1:
            ece = float(
                ece_fct(
                    torch.tensor(concepts_pred_probs[j]),
                    torch.tensor(cs[:, j].squeeze()).type(torch.int64),
                )
                .cpu()
                .numpy()
            )
            tl_ece = float(
                tl_ece_fct(
                    torch.tensor(concepts_pred_probs[j]),
                    torch.tensor(cs[:, j].squeeze()).type(torch.int64),
                )
                .cpu()
                .numpy()
            )

        else:
            ece = float(
                ece_fct(
                    torch.tensor(concepts_pred_probs[j][:, 1]),
                    torch.tensor(cs[:, j].squeeze()).type(torch.int64),
                )
                .cpu()
                .numpy()
            )
            tl_ece = float(
                tl_ece_fct(
                    torch.tensor(concepts_pred_probs[j][:, 1]),
                    torch.tensor(cs[:, j].squeeze()).type(torch.int64),
                )
                .cpu()
                .numpy()
            )

        metrics_per_concept.append(
            {
                "AUROC": np.round(auroc, n_decimals),
                "AUPR": np.round(aupr, n_decimals),
                "Brier": np.round(brier, n_decimals),
                "ECE": np.round(ece, n_decimals),
                "TL-ECE": np.round(tl_ece, n_decimals),
            }
        )

    auroc = 0.0
    aupr = 0.0
    brier = 0.0
    ece = 0.0
    for j in range(num_concepts):
        auroc += metrics_per_concept[j]["AUROC"]
        aupr += metrics_per_concept[j]["AUPR"]
        brier += metrics_per_concept[j]["Brier"]
        ece += metrics_per_concept[j]["ECE"]
        tl_ece += metrics_per_concept[j]["TL-ECE"]
    auroc /= num_concepts
    aupr /= num_concepts
    brier /= num_concepts
    ece /= num_concepts
    tl_ece /= num_concepts
    metrics_overall = {
        "AUROC": np.round(auroc, n_decimals),
        "AUPR": np.round(aupr, n_decimals),
        "Brier": np.round(brier, n_decimals),
        "ECE": np.round(ece, n_decimals),
        "TL-ECE": np.round(tl_ece, n_decimals),
    }

    return metrics_overall, metrics_per_concept


def brier_score(y_true, y_prob):
    # NOTE:
    # - for multiclass, @y_true must be of dimensionality (n_samples, ) and @y_prob must be (n_samples, n_classes)
    # - for binary, @y_true must be of dimensionality (n_samples, ) and @y_prob must be (n_samples, )

    if len(y_prob.shape) == 2:
        # NOTE: we use the original definition by Brier for categorical variables
        # See the original paper by Brier https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
        sc = 0
        for j in range(y_prob.shape[1]):
            # sc += np.sum(((y_true == j) * 1. - y_prob[j])**2)
            # Correction to multiclass
            sc += np.sum((np.squeeze((y_true == j) * 1.0) - y_prob[:, j]) ** 2)
        sc /= y_true.shape[0]
        return sc
    elif len(y_prob.shape) == 1:
        return np.mean((y_prob - y_true) ** 2)
