import numpy as np
import torch

from os.path import join
from torch.utils.data import Dataset
from torchvision import transforms
from omegaconf import DictConfig


def get_MNIST_SUM_dataset(config: DictConfig):
    datapath = config.data_path + "mnist/mnist_sum"
    image_datasets = {
        "train": MNIST_SUM_Dataset(
            root=datapath,
            train=True,
            file_name=config.get("train_file_name"),
            subpopulations=config.get("subpopulations")
        ),
        "val": MNIST_SUM_Dataset(
            root=datapath,
            train=False,
            file_name=config.get("val_file_name"),
            subpopulations=config.get("subpopulations")
        ),
        "test": MNIST_SUM_Dataset(
            root=datapath,
            train=False,
            file_name=config.get("test_file_name"),
            subpopulations=config.get("subpopulations")
        ),
    }

    return image_datasets["train"], image_datasets["val"], image_datasets["test"]


def from_concept_to_population_idx(concepts: torch.Tensor):
    """
    Convert a tesnsor of concepts to a tensor of population indices.
    """
    n_concepts = concepts.size()[-1]
    n_half_concepts = n_concepts // 2
    populations_rep = []
    for i in range(n_half_concepts):
        for j in range(n_half_concepts):
            pop = torch.zeros(n_concepts, device=concepts.device)
            pop[i] = 1
            pop[n_half_concepts + j] = 1
            populations_rep.append(pop)

    populations_rep = torch.stack(populations_rep, dim=0)
    populations_indices = []
    for c in concepts:
        for i, p in enumerate(populations_rep):
            if torch.equal(c, p):
                populations_indices.append(i)
                break

            if i == len(populations_rep) - 1:
                assert False, f"Population not found for concept {c} in populations_rep"
    
    populations_indices = torch.tensor(populations_indices)
    return populations_indices


class MNIST_SUM_Dataset(Dataset):
    def __init__(self, root, train=True, file_name=None, subpopulations=None):

        self.stage = "train" if train else "test"
        self.root = root
        self.file_name = file_name

        if file_name is None:
            file_name = f"{self.stage}_sum_mnist_concepts.npz"

        data = np.load(join(self.root, file_name))
        self.imgs = data["imgs"]
        self.labels = data["labels"]
        self.concepts = data["concepts"]
        self.attributes = None
        if "attributes" in data:
            self.attributes = data["attributes"]

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])

        self.subpopulations_dict = self._create_subpopulations(subpopulations)
    
    def _create_subpopulations(self, subpopulations: List[tuple]) -> dict:
        """
        Create subpopulations based on the provided subpopulations dictionary.
        """
        if subpopulations is None:
            return None

        subpopulations_dict = {}
        for i, s in enumerate(subpopulations):
            s = tuple(s)
            subpopulations_dict[s] = i

        return subpopulations_dict
    
    def _get_subpopulation_idx(self, concepts: torch.Tensor, attributes: List[str]) -> int:
        """
        Get the subpopulation index based on the concept and attributes.
        """
        if self.subpopulations_dict is None:
            return 0

        # Extract subgroup representation from concept
        one_side_length = len(concepts) // 2
        left_side_digit, right_side_digit = np.where(concepts[:one_side_length] == 1)[0][0], np.where(concepts[one_side_length:] == 1)[0][0]
        digits_tuple = (left_side_digit, right_side_digit)

        # Create a tuple of the concept and attributes
        for a in attributes:
            if a == 'none':
                continue
            digits_tuple += (a,)
        
        subpop_idx = self.subpopulations_dict[digits_tuple]
        return subpop_idx
    
    def __getitem__(self, idx):
        # Load the MNIST dataset from the specified directory
        try:
            x = self.transforms(self.imgs[idx])
            y = self.labels[idx]
            c = self.concepts[idx]
        except Exception as e:
            print(f"Error loading data from datafile {self.root + self.file_name} at index {idx}: {e}")
            raise 
        y = self.labels[idx]
        c = self.concepts[idx]
        a = self.attributes[idx] if self.attributes is not None else None
        subpop_idx = self._get_subpopulation_idx(c, [a])

        # Return a tuple of images, labels, and extra_dict 
        return {
            "img_code": idx,
            "labels": y,
            "features": x,
            "concepts": c,
            "extra_dict": {"attributes": a, "population_idx": subpop_idx}
        }

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from tqdm import tqdm

    config = OmegaConf.create({
        "data_path": "/home/yk449/datasets/",
        "train_file_name": "train_sum_mnist_concepts_1_2_minority.npz",
        "val_file_name": "test_sum_mnist_concepts.npz",
        "test_file_name": "test_sum_mnist_concepts.npz"
    })
    train_loader, val_loader, test_loader = get_MNIST_SUM_dataset(config)
    print(len(train_loader), len(val_loader), len(test_loader))

    labels = []
    for data in tqdm(test_loader):
        labels.append(data["labels"])
    
    np_labels = np.array(labels)
    hist = np.histogram(np_labels, bins=7)
    print(f"Labels histogram: {hist}")
    print("all data loaded successfully")
