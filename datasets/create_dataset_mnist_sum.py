from typing import Optional
from torchvision.datasets import MNIST
import numpy as np
from tqdm import tqdm


class MNIST_Dataloader(MNIST):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train=train, transform=transform, download=False)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return {
            "img_code": index,
            "labels": img,
            "features": target,
            "concepts": None,
        }


def create_mnist_sum_dataset(
        digits_range: np.ndarray, 
        root_path: str, 
        save_path: str, 
        train: bool = True,
        n_samples: int = 60000, 
        required_distribution: Optional[np.ndarray] = None,
        data_filename: str = "mnist_sum_concepts.npz"):

    n_dists = len(digits_range) ** 2
    dist_hist = np.zeros((len(digits_range), len(digits_range))) 

    if required_distribution is None:
        equal_dist = (n_samples // n_dists)
        required_distribution = np.ones((len(digits_range), len(digits_range))) * equal_dist
    
    # Create a dataset for the specified digits
    dataloader = MNIST(
        root=root_path,
        train=train,
        download=False,
    )

    imgs, labels, concepts = [], [], []
    n_created_train = 0
    img1, img2 = None, None

    pbar = tqdm(total=n_samples, desc="Creating training dataset")
    while n_created_train < n_samples:
        # Get a random sample of images and labels from the dataset
        img, target = dataloader[np.random.randint(len(dataloader))]

        # Check if the target is in the specified digits range
        if target not in digits_range:
            continue

        # If we have two images, create a new image by concatenating them
        if img1 is None:
            img1 = img
            target1 = target
            continue

        elif img2 is None:
            img2 = img
            target2 = target
        
        else:
            raise ValueError("img1 and img2 should be None at this point")

        if dist_hist[target1, target2] >= required_distribution[target1, target2]:
            img1 = img2 = None
            print("Skipping pair due to histogram limit")
            # print(dist_hist)
            continue

        # Create a new image by concatenating the two images
        new_img = np.concatenate((img1, img2), axis=1)
        new_label = target1 + target2
        concept = np.zeros(2 * len(digits_range))
        concept[target1] = 1
        concept[len(digits_range) + target2] = 1

        imgs.append(new_img)
        labels.append(new_label)
        concepts.append(concept)

        dist_hist[target1, target2] += 1
        img1, img2 = None, None
        n_created_train += 1
        pbar.update(1)

    # Convert lists to numpy arrays
    imgs = np.array(imgs)
    labels = np.array(labels)
    concepts = np.array(concepts)

    # Save the training dataset
    stage = "train" if train else "test"
    full_save_path = save_path + f"{stage}_{data_filename}"

    print(f"Saving dataset to {full_save_path}")
    np.savez(
        full_save_path,
        imgs=imgs,
        labels=labels,
        concepts=concepts,
        allow_pickle=True,
    )

    pbar.close()


if __name__ == "__main__":
    digits_range = np.array([0, 1, 2, 3])
    save_path = "/home/yk449/datasets/mnist/mnist_sum/"
    root_path = "/home/yk449/datasets/mnist/"
    
    create_set = 'train'
    if create_set == 'train':
        print("Creating training dataset")
        n_samples = 57000
        required_distribution = np.array([
            [3750, 3750, 3750, 3750],
            [3750, 3750, 3750, 3750],
            [3750, 1500, 3750, 3750],
            [3750, 3750, 3750, 3750]
        ])
        data_filename = "sum_mnist_concepts_2_1_minority_04.npz"
        create_mnist_sum_dataset(
            digits_range=digits_range, 
            root_path=root_path, 
            save_path=save_path, 
            train=True,
            n_samples=n_samples,
            required_distribution=required_distribution,
            data_filename=data_filename,
            )
    else:
        print("Creating testing dataset")
        n_samples = 10000
        create_mnist_sum_dataset(
            digits_range=digits_range, 
            root_path=root_path, 
            save_path=save_path, 
            train=False,
            n_samples=n_samples,)
