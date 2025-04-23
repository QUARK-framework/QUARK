#  Copyright 2021 The QUARK Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import os
from typing import Iterable, TypedDict

import numpy as np
import pandas as pd
import pkg_resources
import torch
import torch.nn as nn
import torchvision
from modules.applications.qml.classification.data.data_handler.DataHandler import DataHandler
from modules.applications.qml.classification.data.data_handler.MetricsClassification import (
    MetricsClassification,
)
from modules.applications.qml.classification.training.Hybrid import Hybrid
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from utils import end_time_measurement, start_time_measurement


class ImageData(DataHandler):
    """
    A data handler for image datasets. This class loads a dataset from a specified path and provides
    methods for data transformation and evaluation.

    """

    def __init__(self):
        """
        The continuous data class loads a dataset from the path
        src/modules/applications/QML/classification/data
        """
        super().__init__("")
        self.submodule_options = ["Hybrid"]
        self.transformation = None
        self.dataset = None
        self.n_registers = None
        self.gc = None
        self.n_qubits = None
        self.data_folder = pkg_resources.resource_filename("modules.applications.qml.classification.data", "Images_2D")

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {"name": "pandas", "version": "2.2.2"},
            {"name": "pillow", "version": "11.1.0"},
            {"name": "torch", "version": "2.2.2"},
            {"name": "torchvision", "version": "0.17.2"},
            {"name": "tqdm", "version": "4.67.1"},
            {"name": "numpy", "version": "1.26.4"},
        ]

    def get_default_submodule(self, option: str) -> Hybrid:
        if option == "Hybrid":
            return Hybrid()

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application

        :return:

                 .. code-block:: python

                    return {
                        "data_set": {
                            "values": ["Concrete_Crack", "MNIST"],
                            "description": "Which dataset do you want to use?",
                        },
                        "n_images_per_class": {
                            "values": [100, 1000, 2000, 3000],
                            "description": "Number of images to extract for each class",
                        },
                        "n_classes": {
                            "values": [2, 4, 10],
                            "description": "How many classes to benchmark on (Concrete_Crack only works with 2)?",
                        },
                        "noise_sigma": {
                            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                            "description": "Variance of gaussian noise",
                        },
                    }

        """
        return {
            "data_set": {
                "values": ["Concrete_Crack", "MNIST"],
                "description": "Which dataset do you want to use?",
            },
            "n_images_per_class": {
                "values": [100, 1000, 2000, 3000],
                "description": "Number of images to extract for each class",
            },
            "n_classes": {
                "values": [2, 4, 10],
                "description": "How many classes to benchmark on (Concrete_Crack only works with 2)?",
            },
            "noise_sigma": {
                "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "description": "Variance of gaussian noise",
            },
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            data_set: str
            n_images_per_class: int
            n_classes: int
            noise_sigma: float
        """

        data_set: str
        n_images_per_class: int
        n_classes: int
        noise_sigma: float

    class CustomDataset(Dataset):
        """
        Dataset object used by Pytorch.
        """

        def __init__(
            self,
            embeddings_file,
            index_selection: Iterable[int],
            transform=None,
            target_transform=None,
        ):
            self.embeddings_df = pd.read_csv(embeddings_file, index_col=0).iloc[index_selection]
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.embeddings_df)

        def __getitem__(self, idx):
            data = self.embeddings_df.iloc[idx, 2:].values.astype(np.float32)
            label = self.embeddings_df.iloc[idx, 0]
            path = self.embeddings_df.iloc[idx, 1]
            return data, self.map_label(label), path

        def map_label(self, label):
            """
            Converts string labels to integers.

            Args:
                label: The original label as string.

            Returns:
                An integer number corresponding to the label.
            """
            if label == "Negative":
                return 0
            elif label == "Positive":
                return 1

    def data_load(self, gen_mod: dict, config: dict) -> dict:
        """
        The chosen dataset is loaded and split into a training set.

        :param gen_mod: Dictionary with collected information of the previous modules
        :type gen_mod: dict
        :param config: Config specifying the parameters of the data handler
        :type config: dict
        :return: Must always return the mapped problem and the time it took to create the mapping
        :rtype: tuple(any, float)
        """
        self.dataset_name = config["data_set"]
        self.n_qubits = gen_mod["n_qubits"]
        self.n_classes = config["n_classes"]
        self.n_images_per_class = config["n_images_per_class"]
        total_n_images = config["n_classes"] * config["n_images_per_class"]
        noise_sigma = config["noise_sigma"]

        if self.dataset_name == "Concrete_Crack":
            if config["n_classes"] != 2:
                raise Exception("Sorry, number of classes does not work for this dataset. Should be 2.")

            logging.info("Creating index")
            self.create_data_index()
            logging.info("Embedding dataset")
            embeddings_file = self.embed_dataset(n_images_per_class=self.n_images_per_class, noise_sigma=noise_sigma)
            self.dataset_train, self.dataset_val, self.dataset_test = self.create_torch_dataset(
                total_n_images, embeddings_file
            )

        elif self.dataset_name == "MNIST":
            self.dataset_train, self.dataset_val, self.dataset_test = self.create_mnist_dataset(
                self.n_classes, self.n_images_per_class
            )

        else:
            logging.error("Unknown use case")

        application_config = {
            "dataset_name": self.dataset_name,
            "n_qubits": self.n_qubits,
            "n_classes": self.n_classes,
            "dataset_train": self.dataset_train,
            "dataset_val": self.dataset_val,
            "dataset_test": self.dataset_test,
            "store_dir_iter": gen_mod["store_dir_iter"],
        }

        self.classification_metrics = MetricsClassification()
        application_config["classification_metrics"] = self.classification_metrics

        return application_config

    # TODO this method is not creating image embeddings.
    # The dataloading for the concrete crack images does:
    # 1. indexing and sampling of the files
    # 2. embedding of the sampled images using resnet
    # 3. packaging into a torch dataloader
    # The dataloading for mnist instead does only steps 1 and 3.
    # Step two for mnist is done in the training part (Hybrid module).
    # This lack of consistency needs to be fixed.
    def create_mnist_dataset(self, n_classes: int, n_images_per_class: int):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ]
        )

        batch_size = 64

        trainset = torchvision.datasets.MNIST(root=self.data_folder, train=True, download=True, transform=transform)

        idx = self.keep_first_k_ones(torch.as_tensor(trainset.targets) == 0, k=n_images_per_class)
        for j in range(1, n_classes):
            idx += self.keep_first_k_ones(torch.as_tensor(trainset.targets) == j, k=n_images_per_class)
        subset_train = torch.utils.data.dataset.Subset(trainset, np.where(idx == 1)[0])

        trainloader = torch.utils.data.DataLoader(subset_train, batch_size=batch_size, shuffle=True, num_workers=0)

        testset = torchvision.datasets.MNIST(root=self.data_folder, train=False, download=True, transform=transform)

        idx = self.keep_first_k_ones(torch.as_tensor(testset.targets) == 0, k=n_images_per_class)
        for j in range(1, n_classes):
            idx += self.keep_first_k_ones(torch.as_tensor(testset.targets) == j, k=n_images_per_class)
        subset_test = torch.utils.data.dataset.Subset(testset, np.where(idx == 1)[0])
        testloader = torch.utils.data.DataLoader(subset_test, batch_size=batch_size, shuffle=False, num_workers=0)

        return trainloader, testloader, testloader

    @staticmethod
    def keep_first_k_ones(tensor, k=1000):
        """
        Keeps only the first k ones in a 1-D tensor of 0s and 1s and sets the rest to 0.

        Args:
            tensor (torch.Tensor): A 1-D tensor containing 0s and 1s.
            k (int): The number of ones to keep.  Defaults to 1000.

        Returns:
            torch.Tensor: A new tensor with only the first k ones remaining.
        """

        indices = (tensor == 1).nonzero(as_tuple=False).flatten()  # indices of all 1s
        if len(indices) > k:
            indices_to_zero = indices[k:]  # indices of ones to change to zeros
            new_tensor = tensor.clone()  # creating a clone is crucial to avoid modifying the input tensor
            new_tensor[indices_to_zero] = 0
            return new_tensor
        else:
            return tensor.clone()

    def create_torch_dataset(
        self,
        tot_train_test_datapoints,
        embeddings_file,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Creates train, val and test dataloaders.
        """
        train_indexes = []
        val_indexes = []
        test_indexes = []
        p_val = 0.15
        p_test = 0.10

        np.random.seed(42)
        for j in range(tot_train_test_datapoints):
            p = np.random.rand()
            if p < p_val:
                val_indexes.append(j)
            elif p < p_val + p_test:
                test_indexes.append(j)
            else:
                train_indexes.append(j)

        train_dataset = self.CustomDataset(embeddings_file=embeddings_file, index_selection=train_indexes)
        val_dataset = self.CustomDataset(embeddings_file=embeddings_file, index_selection=val_indexes)
        test_dataset = self.CustomDataset(embeddings_file=embeddings_file, index_selection=test_indexes)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        return train_dataloader, val_dataloader, test_dataloader

    def create_data_index(self, overwrite: bool = True) -> None:
        """
        Crawls the filenames in the data folder and writes an index in a CSV file.

        Args:
            data_folder: Path to the folder containing the images.
            overwrite: Set to True if file needs to be overwritten.

        Raises:
            FileExistsError: If overwrite is set to False and the index CSV file already exists.
        """
        output_path = os.path.join(self.data_folder, "test_data", "index.csv")
        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError(
                "Index file already exists. To overwrite, call the function with the argument overwrite=True"
            )

        index_list = []
        for root, dirs, files in os.walk(os.path.join(self.data_folder, "test_data")):
            for file in files:
                if file.endswith(".jpg"):
                    label = os.path.basename(root)
                    index_list.append(
                        {
                            "file": f"{label.lower()[0]}_{file}",
                            "label": label,
                            "path": os.path.join(root, file),
                        }
                    )

        index_df = pd.DataFrame(index_list)
        index_df = index_df.set_index("file")
        index_df.to_csv(output_path)

    def sample_index(self, n_samples_per_class: int, random_state: int = 42) -> pd.DataFrame:
        """
        Sample images from the index file.

        Args:
            n_samples_per_class: Number of samples to extract for each class.
            random_state: Random seed for reproducibility.

        Returns:
            A pandas DataFrame.
        """
        index_df = pd.read_csv(os.path.join(self.data_folder, "test_data", "index.csv"), index_col=0)
        min_images_per_class = min(index_df.label.value_counts())
        assert n_samples_per_class <= min_images_per_class, (
            f"n_samples_per_class is too big. Only {min_images_per_class} images per label are available."
        )

        shuffled_df = index_df.sample(frac=1.0, random_state=random_state)
        negative_samples_df = shuffled_df[shuffled_df.label == "Negative"].iloc[0:n_samples_per_class]
        positive_samples_df = shuffled_df[shuffled_df.label == "Positive"].iloc[0:n_samples_per_class]

        return pd.concat([negative_samples_df, positive_samples_df])

    def add_noise(self, image: np.array, sigma: float, mean: float = 0.0) -> np.array:
        """
        Add Gaussian noise to the input images with sigma

        Args:
            image: Image
            sigma: Sigma of Gaussian noise
            mean: Mean of Gaussian noise

        Returns:
            A numpy array with pixel values between 0 and 1.
        """
        gaussian = np.random.normal(mean, sigma, image.shape)
        noisy_df = image + gaussian
        noisy_df = np.clip(noisy_df, 0, 1)
        return noisy_df

    def embed_dataset(self, n_images_per_class: int, noise_sigma: float) -> pd.DataFrame:
        """
        Embed images from the data folder using the pretrained Resnet18 model.
        Stores the embeddings in a CSV file.
        Embeddings are added in an incremental way if an embedding CSV file already exists.

        Args:
            n_images_per_class: Number of images to embed for each class.
            noise_sigma: Sigma of Gaussian noise applied to the dataset

        Returns:
            A pandas DataFrame.
        """
        resnet18_embedder = self.build_truncated_resnet_model()
        resnet18_embedder.eval()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        embedding_file = os.path.join(self.data_folder, "embeddings.csv")
        existing_embedding_df = pd.DataFrame(
            columns=["file", "label", "path"] + [f"e_{j:03d}" for j in range(512)]
        ).set_index("file")

        sampled_df = self.sample_index(n_samples_per_class=n_images_per_class)
        output_df = pd.merge(
            left=sampled_df,
            right=existing_embedding_df,
            how="left",
            on=["file", "label", "path"],
        )
        index_to_embed = sampled_df.index.drop(existing_embedding_df.index)

        imgs_array = np.empty((len(index_to_embed), 3, 227, 227))
        counter = 0
        for idx, row in tqdm(
            sampled_df.loc[index_to_embed].iterrows(),
            total=sampled_df.loc[index_to_embed].shape[0],
        ):
            img_path = row["path"]
            img_array = self.load_image(img_path)
            if noise_sigma > 0:
                img_array = self.add_noise(img_array, noise_sigma)
            img_array = np.rollaxis(img_array, -1, 0)
            imgs_array[counter] = img_array
            counter += 1

        embeddings = []
        batch_size = 16
        for batch_start in tqdm(range(0, len(imgs_array), batch_size)):
            imgs_tensor = torch.tensor(imgs_array[batch_start : batch_start + batch_size]).float()
            imgs_tensor = normalize(imgs_tensor)
            pred = resnet18_embedder(imgs_tensor)
            embeddings.extend(pred.detach().reshape(-1, 512).tolist())

        output_df.loc[index_to_embed, [f"e_{j:03d}" for j in range(512)]] = embeddings
        output_df.to_csv(embedding_file)

        return embedding_file

    def build_truncated_resnet_model(self) -> nn.Module:
        """
        Remove the last layer from a pretrained Resnet18 model.

        Returns:
            A Pytorch model.
        """
        # resnet18 = models.resnet18(pretrained=True)  # Deprecated
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        keep_layers = list(resnet18.children())[:-1]
        truncated_resnet18 = nn.Sequential(*keep_layers)
        return truncated_resnet18

    def load_image(self, image_path: str) -> np.array:
        """
        Load an image from disk.

        Args:
            image_path: Path to image.

        Returns:
            A numpy array with pixel values between 0 and 1.
        """
        img = Image.open(image_path)
        img_array = np.array(img) / 255
        return img_array

    def evaluate(self, solution: list, **kwargs) -> tuple[float, float]:
        """
        Calculate Accuracy in the original data set.

        :param param1:
        :type param1: list
        :return: Accuracy for the test case and the time it took to calculate it.
        :rtype: tuple(float, float)
        """
        start = start_time_measurement()
        evaluate_dict = None

        return evaluate_dict, end_time_measurement(start)
