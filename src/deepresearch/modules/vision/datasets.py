"""Vision dataset loaders for MNIST, CIFAR-10, etc."""

import base64
import io
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class VisionSample:
    """A single vision sample."""

    sample_id: str
    image: Image.Image
    label: int
    label_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_base64(self, format: str = "PNG") -> str:
        """Convert image to base64 string for API calls."""
        buffer = io.BytesIO()
        self.image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def to_data_url(self, format: str = "PNG") -> str:
        """Convert image to data URL for API calls."""
        b64 = self.to_base64(format)
        mime = f"image/{format.lower()}"
        return f"data:{mime};base64,{b64}"


@dataclass
class VisionDatasetConfig:
    """Configuration for a vision dataset."""

    name: str
    num_classes: int
    class_names: list[str]
    image_size: tuple[int, int]
    channels: int  # 1 for grayscale, 3 for RGB
    train_size: int
    test_size: int


# Dataset configurations
VISION_DATASETS = {
    "mnist": VisionDatasetConfig(
        name="MNIST",
        num_classes=10,
        class_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        image_size=(28, 28),
        channels=1,
        train_size=60000,
        test_size=10000,
    ),
    "cifar10": VisionDatasetConfig(
        name="CIFAR-10",
        num_classes=10,
        class_names=[
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ],
        image_size=(32, 32),
        channels=3,
        train_size=50000,
        test_size=10000,
    ),
    "cifar100": VisionDatasetConfig(
        name="CIFAR-100",
        num_classes=100,
        class_names=[
            "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
            "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
            "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
            "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
            "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
            "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
            "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
            "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
            "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
            "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
            "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
            "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
            "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
            "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
        ],
        image_size=(32, 32),
        channels=3,
        train_size=50000,
        test_size=10000,
    ),
}


class VisionDatasetLoader:
    """Loads vision datasets for experiments."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else Path("data/vision")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._datasets: dict[str, Any] = {}

    def get_config(self, dataset: str) -> VisionDatasetConfig:
        """Get dataset configuration."""
        if dataset not in VISION_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(VISION_DATASETS.keys())}")
        return VISION_DATASETS[dataset]

    def _load_torchvision_dataset(self, dataset: str, train: bool = False) -> Any:
        """Load dataset using torchvision."""
        import torchvision
        import torchvision.transforms as transforms

        cache_key = f"{dataset}_{'train' if train else 'test'}"
        if cache_key in self._datasets:
            return self._datasets[cache_key]

        transform = transforms.ToTensor()

        if dataset == "mnist":
            ds = torchvision.datasets.MNIST(
                root=str(self.data_dir),
                train=train,
                download=True,
                transform=transform,
            )
        elif dataset == "cifar10":
            ds = torchvision.datasets.CIFAR10(
                root=str(self.data_dir),
                train=train,
                download=True,
                transform=transform,
            )
        elif dataset == "cifar100":
            ds = torchvision.datasets.CIFAR100(
                root=str(self.data_dir),
                train=train,
                download=True,
                transform=transform,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self._datasets[cache_key] = ds
        return ds

    def load_samples(
        self,
        dataset: str,
        num_samples: int,
        split: str = "test",
        seed: int = 42,
        balanced: bool = True,
    ) -> list[VisionSample]:
        """Load samples from a vision dataset.

        Args:
            dataset: Dataset name (mnist, cifar10, cifar100)
            num_samples: Number of samples to load
            split: 'train' or 'test'
            seed: Random seed for reproducibility
            balanced: If True, sample equally from each class
        """
        config = self.get_config(dataset)
        ds = self._load_torchvision_dataset(dataset, train=(split == "train"))

        random.seed(seed)
        np.random.seed(seed)

        if balanced:
            # Sample equally from each class
            samples_per_class = num_samples // config.num_classes
            extra = num_samples % config.num_classes

            # Group indices by class
            class_indices: dict[int, list[int]] = {i: [] for i in range(config.num_classes)}
            for idx in range(len(ds)):
                _, label = ds[idx]
                class_indices[label].append(idx)

            # Sample from each class
            selected_indices = []
            for class_idx in range(config.num_classes):
                n = samples_per_class + (1 if class_idx < extra else 0)
                indices = class_indices[class_idx]
                selected = random.sample(indices, min(n, len(indices)))
                selected_indices.extend(selected)

            random.shuffle(selected_indices)
        else:
            # Random sampling
            all_indices = list(range(len(ds)))
            selected_indices = random.sample(all_indices, min(num_samples, len(all_indices)))

        # Convert to VisionSample objects
        samples = []
        for idx in selected_indices[:num_samples]:
            tensor, label = ds[idx]

            # Convert tensor to PIL Image
            if dataset == "mnist":
                # MNIST: [1, 28, 28] -> grayscale
                arr = (tensor.squeeze().numpy() * 255).astype(np.uint8)
                image = Image.fromarray(arr, mode="L")
                # Convert to RGB for API compatibility
                image = image.convert("RGB")
            else:
                # CIFAR: [3, 32, 32] -> RGB
                arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                image = Image.fromarray(arr, mode="RGB")

            sample = VisionSample(
                sample_id=f"{dataset}_{split}_{idx}",
                image=image,
                label=label,
                label_name=config.class_names[label],
                metadata={
                    "dataset": dataset,
                    "split": split,
                    "original_index": idx,
                },
            )
            samples.append(sample)

        return samples

    def load_few_shot_examples(
        self,
        dataset: str,
        shots_per_class: int = 1,
        seed: int = 42,
    ) -> dict[int, list[VisionSample]]:
        """Load few-shot examples for each class.

        Args:
            dataset: Dataset name
            shots_per_class: Number of examples per class
            seed: Random seed

        Returns:
            Dictionary mapping class index to list of samples
        """
        config = self.get_config(dataset)
        ds = self._load_torchvision_dataset(dataset, train=True)

        random.seed(seed)

        # Group by class
        class_indices: dict[int, list[int]] = {i: [] for i in range(config.num_classes)}
        for idx in range(len(ds)):
            _, label = ds[idx]
            class_indices[label].append(idx)

        # Sample from each class
        examples: dict[int, list[VisionSample]] = {}
        for class_idx in range(config.num_classes):
            indices = random.sample(
                class_indices[class_idx],
                min(shots_per_class, len(class_indices[class_idx]))
            )
            examples[class_idx] = []

            for idx in indices:
                tensor, label = ds[idx]

                if dataset == "mnist":
                    arr = (tensor.squeeze().numpy() * 255).astype(np.uint8)
                    image = Image.fromarray(arr, mode="L").convert("RGB")
                else:
                    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    image = Image.fromarray(arr, mode="RGB")

                sample = VisionSample(
                    sample_id=f"{dataset}_train_{idx}",
                    image=image,
                    label=label,
                    label_name=config.class_names[label],
                    metadata={"dataset": dataset, "split": "train", "original_index": idx},
                )
                examples[class_idx].append(sample)

        return examples

    def get_class_names(self, dataset: str) -> list[str]:
        """Get class names for a dataset."""
        return self.get_config(dataset).class_names
