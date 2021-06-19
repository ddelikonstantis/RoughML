from pathlib import Path
from zipfile import ZipFile

from src.roughml.data.generators import NonGaussianSurfaceGenerator
from src.roughml.data.sets import NanoroughSurfaceDataset, NanoroughSurfaceMatLabDataset
from src.roughml.plot import plot_against
from src.roughml.shared.configuration import Configuration
from src.roughml.training.manager import TrainingManager


def load_dataset(
    dataset_path, transforms=[], limit=None, cache_dir=Path.cwd() / "surfaces"
):
    if dataset_path.is_file():
        if not cache_dir.is_dir():
            cache_dir.mkdir(parents=True, exist_ok=True)

            with ZipFile(dataset_path, "r") as zip_file:
                zip_file.extractall(cache_dir)

    if cache_dir.is_dir():
        dataset = NanoroughSurfaceMatLabDataset(
            cache_dir, transforms=transforms, limit=limit
        )
    else:
        generate = NonGaussianSurfaceGenerator()
        dataset = NanoroughSurfaceDataset(list(generate(limit)), transforms=transforms)

    return dataset


class TrainingFlow(Configuration):
    def __init__(self, **props):
        super().__init__(**props)

        if not hasattr(self.dataset, "limit"):
            self.dataset.limit = None

        if not hasattr(self.dataset, "transforms"):
            self.dataset.transforms = []

        if not hasattr(self, "training_manager"):
            self.training_manager = {}

    def __call__(self, generator, discriminator):
        dataset = load_dataset(
            self.dataset.path,
            transforms=self.dataset.transforms,
            limit=self.dataset.limit,
            cache_dir=Path.cwd() / "surfaces",
        )

        if hasattr(self.training_manager, "content_loss"):
            training_manager = TrainingManager(**self.training_manager.to_dict())
        else:
            content_loss = self.content_loss_type(surfaces=dataset.surfaces)

            training_manager = TrainingManager(
                content_loss=content_loss, **self.training_manager.to_dict()
            )

        (
            generator_losses,
            discriminator_losses,
            discriminator_output_reals,
            discriminator_output_fakes,
        ) = training_manager(generator, discriminator, dataset)

        plot_against(
            generator_losses,
            discriminator_losses,
            title="Mean Generator vs Discriminator loss per epoch",
            xlabel="Epoch",
            ylabel="Loss",
            labels=("Generator", "Discriminator"),
        )

        plot_against(
            discriminator_output_reals,
            discriminator_output_fakes,
            title="Mean Discriminator Output per epoch",
            xlabel="Epoch",
            ylabel="Discriminator Output",
            labels=("Real Data", "Generator Data"),
        )