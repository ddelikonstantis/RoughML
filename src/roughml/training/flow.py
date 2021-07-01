from pathlib import Path
from zipfile import ZipFile

from roughml.data.generators import NonGaussianSurfaceGenerator
from roughml.data.sets import NanoroughSurfaceDataset
from roughml.plot import animate_epochs, as_3d_surface, as_grayscale_image, plot_against
from roughml.shared.configuration import Configuration
from roughml.training.manager import TrainingManager


def load_dataset(
    dataset_path, transforms=[], limit=None, cache_dir=Path.cwd() / "surfaces"
):
    if dataset_path.is_file():
        if not cache_dir.is_dir():
            cache_dir.mkdir(parents=True, exist_ok=True)

            with ZipFile(dataset_path, "r") as zip_file:
                zip_file.extractall(cache_dir)

    if cache_dir.is_dir():
        dataset = NanoroughSurfaceDataset.from_matlab(
            cache_dir, transforms=transforms, limit=limit
        )
    else:
        generate_surfaces = NonGaussianSurfaceGenerator()

        surfaces = []
        for surface in generate_surfaces(limit):
            surfaces.append(surface)

        dataset = NanoroughSurfaceDataset.from_list(surfaces, transforms=transforms)

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

        if not hasattr(self, "plot"):
            self.plot = Configuration(
                save_directory=None,
                gray_scale={"limit": None, "save_path_fmt": None},
                surface={"limit": None, "save_path_fmt": None},
            )

        if not hasattr(self, "animation"):
            self.animation = Configuration(
                indices=[
                    0,
                ],
                save_path=None,
                parameters={
                    "interval": 1000,
                    "repeat_delay": 1000,
                    "blit": True,
                    "fps": 15,
                    "bitrate": 1800,
                },
            )

        if not hasattr(self.animation, "parameters"):
            self.animation.parameters = Configuration(
                interval=1000,
                repeat_delay=1000,
                blit=True,
                fps=15,
                bitrate=1800,
            )

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
            if self.content_loss.cache.is_file():
                content_loss = self.content_loss.type.from_pickle(
                    self.content_loss.cache
                )
            else:
                content_loss = self.content_loss.type(surfaces=dataset.surfaces)
                content_loss.to_pickle(self.content_loss.cache)

            training_manager = TrainingManager(
                content_loss=content_loss, **self.training_manager.to_dict()
            )

        (
            generator_losses,
            discriminator_losses,
            discriminator_output_reals,
            discriminator_output_fakes,
            fixed_fakes,
        ) = list(zip(*list(training_manager(generator, discriminator, dataset))))

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

        animate_epochs(
            fixed_fakes,
            indices=self.animation.indices,
            save_path=self.animation.save_path,
            **self.animation.parameters.to_dict()
        )

        if self.plot.save_directory is not None:
            self.plot.save_directory.mkdir(parents=True, exist_ok=True)

            if self.plot.grayscale.save_path_fmt is not None:
                for i in range(self.plot.grayscale.limit):
                    true_surface = dataset.surfaces[i].squeeze()
                    fake_surface = fixed_fakes[-1][i].squeeze()

                    as_grayscale_image(
                        true_surface,
                        self.plot.save_directory
                        / (self.plot.grayscale.save_path_fmt % ("true", i)),
                    )
                    as_grayscale_image(
                        fake_surface,
                        self.plot.save_directory
                        / (self.plot.grayscale.save_path_fmt % ("fake", i)),
                    )

            if self.plot.surface.save_path_fmt is not None:
                for i in range(self.plot.surface.limit):
                    true_surface = dataset.surfaces[i].squeeze()
                    fake_surface = fixed_fakes[-1][i].squeeze()

                    as_3d_surface(
                        true_surface,
                        self.plot.save_directory
                        / (self.plot.surface.save_path_fmt % ("true", i)),
                    )
                    as_3d_surface(
                        fake_surface,
                        self.plot.save_directory
                        / (self.plot.surface.save_path_fmt % ("fake", i)),
                    )
