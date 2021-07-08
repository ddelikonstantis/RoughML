import logging
import logging.config
from datetime import datetime

from roughml.plot import animate_epochs, as_3d_surface, as_grayscale_image, plot_against
from roughml.shared.configuration import Configuration
from roughml.training.manager import TrainingManager

logger = logging.getLogger(__name__)


class TrainingFlow(Configuration):
    def __init__(self, **props):
        super().__init__(**props)

        if not hasattr(self, "training_manager"):
            self.training_manager = {}

        if not hasattr(self, "plot"):
            self.plot = Configuration(
                gray_scale={"limit": None, "save_path_fmt": None},
                surface={"limit": None, "save_path_fmt": None},
                against={"save_path_fmt": None},
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
        for path, dataset in self.data.loader():
            dataset_output_dir = (
                self.output_dir
                / path.stem
                / f"{generator.__class__.__name__}_{discriminator.__class__.__name__}"
                / datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            )

            checkpoint_dir = dataset_output_dir / "Checkpoint"
            checkpoint_dir.mkdir(parents=True)

            self.training_manager.checkpoint.directory = checkpoint_dir
            content_loss_cache = checkpoint_dir / self.content_loss.cache

            logging_dir = dataset_output_dir / "Logging"
            logging_dir.mkdir(parents=True)

            logging.config.dictConfig(
                self.logging.callback(self.logging.config.to_dict(), logging_dir)
            )

            plotting_dir = dataset_output_dir / "Plots"
            plotting_dir.mkdir(parents=True)

            animation_save_path = plotting_dir / self.animation.save_path
            self.plot.save_directory = plotting_dir

            logger.info(
                "Running the flow on a %s/%s pair over dataset %s",
                generator.__class__.__name__,
                discriminator.__class__.__name__,
                path,
            )

            if hasattr(self.training_manager, "content_loss"):
                training_manager = TrainingManager(**self.training_manager.to_dict())
            else:
                if content_loss_cache is not None and content_loss_cache.is_file():
                    content_loss = self.content_loss.type.from_pickle(
                        content_loss_cache
                    )
                else:
                    content_loss = self.content_loss.type(surfaces=dataset.surfaces)

                    if content_loss_cache is not None:
                        content_loss.to_pickle(content_loss_cache)

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

            save_path_loss, save_path_discriminator_output = None, None
            if self.plot.against.save_path_fmt is not None:
                save_path_loss = self.plot.against.save_path_fmt % ("loss",)
                save_path_discriminator_output = self.plot.against.save_path_fmt % (
                    "discriminator_output",
                )

            plot_against(
                generator_losses,
                discriminator_losses,
                title="Mean Generator vs Discriminator loss per epoch",
                xlabel="Epoch",
                ylabel="Loss",
                labels=("Generator", "Discriminator"),
                save_path=self.plot.save_directory / save_path_loss,
            )

            plot_against(
                discriminator_output_reals,
                discriminator_output_fakes,
                title="Mean Discriminator Output per epoch",
                xlabel="Epoch",
                ylabel="Discriminator Output",
                labels=("Real Data", "Generator Data"),
                save_path=self.plot.save_directory / save_path_discriminator_output,
            )

            animate_epochs(
                fixed_fakes,
                indices=self.animation.indices,
                save_path=animation_save_path,
                **self.animation.parameters.to_dict(),
            )

            if self.plot.save_directory is not None:
                (
                    self.plot.save_directory / self.plot.grayscale.save_path_fmt
                ).parent.mkdir(parents=True, exist_ok=True)

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
                    (
                        self.plot.save_directory / self.plot.surface.save_path_fmt
                    ).parent.mkdir(parents=True, exist_ok=True)

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
