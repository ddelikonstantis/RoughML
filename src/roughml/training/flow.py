import logging
import logging.config
from datetime import datetime, timedelta
from time import time

import torch
import numpy as np
import pandas as pd

from roughml.content.loss import VectorSpaceContentLoss
from roughml.plot import animate_epochs, as_3d_surface, as_grayscale_image, plot_against
from roughml.shared.configuration import Configuration
from roughml.shared.context_managers import ExceptionLoggingHandler
from roughml.training.manager import TrainingManager

logger = logging.getLogger(__name__)


class TrainingFlow(Configuration):
    def __init__(self, **props):
        super().__init__(**props)

        if not hasattr(self.training, "manager"):
            self.training.manager = Configuration()

        if not hasattr(self.training, "callbacks"):
            self.training.callbacks = []

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
            )

        if not hasattr(self.animation, "parameters"):
            self.animation.parameters = Configuration(
                interval=1000,
                repeat_delay=1000,
                blit=True,
                fps=15,
                bitrate=1800,
            )

        if not hasattr(self, "suppress_exceptions"):
            self.suppress_exceptions = True

        if not hasattr(self, "NGramGraphLoss"):
            self.NGramGraphLoss = Configuration(type=None, cache=None)

            
    def __call__(self, get_generator, get_discriminator):
        for path, dataset in self.data.loader():
            logger.info("Instantiating generator and discriminator")

            generator = get_generator()
            discriminator = get_discriminator(generator)

            start_time = time()

            with ExceptionLoggingHandler(
                logger, suppress_exceptions=self.suppress_exceptions
            ) as exception_logging_handler:
                self.process_dataset(generator, discriminator, path, dataset)

            end_time = time()

            for callback in filter(None, self.training.callbacks):
                callback(
                    log_file=self.logging.config.handlers.file.filename,
                    dataset=path,
                    generator=generator.__class__.__name__,
                    discriminator=discriminator.__class__.__name__,
                    elapsed_time=str(timedelta(seconds=end_time - start_time)),
                    succeeded=exception_logging_handler.success,
                )

    def process_dataset(self, generator, discriminator, path, dataset):
        dataset_output_dir = (
            self.output_dir
            / path.stem
            / f"{generator.__class__.__name__}_{discriminator.__class__.__name__}"
            / datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        )

        checkpoint_dir = dataset_output_dir / "Checkpoint"
        checkpoint_dir.mkdir(parents=True)

        if hasattr(self.training.manager, "checkpoint"):
            self.training.manager.checkpoint.directory = checkpoint_dir

        NGramGraphLoss_cache = None
        if getattr(self.NGramGraphLoss, "cache", None) is not None:
            NGramGraphLoss_cache = checkpoint_dir / self.NGramGraphLoss.cache

        logging_dir = dataset_output_dir / "Logging"
        logging_dir.mkdir(parents=True)

        self.logging.config = self.logging.callback(self.logging.config, logging_dir)

        logging.config.dictConfig(self.logging.config.to_dict())

        plotting_dir = dataset_output_dir / "Plots"
        plotting_dir.mkdir(parents=True)

        animation_save_path = plotting_dir / self.animation.save_path
        self.plot.save_directory = plotting_dir

        self.cuda_available = torch.cuda.is_available()
        self.cuda_id = torch.cuda.current_device()
        self.cuda_name = torch.cuda.get_device_name(self.cuda_id)

        logger.info(
            "Is CUDA supported? %s. Running the framework on device ID:%s with name: %s",
            self.cuda_available,
            self.cuda_id,
            self.cuda_name
        )

        logger.info(
            "Running the flow on a %s/%s pair over dataset %s",
            generator.__class__.__name__,
            discriminator.__class__.__name__,
            path,
        )

        HeightHistogramAndFourierLoss = VectorSpaceContentLoss(surfaces=dataset.surfaces)

        if hasattr(self.training.manager, "NGramGraphLoss"):
            training_manager = TrainingManager(
                HeightHistogramAndFourierLoss=HeightHistogramAndFourierLoss,
                **self.training.manager.to_dict(),
            )
        else:
            NGramGraphLoss = None
            if NGramGraphLoss_cache is not None:
                if NGramGraphLoss_cache.is_file():
                    NGramGraphLoss = self.NGramGraphLoss.type.from_pickle(
                        NGramGraphLoss_cache
                    )
                else:
                    NGramGraphLoss = self.NGramGraphLoss.type(surfaces=dataset.surfaces)
                    NGramGraphLoss.to_pickle(NGramGraphLoss_cache)
            else:
                if getattr(self.NGramGraphLoss, "type", None) is not None:
                    NGramGraphLoss = self.NGramGraphLoss.type(surfaces=dataset.surfaces)

            training_manager = TrainingManager(
                HeightHistogramAndFourierLoss=HeightHistogramAndFourierLoss,
                NGramGraphLoss=NGramGraphLoss,
                **self.training.manager.to_dict(),
            )

        (
            generator_losses,
            discriminator_losses,
            discriminator_output_reals,
            discriminator_output_fakes,
            NGramGraphLosses,
            HeightHistogramAndFourierLosses,
            fixed_fakes,
        ) = list(zip(*list(training_manager(generator, discriminator, dataset))))

        (
            save_path_gen_vs_dis_loss,
            save_path_dis_output,
            save_path_bce_vs_ngraph_loss,
            save_path_fourier_loss,
        ) = (None, None, None, None)
        
        if self.plot.against.save_path_fmt is not None:
            save_path_gen_vs_dis_loss = self.plot.against.save_path_fmt % (
                "gen_vs_dis_loss",
            )
            save_path_bce_vs_ngraph_loss = self.plot.against.save_path_fmt % (
                "bce_vs_ngraph_loss",
            )
            save_path_dis_output = self.plot.against.save_path_fmt % (
                "discriminator_output",
            )
            save_path_fourier_loss = self.plot.against.save_path_fmt % (
                "ngraph_vs_fourier_loss",
            )

        pd.DataFrame(
            data=np.array(
                [
                    generator_losses,
                    discriminator_losses,
                    discriminator_output_reals,
                    discriminator_output_fakes,
                    NGramGraphLosses,
                    HeightHistogramAndFourierLosses,
                ]
            ).T,
            columns=[
                "Generator Loss",
                "Discriminator Loss",
                "Discriminator Output (Real images)",
                "Discriminator Output (Generated/Fake images)",
                f"N-Gram Graph Loss ({self.NGramGraphLoss.type.__name__ if self.NGramGraphLoss.type else 'None'})",
                "Height Histogram and Fourier Loss",
            ],
        ).to_csv(str(checkpoint_dir / "per_epoch_data.csv"))

        plot_against(
            generator_losses,
            discriminator_losses,
            title="Generator and Discriminator loss \n" + str(path).split("Datasets")[1][1:len(str(path).split("Datasets")[1])],
            xlabel="Epochs",
            ylabel="Loss",
            labels=("G", "D"),
            save_path=self.plot.save_directory / save_path_gen_vs_dis_loss,
        )

        plot_against(
            discriminator_output_reals,
            discriminator_output_fakes,
            title="Discriminator Output \n" + str(path).split("Datasets")[1][1:len(str(path).split("Datasets")[1])],
            xlabel="Epochs",
            ylabel="Discriminator Output",
            labels=("Real images", "Generated images"),
            save_path=self.plot.save_directory / save_path_dis_output,
        )

        plot_against(
            generator_losses,
            NGramGraphLosses,
            title="Generator and N-gram graph loss \n" + str(path).split("Datasets")[1][1:len(str(path).split("Datasets")[1])],
            xlabel="Epochs",
            ylabel="Loss",
            labels=("BCE + NGG + HHF", "NGG"),
            save_path=self.plot.save_directory / save_path_bce_vs_ngraph_loss,
        )

        plot_against(
            HeightHistogramAndFourierLosses,
            NGramGraphLosses,
            title="Height Histogram and Fourier vs N-gram graph loss \n" + str(path).split("Datasets")[1][1:len(str(path).split("Datasets")[1])],
            xlabel="Epochs",
            ylabel="Loss",
            labels=("HHF", "NGG"),
            save_path=self.plot.save_directory / save_path_fourier_loss,
        )

        if self.plot.save_directory is not None:
            (self.plot.save_directory / self.plot.grayscale.save_path_fmt).parent.mkdir(
                parents=True, exist_ok=True
            )

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

                logger.info(
                    "Saved grayscale images on path: %s",
                    self.plot.save_directory / (self.plot.grayscale.save_path_fmt.split("/")[0])
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

                logger.info(
                    "Saved 3D surface images on path: %s",
                    self.plot.save_directory / (self.plot.surface.save_path_fmt.split("/")[0])
                )

        animate_epochs(
            fixed_fakes,
            indices=self.animation.indices,
            save_path=animation_save_path,
            **self.animation.parameters.to_dict(),
        )