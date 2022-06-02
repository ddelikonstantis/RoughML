import os

import logging
from xml.dom.pulldom import parseString
import IPython.display as ipyd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction import img_to_graph
import torchvision.utils as vutils
from PIL import Image

logger = logging.getLogger(__name__)


def plot_against(
    first, second, third=None, title="", xlabel="", ylabel="", labels=("", "", ""), save_path=None
):
    if third is not None:
        x = list(range(max(len(first), len(second), len(third))))
    else:
        x = list(range(max(len(first), len(second))))

    plt.plot(x, first, label=labels[0])
    plt.plot(x, second, label=labels[1])
    if third is not None: 
        plt.plot(x, third, label=labels[2])

    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xlim([min(x), max(x)])

    plt.legend()

    if save_path is None:
        plt.show()
    else:
        with save_path.open("wb") as file:
            plt.savefig(file, bbox_inches="tight")

        plt.close()

        logger.info(
            "Saved plot with title: %s on path: %s",
            title,
            save_path
        )


def as_grayscale_image(array, save_path=None):
    # convert tensor pixel values from [0 , 1] to [0, 255]
    array = (((array - array.min()) / (array.max() - array.min())) * 255.9)
    # convert the pixels from float type to int type
    array = np.array(array, dtype=np.uint8)
    # convert to image from array
    img = Image.fromarray(array)
    # save image
    if save_path is None:
        img.show()
    else:
        with save_path.open("wb") as file:
            img.save(file)


def as_3d_surface(array, save_path=False):
    fig = go.Figure(data=[go.Surface(z=array)])

    if save_path is None:
        fig.show()
    else:
        with save_path.open("wb") as file:
            fig.write_image(file)


def animate_epochs(batches_of_tensors, indices=None, save_path=None, **kwargs):
    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111)

    plt.axis("off")

    artists = []
    for index, batch_of_tensors in enumerate(batches_of_tensors, start=1):
        if indices:
            batch_of_tensors = [batch_of_tensors[index] for index in indices]

        grid = vutils.make_grid(batch_of_tensors, padding=2, normalize=True)

        artists.append(
            [
                plt.imshow(
                    np.transpose(grid, (1, 2, 0)),
                    animated=True,
                ),
                plt.text(
                    0.5,
                    1.01,
                    "Epoch %02d" % (index,),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    transform=axes.transAxes,
                    fontsize=16,
                ),
            ]
        )

    if os.name != "nt":
        plt.close()

    try:
        ani = animation.ArtistAnimation(
            fig,
            artists,
            interval=kwargs.get("interval", 1000),
            repeat_delay=kwargs.get("repeat_delay", 1000),
            blit=kwargs.get("blit", True),
        )

        ipyd.display(ipyd.HTML(ani.to_jshtml()))

        if save_path is not None:
            # Set up formatting for the movie files
            Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=kwargs.get("fps", 15), bitrate=kwargs.get("bitrate", 1800))

            ani.save(str(save_path), writer=writer)
            
    except Exception as execErr:
        logger.info(
            "OS error: %s",
            execErr
        )


def dataset_title_plot(dataset_path):
    # function that extracts dataset title from path and uses title for plotting information
    # example dataset title 'dataset_1000_128_03_00_03_04_04_1.00.pt'
    # extract undrescores in dataset name and rename according to attribute for plotting
    # example output dataset title for plotting 'dataset_1000imgs_128dim_03rms_00skew_03kurt_04corrX_04corrY_1.00alpha.pt'
    txt = str(dataset_path).split("Datasets")[1][1:len(str(dataset_path).split("Datasets")[1])] # extract dataset title from path
    attrs = ["imgs", "dim", "rms", "skew", "kurt", "corrX", "corrY"] # attribute list to be added to title
    char_index_list = []
    char_index = 0
    while char_index < len(txt):
        char_index = txt.find('_', char_index)
        if char_index == -1: # end of string or character not found
            del char_index_list[0] # ignore first underscore in title
            temp4 = 0
            for idx, char in enumerate(char_index_list):
                txt = txt[:char + temp4] + attrs[idx] + txt[char + temp4:]
                char_index_list = [x + (len(attrs[idx])+(char_index_list[idx+1]-char_index_list[idx])) for x in char_index_list] # update list with newly added string length
                temp4 = len(attrs[idx]) #+ (char_index_list[idx+1]-char_index_list[idx]) # update index with newly added string length
            break
        char_index_list.append(char_index)
        char_index += 1

    return txt