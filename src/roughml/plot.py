import IPython.display as ipyd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torchvision.utils as vutils


def plot_against(first, second, title="", xlabel="", ylabel="", labels=("", "")):
    x = list(range(max(len(first), len(second))))

    plt.plot(x, first, label=labels[0])
    plt.plot(x, second, label=labels[1])

    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xlim([min(x), max(x)])

    plt.legend()

    plt.show()


def as_grayscale_image(array, save_path=None):
    fig = px.imshow(array, color_continuous_scale="gray")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.update_layout(
        # title=title,
        autosize=True,
        width=500,
        height=500,
        # margin=dict(l=65, r=50, b=65, t=90)
    )

    if save_path is None:
        fig.show()
    else:
        with save_path.open("wb") as file:
            fig.write_image(file)


def as_3d_surface(array, save_path=False):
    fig = go.Figure(data=[go.Surface(z=array)])

    fig.update_layout(
        # title=title,
        autosize=False,
        width=500,
        height=500,
        # margin=dict(l=65, r=50, b=65, t=90)
    )

    if save_path is None:
        fig.show()
    else:
        with save_path.open("wb") as file:
            fig.write_image(file)


def plot_correlation(array):
    x, y = correlation(array)

    fig = px.line(
        # title="1-D height-height correlation function",
        # x="r(nm)", y="G(r) (nm)",
        x=x,
        y=y,
        log_x=True,
        log_y=True,
    )

    fig.update_layout(
        # title=title,
        autosize=True,
        width=500,
        height=500,
        # margin=dict(l=65, r=50, b=65, t=90)
    )

    fig.show()


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

    plt.close()

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
