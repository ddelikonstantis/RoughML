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


def as_grayscale_image(array):
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

    fig.show()


def as_3d_surface(array, autosize=False):
    fig = go.Figure(data=[go.Surface(z=array)])

    fig.update_layout(
        # title=title,
        autosize=autosize,
        width=500,
        height=500,
        # margin=dict(l=65, r=50, b=65, t=90)
    )

    fig.show()


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


def animate_epochs(images):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")

    artists = []
    for image in images:
        grid = vutils.make_grid(image, padding=2, normalize=True)

        artists.append(
            [
                plt.imshow(
                    np.transpose(grid, (1, 2, 0)),
                    animated=True,
                )
            ]
        )

    plt.close()

    ani = animation.ArtistAnimation(
        fig, artists, interval=1000, repeat_delay=1000, blit=True
    )

    ipyd.display(ipyd.HTML(ani.to_jshtml()))
