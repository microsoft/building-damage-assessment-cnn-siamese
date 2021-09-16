# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Class to visualize raster mask labels and hardmax or softmax model predictions, for semantic segmentation tasks.
"""

import json
import os
from io import BytesIO
from typing import Union, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageColor


class RasterLabelVisualizer(object):
    """Visualizes raster mask labels and predictions."""

    def __init__(self, label_map: Union[str, dict]):
        """Constructs a raster label visualizer.

        Args:
            label_map: a path to a JSON file containing a dict, or a dict. The dict needs to have two fields:

            num_to_name {
                numerical category (str or int) : display name (str)
            }

            num_to_color {
                numerical category (str or int) : color representation (an object that matplotlib.colors recognizes
                as a color; additionally a (R, G, B) tuple or list with uint8 values will also be parsed)
            }
        """
        if isinstance(label_map, str):
            assert os.path.exists(label_map)
            with open(label_map) as f:
                label_map = json.load(f)

        assert 'num_to_name' in label_map
        assert isinstance(label_map['num_to_name'], dict)
        assert 'num_to_color' in label_map
        assert isinstance(label_map['num_to_color'], dict)

        self.num_to_name = RasterLabelVisualizer._dict_key_to_int(label_map['num_to_name'])
        self.num_to_color = RasterLabelVisualizer._dict_key_to_int(label_map['num_to_color'])

        assert len(self.num_to_color) == len(self.num_to_name)
        self.num_classes = len(self.num_to_name)

        # check for duplicate names or colors
        assert len(set(self.num_to_color.values())) == self.num_classes, 'There are duplicate colors in the colormap'
        assert len(set(self.num_to_name.values())) == self.num_classes, \
            'There are duplicate class names in the colormap'

        self.num_to_color = RasterLabelVisualizer.standardize_colors(self.num_to_color)

        # create the custom colormap according to colors defined in label_map
        required_colors = []
        # key is originally a string
        for num, color_name in sorted(self.num_to_color.items(), key=lambda x: x[0]):  # num already cast to int
            rgb = mcolors.to_rgb(mcolors.CSS4_COLORS[color_name])
            # mcolors.to_rgb is to [0, 1] values; ImageColor.getrgb gets [1, 255] values
            required_colors.append(rgb)

        self.colormap = mcolors.ListedColormap(required_colors)
        # vmin and vmax appear to be inclusive,
        # so if there are a total of 34 classes, class 0 to class 33 each maps to a color
        self.normalizer = mcolors.Normalize(vmin=0, vmax=self.num_classes - 1)

        self.color_matrix = self._make_color_matrix()

    @staticmethod
    def _dict_key_to_int(d: dict) -> dict:
        return {int(k): v for k, v in d.items()}

    def _make_color_matrix(self) -> np.ndarray:
        """Creates a color matrix of dims (num_classes, 3), where a row corresponds to the RGB values of each class.
        """
        matrix = []
        for num, color in sorted(self.num_to_color.items(), key=lambda x: x[0]):
            rgb = RasterLabelVisualizer.matplotlib_color_to_uint8_rgb(color)
            matrix.append(rgb)
        matrix = np.array(matrix)

        assert matrix.shape == (self.num_classes, 3)

        return matrix

    @staticmethod
    def standardize_colors(num_to_color: dict) -> dict:
        """Return a new dict num_to_color with colors verified. uint8 RGB tuples are converted to a hex string
        as matplotlib.colors do not accepted uint8 intensity values"""
        new = {}
        for num, color in num_to_color.items():
            if mcolors.is_color_like(color):
                new[num] = color
            else:
                # try to see if it's a (r, g, b) tuple or list of uint8 values
                assert len(color) == 3 or len(
                    color) == 4, f'Color {color} is specified as a tuple or list but is not of length 3 or 4'
                for c in color:
                    assert isinstance(c, int) and 0 < c < 256, f'RGB value {c} is out of range'

                new[num] = RasterLabelVisualizer.uint8_rgb_to_hex(color[0], color[1], color[3])  # drop any alpha values
        assert len(new) == len(num_to_color)
        return new

    @staticmethod
    def uint8_rgb_to_hex(r: int, g: int, b: int) -> str:
        """Convert RGB values in uint8 to a hex color string

        Reference
        https://codereview.stackexchange.com/questions/229282/performance-for-simple-code-that-converts-a-rgb-tuple-to-hex-string
        """
        return f'#{r:02x}{g:02x}{b:02x}'

    @staticmethod
    def matplotlib_color_to_uint8_rgb(color: Union[str, tuple, list]) -> Tuple[int, int, int]:
        """Converts any matplotlib recognized color representation to (R, G, B) uint intensity values

        Need to use matplotlib, which recognizes different color formats, to convert to hex,
        then use PIL to convert to uint8 RGB. matplotlib does not support the uint8 RGB format
        """
        color_hex = mcolors.to_hex(color)
        color_rgb = ImageColor.getcolor(color_hex, 'RGB')  # '#DDA0DD' to (221, 160, 221); alpha silently dropped
        return color_rgb

    def get_tiff_colormap(self) -> dict:
        """Returns the object to pass to rasterio dataset object's write_colormap() function,
        which is a dict mapping int values to a tuple of (R, G, B)

        See https://rasterio.readthedocs.io/en/latest/topics/color.html for writing the TIFF colormap
        """
        colormap = {}
        for num, color in self.num_to_color.items():
            # uint8 RGB required by TIFF
            colormap[num] = RasterLabelVisualizer.matplotlib_color_to_uint8_rgb(color)
        return colormap

    def get_tool_colormap(self) -> str:
        """Returns a string that is a JSON of a list of items specifying the name and color
        of classes. Example:
        "[
            {"name": "Water", "color": "#0000FF"},
            {"name": "Tree Canopy", "color": "#008000"},
            {"name": "Field", "color": "#80FF80"},
            {"name": "Built", "color": "#806060"}
        ]"
        """
        classes = []
        for num, name in sorted(self.num_to_name.items(), key=lambda x: int(x[0])):
            color = self.num_to_color[num]
            color_hex = mcolors.to_hex(color)
            classes.append({
                'name': name,
                'color': color_hex
            })
        classes = json.dumps(classes, indent=4)
        return classes

    @staticmethod
    def plot_colortable(name_to_color: dict, title: str, sort_colors: bool = False, emptycols: int = 0) -> plt.Figure:
        """
        function taken from https://matplotlib.org/3.1.0/gallery/color/named_colors.html
        """

        cell_width = 212
        cell_height = 22
        swatch_width = 70
        margin = 12
        topmargin = 40

        # Sort name_to_color by hue, saturation, value and name.
        if sort_colors is True:
            by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                             name)
                            for name, color in name_to_color.items())
            names = [name for hsv, name in by_hsv]
        else:
            names = list(name_to_color)

        n = len(names)
        ncols = 4 - emptycols
        nrows = n // ncols + int(n % ncols > 0)

        width = cell_width * 4 + 2 * margin
        height = cell_height * nrows + margin + topmargin
        dpi = 80  # other numbers don't seem to work well

        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.subplots_adjust(margin / width, margin / height,
                            (width - margin) / width, (height - topmargin) / height)
        ax.set_xlim(0, cell_width * 4)
        ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_axis_off()
        ax.set_title(title, fontsize=24, loc='left', pad=10)

        for i, name in enumerate(names):
            row = i % nrows
            col = i // nrows
            y = row * cell_height

            swatch_start_x = cell_width * col
            swatch_end_x = cell_width * col + swatch_width
            text_pos_x = cell_width * col + swatch_width + 7

            ax.text(text_pos_x, y, name, fontsize=14,
                    horizontalalignment='left',
                    verticalalignment='center')

            ax.hlines(y, swatch_start_x, swatch_end_x,
                      color=name_to_color[name], linewidth=18)

        return fig

    def plot_color_legend(self, legend_title: str = 'Categories') -> plt.Figure:
        """Builds a legend of color block, numerical categories and names of the categories.

        Returns:
            a matplotlib.pyplot Figure
        """
        label_map = {}
        for num, color in self.num_to_color.items():
            label_map['{} {}'.format(num, self.num_to_name[num])] = color

        fig = RasterLabelVisualizer.plot_colortable(label_map, legend_title, sort_colors=False, emptycols=3)
        return fig

    def show_label_raster(self, label_raster: Union[Image.Image, np.ndarray],
                          size: Tuple[int, int] = (10, 10)) -> Tuple[Image.Image, BytesIO]:
        """Visualizes a label mask or hardmax predictions of a model, according to the category color map
        provided when the class was initialized.

        The label_raster provided needs to contain values in [0, num_classes].

        Args:
            label_raster: 2D numpy array or PIL Image where each number indicates the pixel's class
            size: matplotlib size in inches (h, w)

        Returns:
            (im, buf) - PIL image of the matplotlib figure, and a BytesIO buf containing the matplotlib Figure
            saved as a PNG
        """
        if not isinstance(label_raster, np.ndarray):
            label_raster = np.asarray(label_raster)

        label_raster = label_raster.squeeze()
        assert len(label_raster.shape) == 2, 'label_raster provided has more than 2 dimensions after squeezing'

        label_raster.astype(np.uint8)

        # min of 0, which is usually empty / no label
        assert np.min(label_raster) >= 0, f'Invalid value for class label: {np.min(label_raster)}'

        # non-empty, actual class labels start at 1
        assert np.max(label_raster) <= self.num_classes, f'Invalid value for class label: {np.max(label_raster)}'

        _ = plt.figure(figsize=size)
        _ = plt.imshow(label_raster, cmap=self.colormap, norm=self.normalizer, interpolation='none')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        im = Image.open(buf)
        return im, buf

    @staticmethod
    def visualize_matrix(matrix: np.ndarray) -> Image.Image:
        """Shows a 2D matrix of RGB or greyscale values as a PIL Image.

        Args:
            matrix: a (H, W, 3) or (H, W) numpy array, representing a colored or greyscale image

        Returns:
            a PIL Image object
        """
        assert len(matrix.shape) in [2, 3]

        image = Image.fromarray(matrix)
        return image

    def visualize_softmax_predictions(self, softmax_preds: np.ndarray) -> np.ndarray:
        """Visualizes softmax probabilities in RGB according to the class label's assigned colors

        Args:
            softmax_preds: numpy array of dimensions (batch_size, num_classes, H, W) or (num_classes, H, W)

        Returns:
            numpy array of size ((batch_size), H, W, 3). You may need to roll the last axis to in-front before
            writing to TIFF

        Raises:
            ValueError when the dimension of softmax_preds is not compliant
        """

        assert len(softmax_preds.shape) == 4 or len(softmax_preds.shape) == 3

        # row the num_classes dimension to the end
        if len(softmax_preds.shape) == 4:
            assert softmax_preds.shape[1] == self.num_classes
            softmax_preds_transposed = np.transpose(softmax_preds, axes=(0, 2, 3, 1))
        elif len(softmax_preds.shape) == 3:
            assert softmax_preds.shape[0] == self.num_classes
            softmax_preds_transposed = np.transpose(softmax_preds, axes=(1, 2, 0))
        else:
            raise ValueError('softmax_preds does not have the required length in the dimension of the classes')

        # ((batch_size), H, W, num_classes) @ (num_classes * 3) = ((batch_size), H, W, 3)
        colored_view = softmax_preds_transposed @ self.color_matrix
        return colored_view
