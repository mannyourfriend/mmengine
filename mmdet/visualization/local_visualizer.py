# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cv2
import mmcv

try:
    import seaborn as sns
except ImportError:
    sns = None
import torch
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData
from mmengine.visualization import Visualizer

from ..evaluation import INSTANCE_OFFSET
from ..registry import VISUALIZERS
from ..structures import DetDataSample
from ..structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon
from .palette import _get_adaptive_scales, get_palette, jitter_color


@VISUALIZERS.register_module()
class DetLocalVisualizer(Visualizer):
    """
    MMDetection Local Visualizer modified to produce a three-panel output:
    1. Original Image
    2. Image with Bounding Boxes and Labels
    3. Image with Segmentation Masks
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 idToName: Optional[Dict] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (200, 200, 200),
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8) -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir)
        self.idToName = idToName
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.mask_color = mask_color
        self.line_width = line_width
        self.alpha = alpha
        # Set default value. When calling
        # `DetLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}

    def _get_distinct_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generates n distinct colors from a colormap, in BGR format."""
        cmap = cm.get_cmap('tab10', n)
        colors = []
        for i in range(n):
            rgb = tuple(int(255 * c) for c in cmap(i)[:3])
            colors.append(rgb[::-1])  # Convert RGB to BGR for OpenCV
        return colors

    def _draw_bboxes_with_labels(self, image: np.ndarray, instances: InstanceData,
                                 classes: Optional[List[str]]) -> np.ndarray:
        """Draws bounding boxes and labels with scores on the image."""
        self.set_image(image)
        
        if not hasattr(instances, 'bboxes') or instances.bboxes.numel() == 0:
            return self.get_image()

        bboxes = instances.bboxes
        labels = instances.labels
        scores = instances.scores if 'scores' in instances else None
        num_instances = len(labels)
        
        colors = self._get_distinct_colors(num_instances)

        # Draw bounding boxes
        self.draw_bboxes(
            bboxes,
            edge_colors=colors,
            alpha=1.0,  # Solid outlines
            line_widths=self.line_width)

        # Prepare to draw text labels
        positions = bboxes[:, :2] + self.line_width
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas.cpu().numpy())

        for i, pos in enumerate(positions):
            label_id = labels[i]
            label_text = classes[label_id] if classes else f'class {label_id}'
            
            if scores is not None:
                score_text = f'{float(scores[i]) * 100:.1f}%'
                label_text = f'{label_text}: {score_text}'

            self.draw_texts(
                label_text,
                pos,
                colors=self.text_color,
                font_sizes=int(13 * scales[i]),
                bboxes=[{
                    'facecolor': 'black',
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                }])
        return self.get_image()

    def _draw_filled_masks(self, image: np.ndarray, instances: InstanceData) -> np.ndarray:
        """Draws filled, semi-transparent segmentation masks on the image."""
        self.set_image(image)
        
        if not hasattr(instances, 'masks') or instances.masks.numel() == 0:
            return self.get_image()
            
        masks = instances.masks.cpu().numpy()
        num_instances = len(instances.labels)
        colors = self._get_distinct_colors(num_instances)
        
        # Use the base visualizer's method to draw filled masks
        self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)

        return self.get_image()

    @master_only
    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: Optional['DetDataSample'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       pred_score_thr: float = 0.3,
                       step: int = 0) -> None:
        """
        Draws a three-panel visualization (Original, BBoxes, Masks) and
        saves it to backends.
        """
        image = image.clip(0, 255).astype(np.uint8)
        classes = self.dataset_meta.get('classes', None)

        pred_instances = None
        if draw_pred and data_sample and hasattr(data_sample, 'pred_instances'):
            pred_instances = data_sample.pred_instances.cpu()
            if 'scores' in pred_instances:
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr]

        # Panel 1: Original Image
        panel_original = image.copy()

        # If no predictions, the final image is just the original
        if pred_instances is None or len(pred_instances) == 0:
            drawn_img = panel_original
        else:
            # Panel 2: Image with Bounding Boxes and Labels
            panel_bboxes = self._draw_bboxes_with_labels(
                image.copy(), pred_instances, classes)

            # Panel 3: Image with Segmentation Masks
            panel_masks = self._draw_filled_masks(image.copy(),
                                                  pred_instances)

            # Ensure all panels have the same height before concatenation
            target_height = panel_original.shape[0]
            if panel_bboxes.shape[0] != target_height:
                panel_bboxes = cv2.resize(
                    panel_bboxes, (panel_bboxes.shape[1], target_height))
            if panel_masks.shape[0] != target_height:
                panel_masks = cv2.resize(
                    panel_masks, (panel_masks.shape[1], target_height))

            # Stitch the three panels together horizontally
            drawn_img = np.concatenate(
                (panel_original, panel_bboxes, panel_masks), axis=1)

        # Save or show the final composite image
        self.set_image(drawn_img)
        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)
        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)


def random_color(seed):
    """Random a color according to the input seed."""
    if sns is None:
        raise RuntimeError('motmetrics is not installed,\
                           please install it by: pip install seaborn')
    np.random.seed(seed)
    colors = sns.color_palette()
    color = colors[np.random.choice(range(len(colors)))]
    color = tuple([int(255 * c) for c in color])
    return color


@VISUALIZERS.register_module()
class TrackLocalVisualizer(Visualizer):
    """Tracking Local Visualizer for the MOT, VIS tasks.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        alpha (int, float): The transparency of bboxes or mask.
                Defaults to 0.8.
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8) -> None:
        super().__init__(name, image, vis_backends, save_dir)
        self.line_width = line_width
        self.alpha = alpha
        # Set default value. When calling
        # `TrackLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}

    def _draw_instances(self, image: np.ndarray,
                        instances: InstanceData) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)
        classes = self.dataset_meta.get('classes', None)

        # get colors and texts
        # for the MOT and VIS tasks
        colors = [random_color(_id) for _id in instances.instances_id]
        categories = [
            classes[label] if classes is not None else f'cls{label}'
            for label in instances.labels
        ]
        if 'scores' in instances:
            texts = [
                f'{category_name}\n{instance_id} | {score:.2f}'
                for category_name, instance_id, score in zip(
                    categories, instances.instances_id, instances.scores)
            ]
        else:
            texts = [
                f'{category_name}\n{instance_id}' for category_name,
                instance_id in zip(categories, instances.instances_id)
            ]

        # draw bboxes and texts
        if 'bboxes' in instances:
            # draw bboxes
            bboxes = instances.bboxes.clone()
            self.draw_bboxes(
                bboxes,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=self.line_width)
            # draw texts
            if texts is not None:
                positions = bboxes[:, :2] + self.line_width
                areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                    bboxes[:, 2] - bboxes[:, 0])
                scales = _get_adaptive_scales(areas.cpu().numpy())
                for i, pos in enumerate(positions):
                    self.draw_texts(
                        texts[i],
                        pos,
                        colors='black',
                        font_sizes=int(13 * scales[i]),
                        bboxes=[{
                            'facecolor': [c / 255 for c in colors[i]],
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }])

        # draw masks
        if 'masks' in instances:
            masks = instances.masks
            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)
            self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)

        return self.get_image()

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: DetDataSample = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: int = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.
        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (OptTrackSampleList): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT TrackDataSample.
                Default to True.
            draw_pred (bool): Whether to draw Prediction TrackDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (int): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            assert 'gt_instances' in data_sample
            gt_img_data = self._draw_instances(image, data_sample.gt_instances)

        if draw_pred and data_sample is not None:
            assert 'pred_track_instances' in data_sample
            pred_instances = data_sample.pred_track_instances
            if 'scores' in pred_instances:
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr].cpu()
            pred_img_data = self._draw_instances(image, pred_instances)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)