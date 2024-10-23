import torch
import numpy as np

from torch_dimcheck import dimchecked
from unets import Unet, thin_setup

from disk import NpArray, Features
from disk.model.detector import Detector

DEFAULT_SETUP = {**thin_setup, 'bias': True, 'padding': True}

class DISK(torch.nn.Module):
    def __init__(
        self,
        desc_dim=128,
        window=8,
        setup=DEFAULT_SETUP,
        kernel_size=5,
        backbone="unet",
    ):
        super(DISK, self).__init__()

        self.desc_dim = desc_dim
        self.backbone = backbone
        if backbone == "unet":
            self.model = Unet(
                in_features=3, size=kernel_size,
                down=[16, 32, 64, 64, 64],
                up=[64, 64, 64, desc_dim+1],
                setup=setup,
            )
        elif backbone == "dust3r":
            from disk.model.dust3r import DUSt3R
            self.model = DUSt3R()
        elif backbone == "mickey":
            from disk.model.mickey import MicKey
            self.model = MicKey()
        else:
            raise ValueError("backbone type not implemented")

        self.detector = Detector(window=window)

    @dimchecked
    def _split(self, unet_output: ['B', 'C', 'H', 'W']) \
                -> (['B', 'C-1', 'H', 'W'], ['B', 1, 'H', 'W']):
        '''
        Splits the raw Unet output into descriptors and detection heatmap.
        '''
        assert unet_output.shape[1] == self.desc_dim + 1

        descriptors = unet_output[:, :self.desc_dim]
        heatmap     = unet_output[:, self.desc_dim:]

        return descriptors, heatmap

    @dimchecked
    def features(
        self,
        images: ['B', 'C', 'H', 'W'],
        true_shapes: ['B', '2'],
        kind='rng',
        **kwargs
    ) -> (..., ['B', '1', 'H', 'W']):
        '''
            true_shape: real image shape before resizing/padding
            allowed values for `kind`:
            * rng
            * nms
        '''

        B = images.shape[0]
        try:
            if self.backbone in ["unet"]:
                model_output = self.model(images)
            else:
                model_output = self.model(images, true_shapes)
            descriptors, heatmaps = self._split(model_output)
        except RuntimeError as e:
            if 'Trying to downsample' in str(e):
                msg = ('U-Net failed because the input is of wrong shape. With '
                       'a n-step U-Net (n == 4 by default), input images have '
                       'to have height and width as multiples of 2^n (16 by '
                       'default).')
                raise RuntimeError(msg) from e
            else:
                raise

        keypoints = {
            'rng': self.detector.sample,
            'nms': self.detector.nms,
        }[kind](heatmaps, **kwargs)

        features = []
        for i in range(B):
            features.append(keypoints[i].merge_with_descriptors(descriptors[i]))

        return np.array(features, dtype=object), heatmaps

    # @dimchecked
    # def features_multi(
    #     self,
    #     images1: ['B', 'C', 'H', 'W'],
    #     images2: ['B', 'C', 'H', 'W'],
    #     true_shapes1,
    #     true_shapes2,
    #     kind='rng',
    #     **kwargs
    # ) -> NpArray[Features]:
    #     '''
    #         true_shape: real image shape before resizing/padding
    #         allowed values for `kind`:
    #         * rng
    #         * nms
    #     '''
    #
    #     B = images.shape[0]
    #     try:
    #         if self.backbone in ["unet"]:
    #             model_output = self.model(images)
    #         else:
    #             model_output = self.model(images, true_shapes)
    #         descriptors, heatmaps = self._split(model_output)
    #     except RuntimeError as e:
    #         if 'Trying to downsample' in str(e):
    #             msg = ('U-Net failed because the input is of wrong shape. With '
    #                    'a n-step U-Net (n == 4 by default), input images have '
    #                    'to have height and width as multiples of 2^n (16 by '
    #                    'default).')
    #             raise RuntimeError(msg) from e
    #         else:
    #             raise
    #
    #     keypoints = {
    #         'rng': self.detector.sample,
    #         'nms': self.detector.nms,
    #     }[kind](heatmaps, **kwargs)
    #
    #     features = []
    #     for i in range(B):
    #         features.append(keypoints[i].merge_with_descriptors(descriptors[i]))
    #
    #     return np.array(features, dtype=object), heatmaps
