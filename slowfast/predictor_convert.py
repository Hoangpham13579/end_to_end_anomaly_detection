#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import cv2, gc
import torch
import queue
import numpy as np
import torchvision.transforms as transforms

from slowfast.models import build_extractor, build_rtfm
from slowfast.models.rtfm import rtfm_pre_process, rtfm_post_process
from slowfast.utils import logging
from slowfast.utils.misc import gpu_mem_usage, cpu_mem_usage
import torch_tensorrt

logger = logging.get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtractor:
    """
    Extract feature from MP4 video inputs.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """
        self.cfg = cfg
        if cfg.NUM_GPUS:
            self.gpu_id = (
                torch.cuda.current_device() if gpu_id is None else gpu_id
            )

        # NOTE: Should apply transform in each 16-frames image group (as 1 clip)
        self.spatial_transform = transforms.Compose([
            # Convert to Tensor
            transforms.ToTensor(),
            # Resize
            transforms.Resize(self.cfg.DATA.TEST_CROP_SIZE),
            # 10-crop augmentation
            transforms.TenCrop(224),
            # Convert to Tensor
            lambda crops: torch.stack([crop * 255 for crop in crops]),
            # Normalization
            transforms.Normalize(self.cfg.DATA.MEAN, self.cfg.DATA.STD),
        ])

        # Build the feature extractor model and load checkpoint.
        self.extractor = build_extractor(cfg, gpu_id=gpu_id)
        self.extractor.eval()

        # Feature Extractor
        logger.info("Start loading Feature Extractor weights.")
        extractor_weight = torch.load(cfg.DATA.FEATURE_EXTRACTOR_CKPT, map_location="cpu")
        self.extractor.load_state_dict(extractor_weight)
        logger.info("Finish loading Feature Extractor weights")

    def __call__(self, task):
        """
        Returns the prediction results for the current task.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        buffer = self.cfg.DEMO.BUFFER_SIZE // self.cfg.DATA.NUM_FRAMES
        clip_feats = [[] for i in range(self.cfg.DATA.SAMPLING_RATE - buffer)]
        frames = task.frames[task.num_buffer_frames:]

        for i in range(self.cfg.DATA.SAMPLING_RATE - buffer):
            clip = frames[i*self.cfg.DATA.NUM_FRAMES : (i+1)*self.cfg.DATA.NUM_FRAMES]
            if len(clip) != self.cfg.DATA.NUM_FRAMES:
                break

            if (self.cfg.DEMO.INPUT_FORMAT == "BGR") and (not self.cfg.MODEL.WARMUP):
                clip = [
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in clip
                ]

            # FRAME TRANSFORMATION
            clip = [self.spatial_transform(frame) for frame in clip]
            clip = torch.stack(clip, 0).permute(1, 2, 0, 3, 4)  # 10x3x16x224x224
            clip = clip.half()

            with torch.no_grad():
                # FEATURE EXTRACTOR
                for j in range(10):
                    # input shape: clips_num,c,t,h,w  # 24x3x16x224x224
                    feat = self.extractor(torch.unsqueeze(clip[j, :, :, :, :], dim=0).to(self.gpu_id, non_blocking=True))
                    clip_feats[i].append(feat)
                    del feat
                clip_feats[i] = torch.stack(clip_feats[i], 0)

        # Post process feature extraction
        clip_feats = torch.squeeze(torch.stack(clip_feats, 0))
        clip_feats = torch.unsqueeze(clip_feats, 0)  # 1x10x2048
        clip_feats = clip_feats.permute(0, 2, 1, 3).half()  # 1x10x2048x1

        # (NOTE) Delete clip_feats, clip not help with Memory usage

        return task , clip_feats


class AnomalyDetector:
    """
    Synchronous feature extractor.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            async_vis (AsyncVis object): asynchronous visualizer.
            gpu_id (Optional[int]): GPU id.
        """
        self.cfg = cfg
        if cfg.NUM_GPUS:
            self.gpu_id = (
                torch.cuda.current_device() if gpu_id is None else gpu_id
            )

        # RTFM Anomaly Detection model.
        self.rtfm = build_rtfm(cfg)  # Pre-process and Post-process
        self.rtfm.eval()

        logger.info("Start loading RTFM weights.")
        rtfm_weight = torch.load(cfg.MODEL.WEIGHT, map_location="cpu")
        self.rtfm.load_state_dict(rtfm_weight)
        logger.info("Finish loading RTFM weights")

    def __call__(self, task, features):
        """
        Perform anomaly detection and return the task object.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        """
        with torch.no_grad():
            features, inputs = rtfm_pre_process(inputs=features)
            scores = self.rtfm(features.to(self.gpu_id, non_blocking=True))
            logits = rtfm_post_process(scores, inputs)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            task.add_clip_abnormal_score(logits.cpu().numpy())

        return task


class DetectionPipeline:
    """
    Synchronous Action Prediction and Visualization pipeline with AsyncVis.
    """

    def __init__(self, cfg, async_vis=None, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            async_vis (AsyncVis object): asynchronous visualizer.
            gpu_id (Optional[int]): GPU id.
        """
        if cfg.NUM_GPUS:
            self.gpu_id = (
                torch.cuda.current_device() if gpu_id is None else gpu_id
            )

        self.i3d = FeatureExtractor(cfg=cfg, gpu_id=gpu_id)
        self.rtfm = AnomalyDetector(cfg=cfg, gpu_id=gpu_id)
        self.async_vis = async_vis

    def put(self, task):
        """
        Make prediction and put the results in `async_vis` task queue.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        """
        # (NOTE) task.frames: list of 16-frames (np.ndarray) with shape (256, 340, 3) _ (H, W, C) each
        # Feature Extraction
        task, features = self.i3d(task)
        # Anomaly Detection
        task = self.rtfm(task, features)
        del features

        # Visualize
        self.async_vis.get_indices_ls.append(task.id)
        self.async_vis.put(task)

    def get(self):
        """
        Get the visualized clips if any.
        """
        try:
            task = self.async_vis.get()
        except (queue.Empty, IndexError):
            raise IndexError("Results are not available yet.")

        return task