#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


class TaskInfo:
    def __init__(self):
        self.frames = None  # frames of clips
        self.preproc_frames = None  # frames of clips after preprocessing
        self.id = -1
        # self.features = None
        self.clip_abn_score = -1  # of the whole video
        self.bboxes = None
        self.num_buffer_frames = 0
        self.img_height = -1
        self.img_width = -1

    def add_frames(self, idx, frames):
        """
        Add the clip and corresponding id.
        Args:
            idx (int): the current index of the clip.
            frames (list[ndarray]): list of images in "BGR" format.
        """
        self.frames = frames
        self.id = idx

    
    def add_preproc_frames(self, preproc_frames):
        """
        Add the clip after pre-process
        Args:
            idx (int): the current index of the clip.
        """
        self.preproc_frames = preproc_frames


    def add_clip_abnormal_score(self, clip_abn_score):
        """
        Add list of abnormal scores for each clip
        Args:
            features (list[ndarray]): list of clip's abnormal score.
        """
        self.clip_abn_score = clip_abn_score
