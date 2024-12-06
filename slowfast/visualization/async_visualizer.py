#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import atexit
import numpy as np
import torch.multiprocessing as mp
try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass
import time
import cv2
import queue

import slowfast.utils.logging as logging
logger = logging.get_logger(__name__)

class AsyncVis:
    class _VisWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            """
            Visualization Worker for AsyncVis.
            Args:
                task_queue (mp.Queue): a shared queue for incoming task for visualization.
                result_queue (mp.Queue): a shared queue for visualized results.
            """
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            self.task = None
            super().__init__()

        def run(self):
            """
            Draw prediction, Show notification & call API asynchronously.
            """
            while True:
                task = self.task_queue.get()
                if isinstance(task, _StopToken):
                    break

                # Draw Abnomal score on visualization frames
                frames = draw_predictions(task, self.cfg)
                task.frames = np.array(frames)
                logger.info(f' Prediction score:\n {np.round(task.clip_abn_score, 4)}')

                for abn_val in task.clip_abn_score:
                    if abn_val > self.cfg.DEMO.ANOMALY_THRESHOLD:
                        self.task = task
                        break

                self.result_queue.put(task)


    def __init__(self, cfg, n_workers=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            n_workers (Optional[int]): number of CPUs for running video visualizer.
                If not given, use all CPUs.
        """

        num_workers = mp.cpu_count() if n_workers is None else n_workers

        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.get_indices_ls = []
        self.procs = []
        self.result_data = {}
        self.put_id = -1
        for _ in range(max(num_workers, 1)):
            self.procs.append(
                AsyncVis._VisWorker(
                    cfg, self.task_queue, self.result_queue
                )
            )

        for p in self.procs:
            p.start()

        atexit.register(self.shutdown)

    def put(self, task):
        """
        Add the new task to task queue.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes, predictions)
        """
        self.put_id += 1
        self.task_queue.put(task)

    def get(self):
        """
        Return visualized frames/clips in the correct order based on task id if 
        result(s) is available. Otherwise, raise queue.Empty exception.
        """
        get_idx = self.get_indices_ls[0]
        if self.result_data.get(get_idx) is not None:
            res = self.result_data[get_idx]
            del self.result_data[get_idx]
            del self.get_indices_ls[0]
            return res

        while True:
            res = self.result_queue.get(block=True)
            idx = res.id
            if idx == get_idx:
                del self.get_indices_ls[0]
                return res
            self.result_data[idx] = res

    def __call__(self, task):
        """
        How many results are ready to be returned.
        """
        self.put(task)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(_StopToken())

    @property
    def result_available(self):
        return self.result_queue.qsize() + len(self.result_data)

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5


class _StopToken:
    pass


def draw_predictions(task, cfg):
    """
    Draw prediction for the given task.
    Args:
        task (TaskInfo object): task object that contain
            the necessary information for visualization. (e.g. frames, preds)
            All attributes must lie on CPU devices.
    """
    # Abnormal score & frames of each clip
    # (NOTE) "frames" list contains both buffer frames and detection frames
    frames = task.frames
    buffer = frames[: task.num_buffer_frames]
    frames = frames[task.num_buffer_frames :]

    abn_scores = list()
    for abn_val in task.clip_abn_score:
        abn_scores.append(np.array(abn_val).repeat(16))
    abn_scores = np.hstack(abn_scores)

    # DRAW ON FRAMES
    for frame, abn_score in zip(frames, abn_scores):
        if cfg.DEMO.WEBCAM != -1:  # Case WEB_CAM
            frame = cv2.putText(frame, ' Pred :' + str(round(abn_score, 3)), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 200, 240), 2)
                
            # Draw red rectangle if abnormal score is higher than threshold
            if abn_score > cfg.DEMO.ANOMALY_THRESHOLD:
                frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 3)  # w, h
                
        else:  # Case INPUT_VIDEO
            frame = cv2.putText(frame, ' Pred :' + str(round(abn_score, 3)), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 200, 240), 2)
                
            # Draw red rectangle if abnormal score is higher than threshold
            if abn_score > cfg.DEMO.ANOMALY_THRESHOLD:
                frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 3)
    del task, abn_scores

    return buffer + frames


def CallAPIAbnormalEvent():
    pass


