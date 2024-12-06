#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import logging
import numpy as np
import os, gc
import psutil
import torch
from torch import nn

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

logger = logging.get_logger(__name__)


def params_count(model, ignore_bn=False):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters()]).item()
    else:
        count = 0
        for m in model.modules():
            if not isinstance(m, nn.BatchNorm3d):
                for p in m.parameters(recurse=False):
                    count += p.numel()
    return count


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024**3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    # cram = psutil.virtual_memory()
    pid = os.getpid()
    python_process = psutil.Process(pid)
    # usage = (cram.total - cram.available) / 1024**3
    # total = cram.total / 1024**3
    # return usage, total
    return round(python_process.memory_info()[0]/2.**30, 2)


# def launch_job(cfg, init_method, func, daemon=False):
#     """
#     Run 'func' on one or more GPUs, specified in cfg
#     Args:
#         cfg (CfgNode): configs. Details can be found in
#             slowfast/config/defaults.py
#         init_method (str): initialization method to launch the job with multiple
#             devices.
#         func (function): job to run on GPU(s)
#         daemon (bool): The spawned processes’ daemon flag. If set to True,
#             daemonic processes will be created
#     """
#     if cfg.NUM_GPUS > 1:
#         torch.multiprocessing.spawn(
#             mpu.run,
#             nprocs=cfg.NUM_GPUS,
#             args=(
#                 cfg.NUM_GPUS,
#                 func,
#                 init_method,
#                 cfg.SHARD_ID,
#                 cfg.NUM_SHARDS,
#                 cfg.DIST_BACKEND,
#                 cfg,
#             ),
#             daemon=daemon,
#         )
#     else:
#         func(cfg=cfg)


def get_class_names(path, parent_path=None, subset_path=None):
    """
    Read json file with entries {classname: index} and return
    an array of class names in order.
    If parent_path is provided, load and map all children to their ids.
    Args:
        path (str): path to class ids json file.
            File must be in the format {"class1": id1, "class2": id2, ...}
        parent_path (Optional[str]): path to parent-child json file.
            File must be in the format {"parent1": ["child1", "child2", ...], ...}
        subset_path (Optional[str]): path to text file containing a subset
            of class names, separated by newline characters.
    Returns:
        class_names (list of strs): list of class names.
        class_parents (dict): a dictionary where key is the name of the parent class
            and value is a list of ids of the children classes.
        subset_ids (list of ints): list of ids of the classes provided in the
            subset file.
    """
    try:
        with pathmgr.open(path, "r") as f:
            class2idx = json.load(f)
    except Exception as err:
        print("Fail to load file from {} with error {}".format(path, err))
        return

    max_key = max(class2idx.values())
    class_names = [None] * (max_key + 1)

    for k, i in class2idx.items():
        class_names[i] = k

    class_parent = None
    if parent_path is not None and parent_path != "":
        try:
            with pathmgr.open(parent_path, "r") as f:
                d_parent = json.load(f)
        except EnvironmentError as err:
            print(
                "Fail to load file from {} with error {}".format(
                    parent_path, err
                )
            )
            return
        class_parent = {}
        for parent, children in d_parent.items():
            indices = [
                class2idx[c] for c in children if class2idx.get(c) is not None
            ]
            class_parent[parent] = indices

    subset_ids = None
    if subset_path is not None and subset_path != "":
        try:
            with pathmgr.open(subset_path, "r") as f:
                subset = f.read().split("\n")
                subset_ids = [
                    class2idx[name]
                    for name in subset
                    if class2idx.get(name) is not None
                ]
        except EnvironmentError as err:
            print(
                "Fail to load file from {} with error {}".format(
                    subset_path, err
                )
            )
            return

    return class_names, class_parent, subset_ids
