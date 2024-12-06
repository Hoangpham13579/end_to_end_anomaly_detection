import tqdm
import time
import os, glob
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import numpy as np
from time import sleep
from pathlib import Path
import random 

import torch, cv2
torch.backends.cudnn.benchmark = True  # for faster inference

from slowfast.utils import logging
from slowfast.utils.parser import load_config, parse_args
from slowfast.utils.misc import gpu_mem_usage, cpu_mem_usage

from slowfast.loader import VideoManager, ThreadVideoManager
from slowfast.config.defaults import assert_and_infer_cfg
# from slowfast.predictor import FeatureExtractor, AnomalyDetector

from slowfast.visualization.async_visualizer import AsyncVis
from slowfast.predictor import DetectionPipeline

from advanTech.AzureStorage import AzureStorage
from advanTech.Database import Database
from advanTech.Notification import Notification

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.get_logger(__name__)


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def get_output_format(path, cfg, fps=24):
    """
        Return a video writer object.
        Args:
            path (str): path to the output video file.
            fps (int or float): frames per second.
    """
    return cv2.VideoWriter(
        filename=path,
        fourcc=cv2.VideoWriter_fourcc(*"DIVX"),
        fps=float(fps),
        frameSize=(cfg.DEMO.DISPLAY_WIDTH, cfg.DEMO.DISPLAY_HEIGHT),
        isColor=True,
    )

def convert_avi_to_mp4(avi_file_path, output_name):
    """
        Convert .avi file to .mp4 file
    """
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True

def callAdvanTechAPI(task, cfg, azure_upload_flag, abn_time):
    """
        Call AdvanTech API, upload Azure storage, send notification for Anomalous events
    """
    start_time = 0
    end_time = 0
    # blob_container = ['binhduong', 'hcm', 'hanoi']
    blob_container = ['binhduong']
    for abn_val in task.clip_abn_score:
        # Show notification & call AdvanTech API, Azure storage for saving Anomalous Events
        if (abn_val > cfg.DEMO.ANOMALY_THRESHOLD) and \
                azure_upload_flag and \
                (cfg.DEMO.WEBCAM > -1):
            logger.info(' Abnormal event detected...')
            logger.info(f' Abnormal score: {abn_val}, threshold: {cfg.DEMO.ANOMALY_THRESHOLD}')

            # (NOTE) Upload anomalous video & wait for 10s for the next upload
            # blob_folder = str(cfg.DEMO.OUTPUT_FILE.split('/')[-1].split('_')[0]).lower()
            blob_folder = random.choice(blob_container)
            save_path = cfg.DEMO.OUTPUT_FILE[:7] + blob_folder + '_abnormal_' + str(time.strftime('%H:%M:%S')).replace(':', '_') + '_' + str(random.randrange(10)) + '.avi'
            # save_path = cfg.DEMO.OUTPUT_FILE[:-4] + '.avi'
            save_path = save_path.lower()

            # Save 10s anomalous frames as a video
            logger.info(' Save 10s recording video...')
            output_file = get_output_format(save_path, cfg, fps=cfg.DEMO.OUTPUT_FPS)
            for frame in task.frames:
                output_file.write(frame)
            output_file.release()

            # Convert .avi to .mp4
            logger.info(' Convert .avi to .mp4...')
            convert_avi_to_mp4(save_path, save_path[:-4])
            save_path = save_path[:-4]+'.mp4'
            filename = save_path.split('/')[-1]

            # TODO: Send video to Azure Blob Storage (Should wait 10s)
            logger.info(f' Send anomalous video to Azure Blob Storage...')
            azure = AzureStorage()
            azure.start_upload_thread(str(blob_folder).strip(), str(save_path).strip())

            # TODO: Save anomalous video info to database 
            # (Database name: anomaly, Username: postgres, Password: vguwarrior, Host: 34.143.216.9)
            logger.info(f' Save anomalous video {filename} info to database...')
            database = Database("anomaly", "postgres", "vguwarrior", "34.143.216.9")
            database.add_row(int(time.time()), int(time.time()), blob_folder, 'https://ik.imagekit.io/vguwarriors/'+filename, 'On', 'Office')

            # TODO: Send email notification
            logger.info(f' Send email notification...')
            start_time = time.strftime('%H:%M:%S')[:-3] + ":" + \
                            str(int(time.strftime('%H:%M:%S').split(':')[-1]) - 10)
            end_time = time.strftime('%H:%M:%S')[:-3] + ":" + \
                            str(int(time.strftime('%H:%M:%S').split(':')[-1]) + 10)
            noti = Notification("giathinhtran3@gmail.com", "@Advantech2023")
            noti.send_message(str(start_time).replace(":", "-"), str(end_time).replace(":", "-"), blob_folder, 'https://ik.imagekit.io/vguwarriors/'+filename, 'On', 'Office')

            azure_upload_flag = False
            abn_time = time.time()
            logger.info(f' Successfully upload video, save info and send notification...\n')
            break

        # Wait for 60s to upload the next anomalous video
        if (azure_upload_flag == False) and (time.time() - abn_time) >= 100:
            azure_upload_flag = True
            for f in glob.glob("output/*_abnormal_*"):
                print('DELETE!!!! ', f)
                os.remove(f)

    return azure_upload_flag, abn_time

def run_extractor_rtfm(cfg, frame_provider):
    """
    Run demo visualization.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        frame_provider (iterator): Python iterator that return task objects that are filled
            with necessary information such as `frames`, `id` and `num_buffer_frames` for the
            prediction and visualization pipeline.
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)

    # Time computing params
    dt = [0.0, 0.0, 0.0]
    
    async_vis = AsyncVis(cfg, n_workers=cfg.DEMO.NUM_VIS_INSTANCES)

    # Feature Extractor model (I3D Non-local) & Anomaly Detector (RTFM)
    if cfg.NUM_GPUS <= 1:
        model = DetectionPipeline(cfg=cfg, async_vis=async_vis)
    else:
        assert "Not support multi-GPUs yet"

    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
    assert (
        cfg.DEMO.BUFFER_SIZE <= seq_len // 2
    ), "Buffer size cannot be greater than half of sequence length."

    # FEATURE EXTRACTOR & ANOMALY DETECTION
    num_task = 0
    frame_count = 0
    logger.info("Extracting features & Anomaly detection...")
    with torch.no_grad():
        frame_provider.start()
        for able_to_read, task in frame_provider:
            dt[0] = time_synchronized()
            if not able_to_read:
                break
            if task is None:
                sleep(0.02)
                continue
            num_task += 1
            
            model.put(task)
            try:
                task = model.get()
                num_task -= 1
                yield task
            except IndexError:
                continue
            dt[1] = time_synchronized()

            if cfg.DEMO.WEBCAM < 0:
                # FPS (except warmup time)
                time = dt[1] - dt[0]
                dt[2] = dt[2] + time
                frame_count = frame_count + 240

        while num_task != 0:
            try:
                task = model.get()
                logger.info(f'Prediction score:\n {np.round(task.clip_abn_score, 4)}')
                num_task -= 1
                yield task
            except IndexError:
                continue

    if cfg.DEMO.WEBCAM < 0:
        logger.info("Demo FPS: {}".format(round(frame_count/dt[2], 2)))
        logger.info("GPU usage: {}".format(gpu_mem_usage()))
        logger.info("CPU usage: {}".format(cpu_mem_usage()))


def demo():
    """
        Demo
    """
    # Config reading
    args = parse_args()
    cfg = load_config(args, args.cfg_files[0])
    cfg = assert_and_infer_cfg(cfg)

    # DATA LOADER
    start = time.time()
    if cfg.DEMO.THREAD_ENABLE:
        frame_provider = ThreadVideoManager(cfg)
    else:
        frame_provider = VideoManager(cfg)

    # (NOTE) Video input length should be longer than 240 frames
    azure_upload_flag = True
    abn_time = time.time()
    for task in tqdm.tqdm(run_extractor_rtfm(cfg, frame_provider)):
        frame_provider.display(task)
        # # Call AdvanTech API, upload Azure storage, send notification for Anomalous eventss
        # if cfg.DEMO.WEBCAM > -1:
        #     azure_upload_flag, abn_time = callAdvanTechAPI(task,
        #                                                    cfg, 
        #                                                    azure_upload_flag, 
        #                                                    abn_time)

    print("GPU usage: {}".format(gpu_mem_usage()))
    print("CPU usage: {}".format(cpu_mem_usage()))
    logger.info("Finish demo in: {}".format(time.time() - start))

    frame_provider.join()
    frame_provider.clean()

    return


if __name__ == '__main__':
    demo()


