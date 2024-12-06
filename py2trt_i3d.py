#install torch-tensorrt:
#!pip install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases

from pathlib import Path
import numpy as np
import torch, os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.predictor_convert import FeatureExtractor
import torch_tensorrt


# Config reading
args = parse_args()
cfg = load_config(args, args.cfg_files[0])
cfg = assert_and_infer_cfg(cfg)

# Set random seed from configs.
np.random.seed(cfg.RNG_SEED)
torch.manual_seed(cfg.RNG_SEED)

# I3D FEATURE EXTRACTOR
i3d = FeatureExtractor(cfg=cfg).extractor
i3d.eval()
feat = torch.randn((1,3,16,224,224)).half().cuda()
script_i3d = torch.jit.trace(i3d, (feat,)).cuda()
print("Torchscript graph: ", script_i3d.graph)
torchscript_out = script_i3d(feat)

# trtorch
i3d_compile_settings = {
    "inputs":  
    [
        #specify input shape of the tensorrt model
        torch_tensorrt.Input((1,3,16,224,224), dtype=torch.half,)
    ],
    "enabled_precisions": {torch.half},
    "require_full_compilation": False,
    "truncate_long_and_double": True,
    "workspace_size" : 1 << 10,
    # "torch_executed_ops":["aten::Int"],
}

#avoid bug in library
import locale
locale.getpreferredencoding = lambda: "UTF-8"

#convert the model
trt_ts_module = torch_tensorrt.compile(
    i3d, 
    **i3d_compile_settings
)

trt_out = trt_ts_module(feat)
print("\ntorchscript results, ", torchscript_out.size())
print("trt results, ", trt_out.size())
print("diff: ", torch.mean(torch.abs(trt_out - torchscript_out)))

#save the model
torch.jit.save(trt_ts_module, "./ckpt/trt_i3d_nonlocal.ts")
