# Computron
Forked from https://github.com/dlzou/computron/tree/master/computron

## Abstract

Many of the most performant deep learning models today in fields like language and image understanding are fine-tuned models that contain billions of parameters. In anticipation of workloads that involve serving many of such large models to handle different tasks, we develop Computron, a system that uses memory swapping to serve multiple distributed models on a shared GPU cluster. Computron implements a model parallel swapping design that takes advantage of the aggregate CPU-GPU link bandwidth of a cluster to speed up model parameter transfers. This design makes swapping large models feasible and can improve resource utilization. We demonstrate that Computron successfully parallelizes model swapping on multiple GPUs, and we test it on randomized workloads to show how it can tolerate real world variability factors like burstiness and skewed request rates.

## Installation for Development

Clone this repository and its submodules:

```shell
git clone --recurse-submodules git@github.com:samsung-cnct/computron
```

Create an environment, install torch and Colossal-AI from PIP, then install Energon-AI and AlpaServe from the included submodules. Finally, install Computron from source.

```shell
conda create -n computron python=3.10
conda activate computron
pip install torch torchvision colossalai==0.3.2 transformers
pip install -e energonai/
pip install -e alpa_serve/

## In order to run sim.py, ray is needed
pip install ray

pip install -e .
```

## Once you built energonai, ensure to use c++17 for cuda
```shell
(computron) vmuser@yb-dev-1:/data/computron/energonai$ git diff
diff --git a/setup.py b/setup.py
index 5160cc4..4ee86e7 100644
--- a/setup.py
+++ b/setup.py
@@ -126,7 +126,7 @@ if build_cuda_ext:
             cc_flag.append('arch=compute_80,code=sm_80')

         extra_cuda_flags = [
-            '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
+            '-std=c++17', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
             '-U__CUDA_NO_HALF2_OPERATORS__', '-DTHRUST_IGNORE_CUB_VERSION_CHECK'
         ]
         ext_modules.append(
(computron) vmuser@yb-dev-1:
```

## Experiments Computron 
There are two experiments tests done in Computron, as described in the experiments folder: swapping mem perf test and the simulation of workloads for model serving.
The following shows how to run them.

Mem swapping perf with model opt-1.3b, two models and no parallelism, with 24 requests:
```shell
(computron) vmuser@yb-dev-1:/data/computron/experiments$ python round_robin.py opt-1.3b -b -n 2 -t 1 -p 1 -r 24
/home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/kernel/cuda_native/mha/flash_attn_2.py:21: UserWarning: FlashAttention only supports Ampere GPUs or newer.
  warnings.warn('FlashAttention only supports Ampere GPUs or newer.')
/home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/kernel/cuda_native/mha/flash_attn_2.py:27: UserWarning: please install flash_attn from https://github.com/HazyResearch/flash-attention
  warnings.warn('please install flash_attn from https://github.com/HazyResearch/flash-attention')
/home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/kernel/cuda_native/mha/mem_eff_attn.py:14: UserWarning: please install xformers from https://github.com/facebookresearch/xformers
  warnings.warn('please install xformers from https://github.com/facebookresearch/xformers')
Namespace(model_name='opt-1.3b', num_models=2, tp_world_size=1, pp_world_size=1, num_requests=24, blocking=True, no_log=False)
/home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/kernel/cuda_native/mha/flash_attn_2.py:21: UserWarning: FlashAttention only supports Ampere GPUs or newer.
  warnings.warn('FlashAttention only supports Ampere GPUs or newer.')
/home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/kernel/cuda_native/mha/flash_attn_2.py:27: UserWarning: please install flash_attn from https://github.com/HazyResearch/flash-attention
  warnings.warn('please install flash_attn from https://github.com/HazyResearch/flash-attention')
/home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/kernel/cuda_native/mha/mem_eff_attn.py:14: UserWarning: please install xformers from https://github.com/facebookresearch/xformers
  warnings.warn('please install xformers from https://github.com/facebookresearch/xformers')
[12/20/23 02:41:06] INFO     colossalai - colossalai - INFO: /home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/context/parallel_context.py:522 set_device
                    INFO     colossalai - colossalai - INFO: process rank 0 is bound to device 0
[12/20/23 02:41:08] INFO     colossalai - colossalai - INFO: /home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/context/parallel_context.py:558 set_seed
                    INFO     colossalai - colossalai - INFO: initialized seed on rank 0, numpy: 1024, python random: 1024, ParallelMode.DATA: 1024, ParallelMode.TENSOR: 1024,the default parallel seed
                             is ParallelMode.DATA.
                    INFO     colossalai - colossalai - INFO: /home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/initialize.py:115 launch
                    INFO     colossalai - colossalai - INFO: Distributed environment is initialized, data parallel size: 1, pipeline parallel size: 1, tensor parallel size: 1
[12/20/23 02:41:09] INFO     colossalai - energonai - INFO: /data/computron/energonai/energonai/model/model_factory.py:202 create_pipeline_model
                    INFO     colossalai - energonai - INFO: ==> Rank 0 built layer 0-24 / total 24
                    INFO     colossalai - energonai - INFO: /data/computron/energonai/energonai/model/model_factory.py:207 create_pipeline_model
                    INFO     colossalai - energonai - INFO: Rank0/0 model size = 2.837430272 GB
[12/20/23 02:41:12] INFO     colossalai - energonai - INFO: /data/computron/energonai/energonai/model/model_factory.py:202 create_pipeline_model
                    INFO     colossalai - energonai - INFO: ==> Rank 0 built layer 0-24 / total 24
                    INFO     colossalai - energonai - INFO: /data/computron/energonai/energonai/model/model_factory.py:207 create_pipeline_model
                    INFO     colossalai - energonai - INFO: Rank0/0 model size = 2.837430272 GB
[12/20/23 02:41:13] INFO     colossalai - computron - INFO: /data/computron/computron/engine.py:131 __init__
                    INFO     colossalai - computron - INFO: engine started
0 request waiting
[12/20/23 02:41:14] INFO     colossalai - computron - INFO: /data/computron/computron/engine.py:238 _completion_loop
                    INFO     colossalai - computron - INFO: opt0 loaded: True, time: 0.254
[12/20/23 02:41:15] INFO     colossalai - computron - INFO: /data/computron/computron/engine.py:233 _completion_loop
                    INFO     colossalai - computron - INFO: opt0 batch size: 1, time: 1.379
0 response time: 1.6469082832336426
hello worldBLIC
1 request waiting
                    INFO     colossalai - computron - INFO: /data/computron/computron/engine.py:238 _completion_loop
                    INFO     colossalai - computron - INFO: opt0 loaded: False, time: 0.276
                    INFO     colossalai - computron - INFO: /data/computron/computron/engine.py:238 _completion_loop
                    INFO     colossalai - computron - INFO: opt1 loaded: True, time: 0.379
                    INFO     colossalai - computron - INFO: /data/computron/computron/engine.py:233 _completion_loop
                    INFO     colossalai - computron - INFO: opt1 batch size: 1, time: 0.036
1 response time: 0.4335768222808838
hello world Islam
2 request waiting
[12/20/23 02:41:16] INFO     colossalai - computron - INFO: /data/computron/computron/engine.py:238 _completion_loop
                    INFO     colossalai - computron - INFO: opt1 loaded: False, time: 0.301
                    INFO     colossalai - computron - INFO: /data/computron/computron/engine.py:238 _completion_loop
                    INFO     colossalai - computron - INFO: opt0 loaded: True, time: 0.313
                    INFO     colossalai - computron - INFO: /data/computron/computron/engine.py:233 _completion_loop
                    INFO     colossalai - computron - INFO: opt0 batch size: 1, time: 0.028
2 response time: 0.3548769950866699
hello worldinnie
```

Running simulation with default configurations:
```shell
(computron) vmuser@yb-dev-1:/data/computron/experiments$ python sim.py
/home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/kernel/cuda_native/mha/flash_attn_2.py:21: UserWarning: FlashAttention only supports Ampere GPUs or newer.
  warnings.warn('FlashAttention only supports Ampere GPUs or newer.')
/home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/kernel/cuda_native/mha/flash_attn_2.py:27: UserWarning: please install flash_attn from https://github.com/HazyResearch/flash-attention
  warnings.warn('please install flash_attn from https://github.com/HazyResearch/flash-attention')
/home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/kernel/cuda_native/mha/mem_eff_attn.py:14: UserWarning: please install xformers from https://github.com/facebookresearch/xformers
  warnings.warn('please install xformers from https://github.com/facebookresearch/xformers')
Namespace(model_name='opt-1.3b', config=0, tp_world_size=1, pp_world_size=1, batch_size=32, duration=30, no_log=False)
/home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/kernel/cuda_native/mha/flash_attn_2.py:21: UserWarning: FlashAttention only supports Ampere GPUs or newer.
  warnings.warn('FlashAttention only supports Ampere GPUs or newer.')
/home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/kernel/cuda_native/mha/flash_attn_2.py:27: UserWarning: please install flash_attn from https://github.com/HazyResearch/flash-attention
  warnings.warn('please install flash_attn from https://github.com/HazyResearch/flash-attention')
/home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/kernel/cuda_native/mha/mem_eff_attn.py:14: UserWarning: please install xformers from https://github.com/facebookresearch/xformers
  warnings.warn('please install xformers from https://github.com/facebookresearch/xformers')
[12/20/23 02:35:59] INFO     colossalai - colossalai - INFO: /home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/context/parallel_context.py:522 set_device
                    INFO     colossalai - colossalai - INFO: process rank 0 is bound to device 0
[12/20/23 02:36:00] INFO     colossalai - colossalai - INFO: /home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/context/parallel_context.py:558 set_seed
                    INFO     colossalai - colossalai - INFO: initialized seed on rank 0, numpy: 1024, python random: 1024, ParallelMode.DATA: 1024, ParallelMode.TENSOR: 1024,the default parallel seed
                             is ParallelMode.DATA.
                    INFO     colossalai - colossalai - INFO: /home/vmuser/anaconda3/envs/computron/lib/python3.10/site-packages/colossalai/initialize.py:115 launch
                    INFO     colossalai - colossalai - INFO: Distributed environment is initialized, data parallel size: 1, pipeline parallel size: 1, tensor parallel size: 1
[12/20/23 02:36:01] INFO     colossalai - energonai - INFO: /data/computron/energonai/energonai/model/model_factory.py:202 create_pipeline_model
                    INFO     colossalai - energonai - INFO: ==> Rank 0 built layer 0-24 / total 24
                    INFO     colossalai - energonai - INFO: /data/computron/energonai/energonai/model/model_factory.py:207 create_pipeline_model
                    INFO     colossalai - energonai - INFO: Rank0/0 model size = 2.837430272 GB
[12/20/23 02:36:04] INFO     colossalai - energonai - INFO: /data/computron/energonai/energonai/model/model_factory.py:202 create_pipeline_model
                    INFO     colossalai - energonai - INFO: ==> Rank 0 built layer 0-24 / total 24
                    INFO     colossalai - energonai - INFO: /data/computron/energonai/energonai/model/model_factory.py:207 create_pipeline_model
                    INFO     colossalai - energonai - INFO: Rank0/0 model size = 2.837430272 GB
[12/20/23 02:36:07] INFO     colossalai - energonai - INFO: /data/computron/energonai/energonai/model/model_factory.py:202 create_pipeline_model
                    INFO     colossalai - energonai - INFO: ==> Rank 0 built layer 0-24 / total 24
                    INFO     colossalai - energonai - INFO: /data/computron/energonai/energonai/model/model_factory.py:207 create_pipeline_model
                    INFO     colossalai - energonai - INFO: Rank0/0 model size = 2.837430272 GB
[12/20/23 02:36:08] INFO     colossalai - computron - INFO: /data/computron/computron/engine.py:131 __init__
                    INFO     colossalai - computron - INFO: engine started
client model: opt0, arrival: Gamma(1, 0.25), num requests: 30
client model: opt1, arrival: Gamma(1, 0.25), num requests: 30
client model: opt2, arrival: Gamma(1, 0.25), num requests: 31
[12/20/23 02:36:09] INFO     colossalai - computron - INFO: /data/computron/computron/engine.py:238 _completion_loop
                    INFO     colossalai - computron - INFO: opt0 loaded: True, time: 0.256
^C[12/20/23 02:36:10] INFO     colossalai - computron - INFO: /data/computron/computron/engine.py:238 _completion_loop     
```
