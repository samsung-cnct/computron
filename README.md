# Computron

## Abstract

Many of the most performant deep learning models today in fields like language and image understanding are fine-tuned models that contain billions of parameters. In anticipation of workloads that involve serving many of such large models to handle different tasks, we develop Computron, a system that uses memory swapping to serve multiple distributed models on a shared GPU cluster. Computron implements a model parallel swapping design that takes advantage of the aggregate CPU-GPU link bandwidth of a cluster to speed up model parameter transfers. This design makes swapping large models feasible and can improve resource utilization. We demonstrate that Computron successfully parallelizes model swapping on multiple GPUs, and we test it on randomized workloads to show how it can tolerate real world variability factors like burstiness and skewed request rates.

## Installation for Development

Clone this repository and its submodules:

```shell
git clone --recurse-submodules git@github.com:dlzou/computron.git
```

Create an environment, install torch and Colossal-AI from PIP, then install Energon-AI and AlpaServe from the included submodules. Finally, install Computron from source.

```shell
conda create -n computron python=3.10
conda activate computron
pip install torch torchvision colossalai transformers
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

