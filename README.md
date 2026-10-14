# GVSynergy-Det

Implementation of **GVSynergy-Det: Synergistic Gaussian-Voxel Representations for Multi-View 3D Object Detection**, IEEE Transactions on Circuits and Systems for Video Technology, 2026.

[Paper](https://doi.org/10.1109/TCSVT.2026.3733040)

The detector combines multi-view voxel features with a predicted 3D Gaussian representation and adaptive feature fusion. Configurations are provided for ScanNet and ARKitScenes.

This repository is an **MMDetection3D v1.2.0 project**. It must be placed at `mmdetection3d/projects/GVSynergyDet`. 

## Installation

Use a dedicated Linux environment with an NVIDIA GPU, a CUDA-compatible host compiler, and CUDA 11.7. The version choices below follow the development environment.

| Component | Version |
| --- | --- |
| Python | 3.8 |
| PyTorch / torchvision | 2.0.0+cu117 / 0.15.0+cu117 |
| CUDA toolkit | 11.7 |
| MMEngine / MMCV / MMDetection | 0.10.4 / 2.0.1 / 3.0.0 |
| MMDetection3D | v1.2.0 |
| xFormers | 0.0.18 |

### 1. Arrange the repositories

Download this repository into a directory named `GVSynergyDet`. From its parent directory:

```bash
git clone --branch v1.2.0 --depth 1 https://github.com/open-mmlab/mmdetection3d.git
mkdir -p mmdetection3d/projects/GVSynergyDet
cp -a GVSynergyDet/. mmdetection3d/projects/GVSynergyDet/
cd mmdetection3d
```

Run all subsequent commands from the **MMDetection3D root**, unless stated otherwise. The copy includes hidden files; if the downloaded project is itself a Git checkout, its nested Git history is not needed to run the model.

### 2. Create the environment

```bash
conda create -n gvsynergydet python=3.8 -y
conda activate gvsynergydet
python -m pip install pip==25.0.1 setuptools==60.2.0 wheel==0.45.1

python -m pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 \
  --index-url https://download.pytorch.org/whl/cu117

conda install -c nvidia/label/cuda-11.7.0 cuda-toolkit -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
which nvcc
nvcc --version

python -m pip install -r projects/GVSynergyDet/requirements-extra.txt
python -m pip install openmim==0.3.9
mim install mmengine==0.10.4
mim install mmcv==2.0.1
mim install mmdet==3.0.0
python -m pip install -v -e . --no-build-isolation
python -m pip install xformers==0.0.18 --no-deps
```

Check that `nvcc` reports **11.7** before compiling extensions. The PyTorch wheel supplies CUDA runtime libraries, but the rasterizer also requires the CUDA compiler. Keep this PyTorch/xFormers combination together. The extra requirements explicitly include `pyre-extensions` and `typeguard`, which can otherwise be missed when using `--no-deps`.



### 3. Obtain the exact dependency sources

Two source snapshots are used. If the release already contains both directories below, skip the clone/checkout commands and proceed to step 4. See [THIRD_PARTY.md](THIRD_PARTY.md) for versions and licenses.

```bash
GVS_PROJECT="$PWD/projects/GVSynergyDet"
mkdir -p "$GVS_PROJECT/third_party"

git clone https://github.com/facebookresearch/dinov2.git \
  "$GVS_PROJECT/third_party/dinov2_py38"
git -C "$GVS_PROJECT/third_party/dinov2_py38" checkout \
  c3c2683a13cde94d4d99f523cf4170384b00c34c

git clone https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git \
  "$GVS_PROJECT/third_party/diff-gaussian-rasterization-w-depth"
git -C "$GVS_PROJECT/third_party/diff-gaussian-rasterization-w-depth" checkout \
  cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110
git -C "$GVS_PROJECT/third_party/diff-gaussian-rasterization-w-depth" \
  submodule update --init --recursive
```

Do not substitute the current DINOv2 branch for this historical snapshot. Its package installer declares Python >=3.9, so the Python 3.8 setup uses the model source directly instead of `pip install dinov2`.

Use the **w-depth fork** above: this project expects `(color, radii, depth)` from the rasterizer. The original Graphdeco rasterizer has a different interface.

### 4. Enable DINOv2 and build the rasterizer

```bash
GVS_PROJECT="$PWD/projects/GVSynergyDet"
export PYTHONPATH="$GVS_PROJECT/third_party/dinov2_py38:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install --no-deps --no-build-isolation \
  -e "$GVS_PROJECT/third_party/diff-gaussian-rasterization-w-depth"
```

An `export` affects only the current shell and its child processes. Repeat the CUDA and `PYTHONPATH` exports in new terminals, or put them in a local environment activation script. Alternatively, register DINOv2 once in this conda environment:

```bash
GVS_DINO_SOURCE="$PWD/projects/GVSynergyDet/third_party/dinov2_py38" python - <<'PY'
import os
import site
from pathlib import Path

source = Path(os.environ['GVS_DINO_SOURCE']).resolve()
assert (source / 'dinov2').is_dir()
Path(site.getsitepackages()[0], 'gvsynergydet_dinov2.pth').write_text(str(source) + '\n')
PY
```

The `.pth` registration persists for this environment; update it if the source directory moves. Do not commit machine-specific activation scripts or generated `.pth` files.



## Data

Set `data_root` in the relevant configuration to the directory containing the prepared annotations and RGB frames. Paths stored in the annotations should be relative to this directory when distributing annotations. Absolute image paths are supported locally.

```text
data/
  scannet/
    scannet_infos_train.pkl
    scannet_infos_val.pkl
    posed_images/<scene>/<frame>.jpg
  arkit/
    arkit_infos_train.pkl
    arkit_infos_val.pkl
    Training/<scene>/<scene>_frames/lowres_wide/<scene>_<timestamp>.png
    Validation/<scene>/<scene>_frames/lowres_wide/<scene>_<timestamp>.png
```

The supplied pipelines read **RGB images, camera parameters, and box annotations**. 

The annotation files use MMDetection3D's `dict(metainfo=..., data_list=...)` format with these additional multi-view fields in each sample:

| Field | ScanNet | ARKitScenes |
| --- | --- | --- |
| `images['CAM0']['img_path']` | List of RGB frame paths | List of RGB frame paths |
| `images['CAM0']['cam2img']` | Shared camera intrinsic matrix, normally 4 x 4 | Per-frame 3 x 3 intrinsic matrices |
| `images['CAM0']['lidar2cam']` | Per-frame aligned-world-to-camera matrices, `inverse(A @ C2W)` | Historical annotations store per-frame raw camera-to-world matrices here |
| `images['CAM0']['lidar2cam_ori']` | Required: raw camera-to-world matrices | Not used |
| `axis_align_matrix` | 4 x 4 alignment matrix `A` | 4 x 4 alignment matrix |
| `instances` | `bbox_3d` with 6 values and `bbox_label_3d` | `bbox_3d` with 7 values, including yaw, and `bbox_label_3d` |

Camera matrices and image lists must follow the same frame order. Boxes use the dataset's aligned depth coordinates and the class order in the corresponding config. Retain standard MMDetection3D sample metadata. In particular, the supplied ScanNet dataset configuration also requires `lidar_points` metadata containing `lidar_path` and `num_pts_feats` for its base parser, even though no point cloud is read by this pipeline.


Edit `data_root` in the config itself. Changing only the top-level value through `--cfg-options data_root=...` does not update all the dataset, loader, and evaluator values constructed from it.

## Training and evaluation

From the MMDetection3D root, after setting `data_root`:

```bash
# ScanNet: three GPUs, one sample per GPU as configured.
CUDA_VISIBLE_DEVICES=1,2,3 ./tools/dist_train.sh \
  projects/GVSynergyDet/configs/gvsynergydet_scannet.py 3

# ScanNet: evaluation with an existing compatible checkpoint.
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  projects/GVSynergyDet/configs/gvsynergydet_scannet.py \
  checkpoints/gvsynergydet_scannet.pth

# ARKitScenes: 
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  projects/GVSynergyDet/configs/gvsynergydet_arkit.py \
  checkpoints/gvsynergydet_arkit.pth
```

The checkpoint filenames above are examples, not bundled downloads. Use the matching dataset checkpoint. Changing the number of GPUs changes the effective batch size and may require learning-rate adjustments.



## Citation

```bibtex
@article{zhang2026gvsynergydet,
  title={GVSynergy-Det: Synergistic Gaussian-Voxel Representations for Multi-View 3D Object Detection},
  author={Zhang, Yi and Wang, Yi and Yao, Lei and Chau, Lap-Pui},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  doi={10.1109/TCSVT.2026.3733040}
}
```

## Acknowledgments and licenses

This project builds on MMDetection3D/ImVoxelNet, DINOv2, Atlas, and Gaussian rasterization code. Existing source attributions and dependency license files are retained. See [THIRD_PARTY.md](THIRD_PARTY.md). 