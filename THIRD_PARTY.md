# Third-party sources

The release uses the following historical sources. Preserve their copyright
notices and license files when redistributing them. Licenses listed here apply
to the corresponding component, not automatically to the whole project.

| Component | Source/version | License location |
| --- | --- | --- |
| MMDetection3D | [v1.2.0](https://github.com/open-mmlab/mmdetection3d/tree/v1.2.0) | Upstream `LICENSE` (Apache-2.0); copy under `third_party/licenses/mmdetection3d-LICENSE` |
| DINOv2 | [`c3c2683a13cde94d4d99f523cf4170384b00c34c`](https://github.com/facebookresearch/dinov2/tree/c3c2683a13cde94d4d99f523cf4170384b00c34c) | `third_party/dinov2_py38/LICENSE` (CC BY-NC 4.0 at this commit) |
| Depth-enabled Gaussian rasterizer | [`cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110`](https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/tree/cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110) | `third_party/diff-gaussian-rasterization-w-depth/LICENSE.md` (Gaussian-Splatting License, including noncommercial restrictions) |
| GLM | Rasterizer's pinned `third_party/glm` submodule | Its source headers and `copying.txt` |

The bundled DINOv2 snapshot matches the listed commit. The rasterizer snapshot
includes its GLM source dependency. Git history, build products, caches, and
compiled binaries are excluded. If a source-only project copy lacks these
snapshots, use the pinned clone instructions in the README.

Additional code-level acknowledgments, including Atlas, remain in the source
files. Preserve upstream author names and contact information: these are
attributions, not local experiment metadata.

The project author's license for original contributions remains to be selected
before a licensed public release. Do not replace the historical DINOv2 or
rasterizer licenses with a blanket Apache-2.0 or MIT license.
