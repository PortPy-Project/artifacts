# PortPy Artifacts

Pretrained models and reference optimization solutions for
[PortPy](https://github.com/PortPy-Project/PortPy).

Each file is published here rather than in the PortPy repository because it is **expensive to
regenerate** — the model weights are GPU-days of training, and each MIP reference solution
is roughly 90 solver-hours.

That is also the scope rule for this repository: *things that are expensive to regenerate.*
Source code belongs in [PortPy](https://github.com/PortPy-Project/PortPy), patient data in
the [PortPy dataset](https://huggingface.co/datasets/PortPy-Project/PortPy_Dataset).

## Downloading

From PortPy (no extra dependency, nothing to configure):

```python
import portpy.photon as pp

pp.list_artifacts()                                   # what is available
path = pp.download_artifact("beamlet_vit3d.pth")      # cached; reruns are instant
pp.download_artifacts()                               # everything, about 910 MB
```

From the command line:

```bash
python -m portpy.photon.utils.download_artifacts --list
python -m portpy.photon.utils.download_artifacts --names mednext_lung.pth --out ./portpy_artifacts
```

Or directly, with no PortPy at all:

```bash
curl -LO https://github.com/PortPy-Project/artifacts/releases/latest/download/mednext_lung.pth
```

`download_artifact` follows the **latest** release by default, so a new release reaches you
without updating PortPy. Pass `tag="v1.0.0"` to pin a vintage. A published tag never changes;
when a future release *modifies* an existing artifact rather than adding one, it will be given
a new file name, so a cached copy can never be silently wrong.

## Contents

### Beamlet dose prediction

Patient grid, 4 input channels, 1 output. All three share the `beamlet_unet3d` decoder, so
comparing them isolates the encoder.

| file | model | test MAE | size |
|---|---|---:|---:|
| `beamlet_unet3d.pth` | U-Net3D | 0.000140 ± 0.000028 | 86.2 MB |
| `beamlet_vit3d.pth` | ViT-3D (ViT-B scale) | 0.000336 ± 0.000178 | 420.6 MB |
| `beamlet_swin_unetr3d.pth` | Swin UNETR-3D | 0.000351 ± 0.000079 | 228.2 MB |

```python
import torch, portpy.photon as pp
from portpy.ai.models.networks3d import define_G

path = pp.download_artifact("beamlet_vit3d.pth")
net = define_G(4, 1, 32, "beamlet_vit3d", gpu_ids=[])
net.load_state_dict(torch.load(path, map_location="cpu"))   # strict by default
net.eval()
```

### Total dose prediction

8 input channels, 1 output. MedNeXt-B at kernel size 3.

| file | model | site | size |
|---|---|---|---:|
| `mednext_lung.pth` | MedNeXt | lung | 40.2 MB |
| `mednext_prostate.pth` | MedNeXt | prostate | 40.2 MB |
| `mednext_attn_lung.pth` | MedNeXt-Attention | lung | 40.3 MB |
| `mednext_attn_prostate.pth` | MedNeXt-Attention | prostate | 40.3 MB |
| `mednext_duo_joint.pth` | MedNeXt-DualDecoder | **both** | 13.9 MB |

```python
net = define_G(8, 1, 32, "mednext", gpu_ids=[])                 # or "mednext_attn"
```

`mednext_duo_joint.pth` is a single model covering both sites: a shared encoder with one
decoder per site, at 16 base filters. It needs both the narrower width and a site argument:

```python
net = define_G(8, 1, 32, "mednext_duo", mednext_n_channels=16, gpu_ids=[])
out = net(x, site="prostate")      # or site="lung"
```

### MIP reference solutions

Globally-optimising mixed-integer programs for `Lung_Patient_6`, so the optimization
baselines can be reproduced **without re-running the solver**.

| file | problem | objective | arrays | size |
|---|---|---:|---|---:|
| `imrt_mip_Lung_Patient_6.npz` | IMRT direct aperture, 5 beams × 10 apertures | 1370.956 | `optimal_intensity` (106), `MU` (50), `aperture_open` (50, 106) bool | 3.5 KB |
| `vmat_mip_Lung_Patient_6.npz` | VMAT, 7 control points | 152.2444 | `optimal_intensity` (147), `MU` (7), `left_leaf_pos` (35), `right_leaf_pos` (35) | 2.6 KB |

```python
import numpy as np, portpy.photon as pp

sol = np.load(pp.download_artifact("imrt_mip_Lung_Patient_6.npz"))
intensity, mu, aperture = sol["optimal_intensity"], sol["MU"], sol["aperture_open"]
```

They are `.npz`, not pickles, deliberately: a reference solution must open on any machine
with NumPy, must not need PortPy importable, must not carry a library version dependency,
and must not execute code on load.

## License

Apache 2.0 with the Commons Clause, matching PortPy: free for non-commercial academic and
research use. The MedNeXt architecture (MIC-DKFZ) and the GDP-HMM challenge solutions these
baselines reimplement are Apache-2.0; attribution is kept in the corresponding PortPy modules.

## Citation

```bibtex
@inproceedings{jhanwar2026portpyai,
  title={PortPy.AI: A Benchmark for AI and Optimization in Cancer Radiotherapy Planning},
  author={Jhanwar, Gourav and Tefagh, Mojtaba and Huang, Qijie and Golkarieh, Alireza and
          Hong, Linda and Pham, Hai and Taasti, Vicki Trier and Tuomaala, Seppo and
          Magliari, Anthony and Folkerts, Michael and Zhou, Ying and Yang, Jie and
          Nadeem, Saad and Zarepisheh, Masoud},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS),
             Evaluations and Datasets Track},
  year={2026}
}
```