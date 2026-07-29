# Candidate Upstream Revisions and Assets

Verified: 2026-07-29

These revisions are recorded for reproducibility planning only. No upstream repository, package environment, checkpoint, or dataset has been downloaded or installed.

| Component | Official repository | Candidate revision |
|---|---|---|
| OpenVLA-OFT | https://github.com/moojink/openvla-oft | `e4287e94541f459edc4feabc4e181f537cd569a8` |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| VLA-Cache | https://github.com/siyuhsu/vla-cache | `a4909880573868dee2769343d52e793c0341678b` |
| OpenVLA-OFT Transformers fork | https://github.com/moojink/transformers-openvla-oft | `bc339d9ad707454c0c115970db43c260067c61ab` |
| dlimp OpenVLA fork | https://github.com/moojink/dlimp_openvla | `040105d256bd28866cc6620621a3d5f7b6b91b46` |

Candidate combined checkpoint:

- ID: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`
- revision: `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- total remote file size: `15,939,168,050` bytes
- access status when checked: public and not gated

Optional training dataset, which is not needed for evaluation and must not be downloaded:

- ID: `openvla/modified_libero_rlds`
- revision: `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`
- total remote file size: `10,230,693,345` bytes

Before installation, preserve these revisions in setup inputs. Save the resolved environment inventory after installation. Do not silently advance any revision between calibration and final evaluation.
