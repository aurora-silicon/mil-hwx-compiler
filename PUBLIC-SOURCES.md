# Public camera implementation sources

The compiler and package builder are MIT. The camera builder consumes the
audited exports from
[`aurora-silicon/omarchy-ane`](https://github.com/aurora-silicon/omarchy-ane),
whose `examples/camera-effects/LICENCE-AUDIT.md` records the complete licence
and conversion audit. No model weights or evaluation images are added here.

| Model | Public weights | Licence | SHA-256 |
| --- | --- | --- | --- |
| General selfie segmentation | [MediaPipe release](https://storage.googleapis.com/mediapipe-assets/selfie_segmentation.tflite) | Apache-2.0; [Google model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf) | `9ee168ec7c8f2a16c56fe8e1cfbc514974cbbb7e434051b455635f1bd1462f5c` |
| YuNet 2023mar | [OpenCV Zoo, pinned revision](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet) | MIT, including weights | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |

Camera parity uses the existing COCO 2017 CC BY 2.0 portrait selection.
Inputs, attribution metadata and hashes remain evaluation artifacts; images
are not committed. Compiler encoders retain their existing decoded-field
provenance and public register references.
