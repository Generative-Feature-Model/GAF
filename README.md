# Generative Attention Model-Based Feature Semantics for Temporal Action Detection

This is an official implementation in PyTorch of GAF. 

## Abstract

Temporal action detection is an important yet challenging task in video analysis, aiming at inferring both the action category and localization of the start and end frame for each action instance in the untrimmed video. However, current methods ignore intra and inter relations of feature semantics, and suffer from deviated action boundaries and inaccurate detection, which greatly limits the utility of the detection task. In this paper, we propose a novel generative attention mechanism to simultaneously model the intra and inter dependencies of temporal action feature semantics by using the differences of actions’ foreground and background. Extensive experiments show that, compared with the state-of-the-art, our method achieves better performance on THUMOS14, and comparable performance on ActivityNet v1.3. Particu-
larly, for complex background and small objective action detection tasks, our method achieves around 3.6% mAP improvement on THUMOS14.

## Summary

- Propose a novel generative attention mechanism to simultaneously model the intra and inter dependencies of temporal action feature representation.
- Leverage inter and intra relations of the feature dimension to locate the areas and boundaries of action instances.
- For complex background and small objective action detection tasks, our method get better performance.

## Getting Started

### Environment

- Python 3.7
- PyTorch == 1.4.0
- NVIDIA GPU

### Setup

```shell script
pip3 install -r requirements.txt
python3 setup.py develop
```

### Data Preparation

- **THUMOS14 RGB data:**

1. Download pre-processed RGB npy data (13.7GB): [Baiduyun](https://pan.baidu.com/s/1MRm6F9cgOv4MSlNajwaI4g ), password：xot6
2. Unzip the RGB npy data to `./datasets/thumos14/validation_npy/` and `./datasets/thumos14/test_npy/`

- **THUMOS14 flow data:**

1. Because it costs more time to generate flow data for THUMOS14, to make easy to run flow model, we provide the pre-processed flow data in Baiduyun(3.4GB):
   [Baiduyun](https://pan.baidu.com/s/1_Zm_FQRnTtTkXEAkCQgnAg ), password：7rpw
1. Unzip the flow npy data to `./datasets/thumos14/validation_flow_npy/` and `./datasets/thumos14/test_flow_npy/`



### Inference

We provide the pretrained models contain RGB and flow models for THUMOS14 dataset:[Google_Drive](https://drive.google.com/drive/folders/10RO2OrTm3p-ATiSnOyhYPRAc80y_4UMS?usp=sharing)

```shell script
# run RGB model
python3 GAF/thumos14/test.py configs/thumos14.yaml --checkpoint_path=models/thumos14/checkpoint-15.ckpt --output_json=thumos14_rgb.json

# run flow model
python3 GAF/thumos14/test.py configs/thumos14_flow.yaml --checkpoint_path=models/thumos14_flow/checkpoint-16.ckpt --output_json=thumos14_flow.json

# run fusion (RGB + flow) model
python3 GAF/thumos14/test.py configs/thumos14.yaml --fusion --output_json=thumos14_fusion.json
```

### Evaluation

The output json results of pretrained model can be downloaded from:[Google_Drive](https://drive.google.com/file/d/1pmQjIT57OlJLLJZh0a7GExhr3V3HKvtf/view?usp=sharing)

```shell script
# evaluate THUMOS14 fusion result as example
python3 GAF/thumos14/eval.py output/thumos14_fusion.json
```

### Result

#### Thumos14

| Type            |                            Model                             |                           Backbone                           | 0.1                                                          | 0.2                                                          | 0.3                                                          | 0.4                                                          | 0.5                                                          |
| --------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Non-Attention   | BSN<br />BU-TAL <br />GTAN<br />DBS<br />G-TAD<br />GCM<br />ContextLoc<br />AFSD | TS<br />I3D<br />P3D<br />C3D<br />TS<br />I3D<br />I3D<br />I3D<br /> | -<br />-<br />69.1<br />56.7<br />-<br />72.5<br />-<br />72.2 | -<br />-<br />63.7<br />54.7<br />-<br />70.9<br />-<br />70.8 | 53.5<br />53.9<br />57.8<br />50.6<br />54.5<br />66.5<br />68.3<br />67.1 | 45.0<br />50.7<br />47.2<br />43.1<br />47.6<br />60.8<br />**63.8**<br />62.2 | 36.9<br />45.4<br />38.8<br />34.3<br />40.2<br />51.9<br />54.3<br />55.5 |
| Attention-Based |         TadTR<br />AGCN<br />CSA<br />RAM<br />Ours          |            C3D<br />C3D<br />TS<br />TS<br />I3D             | -<br />59.3<br />-<br />65.4<br />**73.6**                   | -<br />59.6<br />-<br />63.1<br />**72.1**                   | 59.6<br />57.1<br />64.4<br />58.8<br />**68.9**             | 55.0<br />51.6<br />58.0<br />52.7<br />62.1                 |46.6<br />38.6<br />49.2<br />43.7<br />**55.9**   |

#### ActivityNet v1.3

| Type            |                            Model                             |                           Backbone                           | 0.5                                                          | 0.75                                                         | 0.95                                                         |
| --------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Non-Attention   | BSN<br />BU-TAL <br />GTAN<br />DBS<br />G-TAD<br />GCM<br />ContextLoc<br />AFSD | TS<br />I3D<br />P3D<br />C3D<br />TS<br />I3D<br />I3D<br />I3D<br /> | 46.5<br />43.5<br />52.6<br />43.2<br />50.4<br />51.0<br />**56.0**<br />52.4 | 30.0<br />33.9<br />34.1<br />25.8<br />34.6<br />35.2<br />35.2<br />35.3 | 8.0<br />**9.2**<br />8.9<br />6.1<br />9.0<br />7.4<br />3.6<br />6.5 |
| Attention-Based |         TadTR<br />AGCN<br />CSA<br />RAM<br />Ours          |            C3D<br />C3D<br />TS<br />TS<br />I3D             | 47.6<br />30.4<br />51.9<br />37.0<br />53.2                 | 31.7<br />-<br />36.9<br />23.1<br />**36.9**                | 8.0<br />-<br />8.7<br />3.3<br />3.9                        |

### Training

```shell script
# train the RGB model
python3 GAF/thumos14/train.py configs/thumos14.yaml --lw=10 --cw=1 --piou=0.5

# train the flow model
python3 GAF/thumos14/train.py configs/thumos14_flow.yaml --lw=10 --cw=1 --piou=0.5
```

### 
