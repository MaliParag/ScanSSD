# ScanSSD: Scanning Single Shot Detector for Math in Document Images

A [PyTorch](http://pytorch.org/) implementation of ScanSSD [Scanning Single Shot MultiBox Detector](https://paragmali.me/scanning-single-shot-detector-for-math-in-document-images/) by [**Parag Mali**](https://github.com/MaliParag/). It was developed using SSD implementation by [**Max deGroot**](https://github.com/amdegroot).


<img align="right" src=
"https://github.com/maliparag/scanssd/blob/master/images/detailed_math512_arch.png" height = 400/>

## Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#training-scanssd'>Training</a>
- <a href='#testing'>Testing</a>
- <a href='#performance'>Performance</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- Install [PyTorch](http://pytorch.org/)
- Clone this repository. Requires Python3
- Download the dataset by following the instructions on (https://github.com/MaliParag/TFD-ICDAR2019).
- Install [Visdom](https://github.com/facebookresearch/visdom) for real-time loss visualization during training!
  * To use Visdom in the browser:
  ```Shell
  # First install Python server and client
  pip install visdom
  # Start the server (probably in a screen or tmux)
  python -m visdom.server
  ```
  * Then (during training) navigate to http://localhost:8097/ (see the Train section below for training details).

## Training ScanSSD

- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights [here] (https://drive.google.com/file/d/1GqiyZ1TglNW5GrNQfXQ72S8mChhJ4_sD/view?usp=sharing)
- By default, we assume you have downloaded the file in the `ssd.pytorch/weights` dir:

- Run command

```Shell
python3 train.py 
					--dataset GTDB 
					--dataset_root ../ 
					--cuda True 
					--visdom True 
					--batch_size 16 
					--num_workers 4 
					--exp_name HBOXES_iter1 
					--model_type 512 
					--training_data training_data 
					--validation_data validation_data 
					--cfg hboxes512 
					--loss_fun ce 
					--kernel 3 3 
					--padding 1 1 
					--neg_mining True
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

## Testing
To test a trained network:

```Shell
python3 test.py 
				--dataset_root ../ 
				--trained_model HBOXES512_iter1GTDB.pth  
				--visual_threshold 0.25 
				--cuda True 
				--exp_name test_real_world_iter1 
				--test_data testing_data  
				--model_type 512 
				--cfg hboxes512 
				--padding 3 3 
				--kernel 1 1 
				--batch_size 8

```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  

## Performance

#### ICDAR 2019 Test


| Metric | Precision | Recall | F-score |
|:-:|:-:|:-:|:-:|
| IOU50 | 85.05 % | 75.85% | 80.19% |
| IOU75 | 77.38 % | 69.01% | 72.96% |

##### FPS
**GTX 1080:** ~27 FPS for 512 * 512 input images

## Acknowledgements
- [**Max deGroot**](https://github.com/amdegroot) for providing open-source SSD code
