
This repository shows how to train ResNet models in PyTorch on publicly available face recognition datasets.

### Setup

* Install [Anaconda](https://conda.io/docs/user-guide/install/linux.html) if not already installed in the system.
* Create an Anaconda environment: `conda create -n resnet-face python=2.7` and activate it: `source activate resnet-face`.
* Install PyTorch and TorchVision inside the Anaconda environment. First add a channel to conda: `conda config --add channels soumith`. Then install: `conda install pytorch torchvision cuda80 -c soumith`.
* Install the dependencies using conda: `conda install scipy Pillow tqdm scikit-learn scikit-image numpy matplotlib ipython pyyaml`.
* *Notes*:
    * Multiple GPUs (we used 5 GeForce GTX 1080Ti in parallel) recommended for the training to finish in reasonable time.
    * Tested on server running CentOS
    * Using [PyTorch](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) 
    * Optional - an explanatory [blogpost](https://blog.waya.ai/deep-residual-learning-9610bb62c355) on Deep Residual Networks (ResNets) structure.

### Contents
- [ResNet-50 on UMD-Faces](https://github.com/AruniRC/resnet-face-pytorch#pytorch-resnet-on-umd-face)
    - [Dataset preparation](https://github.com/AruniRC/resnet-face-pytorch#dataset-preparation)
    - [Training](https://github.com/AruniRC/resnet-face-pytorch#training)
    - [Evaluation demo](https://github.com/AruniRC/resnet-face-pytorch#evaluation)
- [ResNet-50 on VGGFace2](https://github.com/AruniRC/resnet-face-pytorch#pytorch-resnet-on-vggface2)
    - [Dataset preparation](https://github.com/AruniRC/resnet-face-pytorch#dataset-preparation-1)
    - [Training](https://github.com/AruniRC/resnet-face-pytorch#training-1)
    - [Evaluation LFW](https://github.com/AruniRC/resnet-face-pytorch#evaluation-1)



## PyTorch ResNet on UMD-Face

Demo to train a ResNet-50 model on the [UMDFaces](http://www.umdfaces.io/) dataset.

### Dataset preparation

* Download the [UMDFaces](http://www.umdfaces.io/) dataset (the 3 batches of _still_ images), which contains 367,888 face annotations for 8,277 subjects, split into 3 batches.
* The images need to be cropped into 'train' and 'val' folders. Since the UMDFaces dataset does not specify training and validation sets, by default we select two images from every subject for validation. 
* The cropping code is in the Python script [umd-face/run_crop_face.py](./umd-face/run_crop_face.py). It takes the following command-line arguments:
    * --dataset_path (-d)
    * --output_path (-o)
    * --batch (-b) 
* The following shell command does the cropping for each batch in parallel, using default `dataset_path` and `output_path`.
`for i in {0..2}; python umd-face/run_crop_face -b $i &; done`.

:small_red_triangle: **TODO** - takes very long, convert into shell+ImageMagick script.


### Usage

#### Training:  
* Training script is [umd-face/train_resnet_umdface.py](./umd-face/train_resnet_umdface.py)
* Multiple GPUs: 
    * Under section 3 ("Model") of the training script, we specify which GPUs to use in parallel: `model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4]).cuda()`. Change these numbers depending on the number of available GPUs. 
    * Use `watch -d nvidia-smi` to constantly monitor the multi-GPU usage from the terminal. 
    * :small_red_triangle: **TODO** - make this into command-line args.
* At the terminal, specify where the cropped face images are saved using an environment variable: `DATASET_PATH=local/path/to/cropped/umd/faces`
* [config.py](./config.py) lists all the training configurations (i.e. model hyper-parameters) in a numbered Python dict.
    * The training of ResNet-50 was done in 3 stages (*configs 4, 5 and 6*), each of *30 epochs*. For the first stage, we started with the ImageNet-pre-trained model from PyTorch. After the first stage, we started from the saved model of the previous stage (using the `--model_path` or `-m` command-line argument) and divided the learning rate by a factor of 10.
    * Stage 1 (config-4): train on  the *full UMDFaces dataset for 30 epochs* (42180 iterations with batchsize 250) with a learning rate of 0.001, starting from an ImageNet pre-trained model. These settings are defined in *config-4* of [config.py](./config.py), which is selected using the `-c 4` flag in the command. Example to train a ResNet-50 on UMDFaces dataset using config-4: run `python umd-face/train_resnet_umdface.py -c 4 -d $DATASET_PATH`.
    * Stage 2 (config-5): use the best model checkpointed from config-4 to initialize the network and train it using config-5 `python umd-face/train_resnet_umdface.py -c 5 -m ./umd-face/logs/MODEL-resnet_umdfaces_CFG-004_TIMESTAMP/model_best.pth.tar -d $DATASET_PATH` and so on for the subsequent stages.
* *Training logs:* Each time the training script is run, a new output folder with a timestamp is created by default under `./umd-face/logs` , i.e.  `./umd-face/logs/MODEL-CFG-TIMESTAMP/`. Under an experiment's log folder the settings for each experiment can be viewed in `config.yaml`; metrics such as the training and validation losses are updated in `log.csv`. 
Most of the usual settings (data augmentations, learning rates, number of epochs to train, etc.) can be customized by editing `config.py` and `umd-face/train_resnet_umdface.py`.
* *Plotting CSV logs:* The log-file plotting utility function can be called from the command line as shown in the snippet below. The figure is saved under the log folder in the output location of that experiment.
    * `LOG_FILE=umd-face/logs/MODEL-resnet_umdfaces_CFG-004_TIMESTAMP/log.csv`
    * `python -c "from utils import plot_log_csv; plot_log_csv('$LOG_FILE')"`
    * If that gives parsing errors: `python -c "from utils import plot_log; plot_log('$LOG_FILE')"`

stage 1 |   stage 2  | stage 3  
:------:|:----------:|:--------:
![](samples/stage1_log_plots.png)|  ![](samples/stage2_log_plots.png) | ![](samples/stage3_log_plots.png) 


#### Pre-trained model: 

:red_circle: TODO - release pre-trained ResNet-50 on UMD-Faces :construction:


#### Evaluation: 

*Verification demo:* We have a short script, [run_resnet_demo.py](./run_resnet_demo.py) to demonstrate the usage of the model on a toy face verification example. The visualized output of the demo is saved in the the root directory of the project. The 3 sample images are taken from the [LFW](http://vis-www.cs.umass.edu/lfw/) dataset.

![](samples/demo_verif.png)


---


## PyTorch ResNet on VGGFace2

Training a *ResNet-50* model in PyTorch on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset.


### Dataset preparation

* Register on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) website and download their dataset
* VGGFace2 provides loosely-cropped images. We use crops from the [Faster R-CNN face detector](https://github.com/playerkk/face-py-faster-rcnn), saved as a CSV in `[filename, subject_id, xmin, ymin, width, height]` format (the CSV with pre-computed face crops is not yet made available).  
* The `vgg-face-2/crop_face.sh` script is used to crop the face images into a separate output folder. Please look at the settings section in the script to assign correct paths depending on where the VGGFace2 data was downloaded on your local machine. This takes about a day. **TODO** - multi-process.
    * Training images are under `OUTPUT_PATH/train-crop`
    * Validation images (2 images per subject) under `OUTPUT_PATH/val-crop`

### Training

* We used 7 GeForce GTX 1080Ti GPUs in parallel (PyTorch DataParallel) to train the network, using Batch Normalization, following the training procedure described in the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) paper.
* Training settings are defined under *configs-20, 21, 22* in [config.py](./config.py).
* Briefly, training is done using SGD optimizer, starting with a learning rate of 0.1, which gets divided by 10 in subsequent stages. A new stage is begun whenever the validation curve flattens.
* First stage training command, starting with a ResNet-50 from scratch: `python vgg-face-2/train_resnet50_vggface_scratch.py -c 20`
* Subsequent training stages (with lowered learning rates) would be: 
    - `python vgg-face-2/train_resnet50_vggface_scratch.py -c 21 -m PATH_TO_BEST_MODEL_CFG-20`
    - `python vgg-face-2/train_resnet50_vggface_scratch.py -c 22 -m PATH_TO_BEST_MODEL_CFG-21`

### Evaluation

Instructions on how to setup and run the LFW evaluation are at [lfw/README.md](./lfw/README.md). 

DevTest |  10 fold   
:------:|:----------:
![](samples/lfw_roc_devTest.png)|  ![](samples/lfw_roc_10fold.png)




