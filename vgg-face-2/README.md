## PyTorch ResNet on VGG Face 2

Demo to train a ResNet model on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset.


### Setup

* Install [Anaconda](https://conda.io/docs/user-guide/install/linux.html) if not already installed in the system.
* Create an Anaconda environment: `conda create -n resnet-face python=2.7` and activate it: `source activate resnet-face`.
* Install PyTorch and TorchVision inside the Anaconda environment. First add a channel to conda: `conda config --add channels soumith`. Then install: `conda install pytorch torchvision cuda80 -c soumith`.
* Install the dependencies using conda: `conda install scipy Pillow tqdm scikit-learn scikit-image numpy matplotlib ipython pyyaml`.


### Dataset preparation

:small_red_triangle: TODO

* Download the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset, after registering on their website. 
* the images need to be cropped into 'train' and 'val' folders. The following shell command does this for each batch in parallel.
`for i in {0..2}; python umd-face/run_crop_face -b $i &; done`

 - must parallellize this more: takes very long. Maybe MATLAB parfor?


### Usage

:small_red_triangle: TODO


#### Training: 

:small_red_triangle: TODO

* Look under `config.py` to select a training configuration
* Training script is `umd-face/train_resnet_umdface.py`.
* :small_red_triangle: Multiple GPUs: Under section 3 ("Model") of the training script, we specify which GPUs to use in parallel: `model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4]).cuda()`. Change these numbers depending on the number of available GPUs. Use `watch -d nvidia-smi` to constantly monitor the multi-GPU usage from the terminal. TODO - make this into command args.
* At the terminal, specify where the cropped face images are saved using an environment variable: `DATASET_PATH=local/path/to/cropped/umd/faces`
* The training of ResNet-50 was done in 3 stages (*configs 4, 5 and 6*), each of *30 epochs*. After the first stage, we started from the saved model of the previous stage (using the `--model_path` or `-m` command-line argument) and divided the learning rate by a factor of 10.
* Stage 1 (config-4): train on  the *full UMDFaces dataset for 30 epochs* (42180 iterations with batchsize 250) with a learning rate of 0.001, starting from an ImageNet pre-trained model. These settings are defined in *config-4* of `config.py`, which is selected using the `-c 4` flag in the command. Example to train a ResNet-50 on UMDFaces dataset using config-4: Run `python umd-face/train_resnet_umdface.py -c 4 -d $DATASET_PATH`.
* Stage 2 (config-5): use the best model checkpointed from config-4 to initialize the network and train it using config-5 `python umd-face/train_resnet_umdface.py -c 5 -m ./umd-face/logs/MODEL-resnet_umdfaces_CFG-004_TIMESTAMP/model_best.pth.tar -d $DATASET_PATH` and so on for the subsequent stages.
* *Training logs:* Each time the training script is run, a new output folder with a timestamp is created by default under `./umd-face/logs` , i.e.  `./umd-face/logs/MODEL-CFG-TIMESTAMP/`. Under an experiment's log folder the settings for each experiment can be viewed in `config.yml`; metrics such as the training and validation losses are updated in `log.csv`. 
Most of the usual settings (data augmentations, learning rates, number of epochs to train, etc.) can be customized by editing `config.py` and `umd-face/train_resnet_umdface.py`.
* *Plotting CSV logs:* The log-file plotting utility function can be called from the command line as shown in the snippet below. The figure is saved under the log folder in the output location of that experiment.
    * `LOG_FILE=umd-face/logs/MODEL-resnet_umdfaces_CFG-004_TIMESTAMP/log.csv`
    * `python -c "from utils import plot_log_csv; plot_log_csv('$LOG_FILE')"`

stage 1 |   stage 2  | stage 3  
:------:|:----------:|:--------:
![](samples/stage1_log_plots.png)|  ![](samples/stage2_log_plots.png) | ![](samples/stage3_log_plots.png) 




