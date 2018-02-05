## LFW setup

This README shows how to set up the downloaded face images from the LFW dataset for face verification evaluation. 

Download the deep-funneled (roughly-aligned) images from LFW: `http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz`. Extract these at some local folder, referred to henceforth as LFW_DIR.

NOTE: all scripts are to be executed from the project root directory, *not* the sub-folder where this README file is located.

### Validation results

The LFW database provides DevTrain and DevTest splits as validation sets for developing code without overfitting to the 10 folds used in the final evaluation. The DevTest pairs are saved at `./lfw/data/pairsDevTest.txt`

Run `python ./lfw/eval_lfw.py -d LFW_DATASET_PATH -m MODEL_PATH` to get AUC score for verification on the dev test split. The number should be around 0.9989.

