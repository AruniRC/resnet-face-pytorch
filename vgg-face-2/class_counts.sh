

# Utility script to get class frequencies of VGGFace2 based on #images in class-folder

# >>>>>> Set paths here <<<<<<
DATA_PATH=/home/renyi/arunirc/data1/datasets/vggface2/train
DATA_PATH=/home/renyi/arunirc/data1/datasets/vggface2/train-crop
touch vgg-face-2/vggface_class_counts.txt

for folder in `ls ${DATA_PATH}`; do
    img_count=`ls ${DATA_PATH}/${folder} | wc -l`
    echo ${folder}' '${img_count}
    echo ${folder}' '${img_count} >> vgg-face-2/vggface_class_counts.txt
done
