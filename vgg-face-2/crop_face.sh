

# Utility script to crop large image datasets in parallel from the console

# INFO: kill all sub-processes: kill $(ps -s $$ -o pid=)

crop_image_multi () {
    DATA_PATH=$1
    OUT_PATH=$2
    ANNOT_FILE=$3
    SUBJECT_FOLDER=$4

    OLDIFS=$IFS
    IFS=,
    [ ! -f $ANNOT_FILE ] && { echo "$ANNOT_FILE file not found"; exit 99; }
    
    COUNT=0
    while read flname sid xmin ymin width height
    do
        IFS='/' read -r -a array <<< "$flname"
        SUBJECT_DIR="${array[-2]}"

        if [ "${SUBJECT_DIR}" == "${SUBJECT_FOLDER}" ]; then
            mkdir -p ${OUT_PATH}/${SUBJECT_DIR}
            IMG_PATH=${DATA_PATH}/${array[-2]}/${array[-1]}
            IMG_OUT=${OUT_PATH}/${array[-2]}/${array[-1]}
            convert -crop ${width}x${height}+${xmin}+${ymin} ${IMG_PATH} ${IMG_OUT}
        fi

    done < $ANNOT_FILE
    IFS=$OLDIFS
}

crop_image () {
    DATA_PATH=$1
    OUT_PATH=$2
    ANNOT_FILE=$3

    OLDIFS=$IFS
    IFS=,
    [ ! -f $ANNOT_FILE ] && { echo "$ANNOT_FILE file not found"; exit 99; }
    
    COUNT=0
    while read flname sid xmin ymin width height
    do
        IFS='/' read -r -a array <<< "$flname"
        SUBJECT_DIR="${array[-2]}"
        mkdir -p ${OUT_PATH}/${SUBJECT_DIR}
        IMG_PATH=${DATA_PATH}/${array[-2]}/${array[-1]}
        IMG_OUT=${OUT_PATH}/${array[-2]}/${array[-1]}
        convert -crop ${width}x${height}+${xmin}+${ymin} ${IMG_PATH} ${IMG_OUT}

    done < $ANNOT_FILE
    IFS=$OLDIFS
}


# >>>>>> Set paths here <<<<<<
DATA_PATH=/home/renyi/arunirc/data1/datasets/vggface2/train
OUT_PATH=/home/renyi/arunirc/data1/datasets/vggface2/train-crop
ANNOT_FILE=/home/renyi/arunirc/data1/datasets/vggface2/vggface2_disk1.csv


COUNT=0
WAIT_COUNT=0
MAXPROG=`ls ${DATA_PATH} | wc -l`

for folder in `ls ${DATA_PATH}`; do
    
    # show progress
    ((WAIT_COUNT += 1))
    ((COUNT += 1))
    echo ${COUNT}
    echo -n "$((${COUNT}*100/${MAXPROG})) %     "
    echo -n R | tr 'R' '\r'

    crop_image ${DATA_PATH} ${OUT_PATH} ${ANNOT_FILE}

    # if [ ${WAIT_COUNT} -eq 20 ]; then
    #     # dont spawn more than 20 processes at a time
    #     echo 'waiting for 20 processes to finish'
    #     echo "Progress: $((${COUNT}*100/${MAXPROG})) %     "
    #     wait
    #     ((WAIT_COUNT=0))
    # fi

    # if [ ${COUNT} -ge 20 ]; then
    #     break
    # fi
done

wait
echo "Done cropping"

# mkdir -p ${OUT_PATH}
# date
# crop_image ${DATA_PATH} ${OUT_PATH} ${ANNOT_FILE}
# date