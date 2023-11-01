#!/bin/bash

DATA=$1
if [[ $DATA == "alfred" ]]; then
    FILE=(alfred_test alfred_train_0913)
elif [[ $DATA == "babyai" ]]; then
    FILE=(babyai_test babyai_train_0910)
elif [[ $DATA == "iqa" ]]; then
    FILE=(iqa_test iqa_contains_0915 iqa_counts_0915 iqa_exist_0915)
elif [[ $DATA == "rearrange" ]]; then
    FILE=(rearrange_test rearrange_0915)
elif [[ $DATA == "teach" ]]; then
    FILE=(teach_test teach_train_0912)
elif [[ $DATA == "cwah" ]]; then
    FILE=(cwah_test)
else
    echo "Usage: bash download.sh <dataset>"
    echo "Available datasets are: alfred, babyai, iqa, rearrange, teach, cwah"
    exit 1
fi

BASE_URL=https://bigai-nlco.s3.ap-southeast-1.amazonaws.com/langsuite/
for f in ${FILE[@]}; do
    echo "Downloading $f"
    URL=${BASE_URL}$f.zip
    TARGET_DIR=./data/$DATA/$f
    ZIP_FILE=./data/$DATA/$f.zip
    mkdir -p $TARGET_DIR
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d $TARGET_DIR
done
