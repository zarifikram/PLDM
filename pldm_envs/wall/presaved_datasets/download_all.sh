# change if you need to
export http_proxy=http://127.0.0.1:1080
export https_proxy=http://127.0.0.1:1087
OUTPUT_PATH=/data/zikram/PLDM/wall/

FILE_ID=1NwR-ui-akIgR2xcoJHYiHjOk9U5bYCa0
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${OUTPUT_PATH}/wall_data.tar.gz"

cd $OUTPUT_PATH

tar -xvf wall_data.tar.gz

rm wall_data.tar.gz
