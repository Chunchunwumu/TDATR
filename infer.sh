export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=105
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NET_GDR_LEVEL=PXB
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1
export MODULEPATH=/opt/tool/modulefiles_arm/
USE_NAIIVE=true
USE_OCR_PLUG=false

ROOT_DIR=PROJECT_PATH
echo "root_dir= " $ROOT_DIR
cd $root_dir

TEST_FILE=YOUR_TEST_IMAGE_PATH_LIST.json
minigpt4_v=TDATR
CFG_NAME=config
py_name=infer

CKPT_PATH="model.pt"
echo $CKPT_PATH

export CUDA_VISIBLE_DEVICES=DEVICE_ID
# only support single GPU
i=0
for CUR_TEST_FILE in ${TEST_FILE[@]}
do
  echo $CUR_TEST_FILE
  let i=i+1
  echo "run script..."
  python $ROOT_DIR/$minigpt4_v/eval/${py_name}.py \
      --config-dir $ROOT_DIR/configs/ \
      --config-name $CFG_NAME \
      common.user_dir=$ROOT_DIR/$minigpt4_v \
      common.npu=true \
      common.npu_jit_compile=false \
      +model.rectification_rotate_flag=false \
      +model.rectification_textline_height_flag=false \
      task.use_ocr_plug=$USE_OCR_PLUG \
      model.use_naiive=$USE_NAIIVE \
      model.lora.apply_lora=false \
      +model.use_vit_encoder=false \
      +model.use_donut_encoder=true \
      +model.use_cfgi=true \
      model.ckpt=$CKPT_PATH \
      generation.prompt_path=$CUR_TEST_FILE \
      generation.no_repeat_ngram_size=15 \
      generation.min_len=1 \
      generation.max_len=4096 \
      generation.temperature=0.5 \
      task.seed=42 \

done
