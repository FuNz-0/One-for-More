#!/bin/bash

# --- 选配：如果你只用一个GPU，可以在这里指定 ---
# export CUDA_VISIBLE_DEVICES=0 

# ----------------------------------------------------
# 任务 1-3: MVTec 数据集 (Setting 2, 3, 4)
# ----------------------------------------------------
echo "--- 开始 MVTec Setting 2 训练 ---"
python scripts/train_mvtec.py --setting 2
if [ $? -ne 0 ]; then echo "MVTec Setting 2 失败，退出。"; exit 1; fi

echo "--- 开始 MVTec Setting 3 训练 ---"
python scripts/train_mvtec.py --setting 3
if [ $? -ne 0 ]; then echo "MVTec Setting 3 失败，退出。"; exit 1; fi

echo "--- 开始 MVTec Setting 4 训练 ---"
python scripts/train_mvtec.py --setting 4
if [ $? -ne 0 ]; then echo "MVTec Setting 4 失败，退出。"; exit 1; fi


# ----------------------------------------------------
# 任务 4-6: VisA 数据集 (Setting 1, 2, 3)
# ----------------------------------------------------
echo "--- 开始 VisA Setting 1 训练 ---"
python scripts/train_visa.py --setting 1
if [ $? -ne 0 ]; then echo "VisA Setting 1 失败，退出。"; exit 1; fi

echo "--- 开始 VisA Setting 2 训练 ---"
python scripts/train_visa.py --setting 2
if [ $? -ne 0 ]; then echo "VisA Setting 2 失败，退出。"; exit 1; fi

echo "--- 开始 VisA Setting 3 训练 ---"
python scripts/train_visa.py --setting 3
if [ $? -ne 0 ]; then echo "VisA Setting 3 失败，退出。"; exit 1; fi

echo "--- 🎉 所有训练任务成功完成！ ---"