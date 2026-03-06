#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import random

CALIB_DIR = Path("/mnt/c/Users/35207/Desktop/EXAMPLE/SP14/calib_images")
OUT_PATH = Path("/home/caribbean/build/calib_set_1024.npy")

TARGET_W = 800
TARGET_H = 800
MAX_IMAGES = 1024  

def main():
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        img_paths.extend(CALIB_DIR.glob(ext))

    img_paths = sorted(img_paths)
    if not img_paths:
        raise RuntimeError(f"在 {CALIB_DIR} 下没有找到任何 jpg/png 图片")

    # 打乱，取前 MAX_IMAGES 张
    random.shuffle(img_paths)
    img_paths = img_paths[:MAX_IMAGES]

    imgs = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"警告：无法读取图片 {p}，跳过")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (TARGET_W, TARGET_H))
        img = img.astype(np.float32) / 255.0  # ★ 和 ONNX 一样 /255
        imgs.append(img)

    if not imgs:
        raise RuntimeError("没有任何有效的校准图片")

    calib_array = np.stack(imgs, axis=0)  # [N,800,800,3]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, calib_array)
    print("保存校准集到：", OUT_PATH)
    print("calib_array.shape =", calib_array.shape, "dtype =", calib_array.dtype)

if __name__ == "__main__":
    main()
