import os
import lmdb
import cv2
import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
# import time

def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

# 關鍵修改：將 cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
# 改為使用 open() + np.frombuffer() + cv2.imdecode() 組合。
def load_image(img_path):
    """讀取圖片，前處理後轉 PNG bytes；不破壞淡字"""
    try:
        # ✅ 修改：使用 Unicode 安全的讀取方式
        with open(img_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None

        # 去除背景不均 (保留淡筆畫)
        bg = cv2.GaussianBlur(img, (51, 51), 0)
        img = cv2.addWeighted(img, 1.5, bg, -0.5, 0)

        # 局部對比增強 CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        # 銳化 (unsharp mask)
        blur_small = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.addWeighted(img, 1.2, blur_small, -0.2, 0)

        # 限制範圍 (避免超亮超暗像素)
        img = np.clip(img, 0, 255).astype(np.uint8)

        _, img_bin = cv2.imencode('.png', img)
        return img_bin.tobytes()

    except Exception as e:
        return None

# def load_image(img_path):
#     """讀取圖片，前處理後轉 PNG bytes；不破壞淡字"""
#     try:
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             return None

#         # 去除背景不均 (保留淡筆畫)
#         bg = cv2.GaussianBlur(img, (51, 51), 0)
#         img = cv2.addWeighted(img, 1.5, bg, -0.5, 0)

#         # 局部對比增強 CLAHE
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         img = clahe.apply(img)

#         # 銳化 (unsharp mask)
#         blur_small = cv2.GaussianBlur(img, (3, 3), 0)
#         img = cv2.addWeighted(img, 1.2, blur_small, -0.2, 0)

#         # 限制範圍 (避免超亮超暗像素)
#         img = np.clip(img, 0, 255).astype(np.uint8)

#         cv2.imwrite('./test.png', img)
#         # time.sleep(1)
#         _, img_bin = cv2.imencode('.png', img)
#         return img_bin.tobytes()

#     except Exception:
#         print(f"⚠️ Failed to process {img_path}")
#         return None



def create_lmdb(output_path, samples, num_workers=8):
    """
    samples: list of (image_path, label)
    """
    if os.path.exists(output_path):
        print(f"⚠️ Folder {output_path} already exists, removing...")
        shutil.rmtree(output_path)

    # 新增：確保父目錄存在
    parent_dir = os.path.dirname(output_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"✅ Created parent directory: {parent_dir}")

    os.makedirs(output_path, exist_ok=True)
    # 改為更合理的大小（20GB 足夠儲存 10 萬張圖片）
    env = lmdb.open(output_path, map_size=21474836480)  # 20GB

    cache = {}
    cnt = 0
    skipped = []

    # 多線程讀圖
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_image, img_path): (img_path, label) for img_path, label in samples}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Writing {output_path}"):
            img_bytes = future.result()
            img_path, label = futures[future]
            if img_bytes is None:
                skipped.append(img_path)
                continue

            # LMDB key 保證連續
            image_key = f"image-{cnt:09d}"
            label_key = f"label-{cnt:09d}"
            cache[image_key] = img_bytes
            cache[label_key] = label.encode()

            if cnt % 1000 == 0:
                write_cache(env, cache)
                cache = {}

            cnt += 1

    # 存最後一批
    write_cache(env, cache)

    # 改為（正確）
    with env.begin(write=True) as txn:
        txn.put("num-samples".encode(), str(cnt).encode())

    # 記錄 dataset 大小
    # with env.begin(write=True) as txn:
    #     txn.put("num-samples".encode(), str(cnt - 1).encode())

    env.close()
    print(f"✅ Created {output_path} with {cnt-1} samples")
    if skipped:
        print(f"⚠️ Skipped {len(skipped)} images due to read errors:")
        for path in skipped:
            print("   ", path)


def main():
    dataset_root = r"E:\python\lab_project\zuyin_tool\augmented_images"
    train_ratio = 0.8   # 80% train / 20% val

    all_samples = []
    for label in os.listdir(dataset_root):
        class_dir = os.path.join(dataset_root, label)
        if not os.path.isdir(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(class_dir, img_file)
                all_samples.append((img_path, label))

    print(f"📊 Total samples: {len(all_samples)}")

    # 打亂
    random.shuffle(all_samples)

    # train / val split
    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"👉 Train: {len(train_samples)}, Val: {len(val_samples)}")

    create_lmdb(r"E:\python\lab_project\zuyin_tool\lmdb_output\train_lmdb", train_samples)
    create_lmdb(r"E:\python\lab_project\zuyin_tool\lmdb_output\val_lmdb", val_samples)


if __name__ == "__main__":
    main()
