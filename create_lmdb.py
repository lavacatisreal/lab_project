# create_lmdb.py
import os
import io
import lmdb
import json
import argparse
from PIL import Image
from tqdm import tqdm

def is_image_file(p):
    p_lower = p.lower()
    return p_lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"))

def load_pairs_from_dir(images_dir, label_file=None, recursive=True):
    """
    掃描資料夾取得 (image_path, label_str) 列表
    - 若提供 label_file（txt/json），優先以其為準（鍵為檔名或相對路徑）
    - 若未提供，嘗試從同名 .txt 或 _label.txt 旁邊找標籤（若沒有則空字串）
    """
    pairs = []
    index = 0
    label_map = {}

    if label_file:
        if label_file.lower().endswith(".json"):
            with open(label_file, "r", encoding="utf-8") as f:
                label_map = json.load(f)
        else:
            # 文本標註：每行 "relative_path<TAB>label"
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip("\n\r")
                    if not line:
                        continue
                    if "\t" in line:
                        k, v = line.split("\t", 1)
                    elif "," in line:
                        k, v = line.split(",", 1)
                    else:
                        # 單欄格式時整行作為檔名，label 空
                        k, v = line, ""
                    label_map[k] = v

    for root, _, files in os.walk(images_dir):
        if not recursive and (os.path.abspath(root) != os.path.abspath(images_dir)):
            continue
        for fn in files:
            if not is_image_file(fn):
                continue
            img_path = os.path.join(root, fn)
            # 相對於 images_dir 的索引鍵
            rel_key = os.path.relpath(img_path, images_dir).replace("\\", "/")

            # 先用標註檔
            if rel_key in label_map:
                label = label_map[rel_key]
            elif fn in label_map:
                label = label_map[fn]
            else:
                # 嘗試同名 .txt
                alt_txt = os.path.splitext(img_path)[0] + ".txt"
                if os.path.exists(alt_txt):
                    with open(alt_txt, "r", encoding="utf-8") as f:
                        label = f.read().strip()
                else:
                    label = ""

            pairs.append((img_path, label))
            index += 1
    return pairs

def write_lmdb(pairs, out_dir, map_size_gb=4, jpeg_quality=95):
    """
    以本專案約定鍵名寫入 LMDB：
    - num-samples: 總樣本數（字串）
    - image-%09d: 影像 bytes（建議統一壓縮格式，這裡用 JPEG）
    - label-%09d: UTF-8 標籤
    """
    os.makedirs(out_dir, exist_ok=True)
    map_size = int(map_size_gb * (1024 ** 3))

    env = lmdb.open(
        out_dir,
        map_size=map_size,
        subdir=True,
        readonly=False,
        meminit=False,
        map_async=True,
        lock=True,  # 寫入時啟用鎖
    )

    cnt = 0
    with env.begin(write=True) as txn:
        for idx, (img_path, label) in enumerate(tqdm(pairs, desc="Writing LMDB"), start=1):
            # 讀圖並以 JPEG 統一編碼（可改為 PNG）
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] skip unreadable image: {img_path} ({e})")
                continue

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            img_bytes = buf.getvalue()

            img_key = f"image-{idx:09d}".encode()
            label_key = f"label-{idx:09d}".encode()

            txn.put(img_key, img_bytes)
            txn.put(label_key, label.encode("utf-8"))
            cnt += 1

        # 寫入總數
        txn.put(b"num-samples", str(cnt).encode())

    # 同步與壓縮
    env.sync()
    env.close()
    return cnt

def sanity_check(lmdb_dir, n=3):
    import lmdb
    from PIL import Image
    import six

    env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        ns = txn.get(b"num-samples")
        print("num-samples:", ns)
        total = int(ns) if ns else 0
        for i in range(1, min(n, total) + 1):
            ib = txn.get(f"image-{i:09d}".encode())
            lb = txn.get(f"label-{i:09d}".encode())
            ok = ib is not None and lb is not None
            print(f"[{i}] keys ok:", ok, "label:", (lb.decode("utf-8") if lb else None))
            if ib:
                buf = io.BytesIO(ib)
                try:
                    img = Image.open(buf).convert("RGB")
                    print(f"   image size: {img.size}")
                except Exception as e:
                    print(f"   image decode failed: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True, help="影像根目錄（遞迴掃描）")
    ap.add_argument("--labels", type=str, default=None, help="標註檔（json 或 txt，txt 一行 'path<TAB>label'）")
    ap.add_argument("--out", type=str, required=True, help="輸出 LMDB 目錄（會建立 data.mdb/lock.mdb）")
    ap.add_argument("--map_size_gb", type=float, default=4.0, help="LMDB map_size（GB）")
    ap.add_argument("--jpeg_quality", type=int, default=95, help="儲存到 LMDB 的 JPEG 品質")
    ap.add_argument("--recursive", action="store_true", help="遞迴掃描 images_dir")
    args = ap.parse_args()

    pairs = load_pairs_from_dir(args.images_dir, args.labels, recursive=args.recursive)
    if len(pairs) == 0:
        print("[ERROR] 找不到影像/標註，請確認 images_dir 與 labels")
        return

    print(f"[INFO] 準備寫入 {len(pairs)} 筆到 {args.out}")
    cnt = write_lmdb(pairs, args.out, map_size_gb=args.map_size_gb, jpeg_quality=args.jpeg_quality)
    print(f"[INFO] 完成寫入，共 {cnt} 筆")
    print("[INFO] Sanity check:")
    sanity_check(args.out, n=3)

if __name__ == "__main__":
    main()
