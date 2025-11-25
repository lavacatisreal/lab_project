* .ttc檔 字體 給create.py生注音的dataset用的
* ids 跟 decompose 用來替換FudanOCR裡的一些文字檔
* split_to_2.py 把資料集分成 train 和 val 然後用成 .lmdb
* ids重複注音 把檔案讀進去list ids_list=list(set(ids_list))就可去掉
## split_to_2.py 
* dataset_root = "E:/python/lab_project/zuyin_tool/augmented_images"：資料集根目錄
* create_lmdb("E:/python/lab_project/zuyin_tool/lmdb_output/train_lmdb", train_samples)
* create_lmdb("E:/python/lab_project/zuyin_tool/lmdb_output/val_lmdb", val_samples)