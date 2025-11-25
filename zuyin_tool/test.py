import lmdb

env = lmdb.open(r"E:\python\lab_project\zuyin_tool\lmdb_output\train_lmdb", 
                readonly=True, lock=False)
with env.begin() as txn:
    v = txn.get(b'num-samples')
    print("num-samples raw:", v)
    if v: 
        print("num-samples int:", int(v))
    
    print("has image-000000000:", txn.get(b'image-000000000') is not None)
    print("has label-000000000:", txn.get(b'label-000000000') is not None)
    
    # 讀取第一個標籤看看
    label = txn.get(b'label-000000000')
    if label:
        print("first label:", label.decode('utf-8'))

env.close()
