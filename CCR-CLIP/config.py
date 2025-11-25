config = {
    'epoch': 40,
    'train_dataset': '/content/drive/MyDrive/improve_FudanOCR/lmdb/train',
    'test_dataset': '/content/drive/MyDrive/improve_FudanOCR/lmdb/val',
    # 'batch': 128,
    'batch': 8,
    'imageW': 128,
    'imageH': 128,
    'alphabet_path': '/content/lab_project/CCR-CLIP/data/decompose_v1.txt',
    'decompose_path': '/content/lab_project/CCR-CLIP/data/bpmf_ids_v2.txt',
    'max_len': 30,
    'lr': 1e-4,
    'exp_name': 'adult_with_notation',
}