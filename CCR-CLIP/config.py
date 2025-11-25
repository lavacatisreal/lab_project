config = {
    'epoch': 40,
    'train_dataset': '/content/drive/MyDrive/improve_FudanOCR/data/lmdb/adult_data/train',
    'test_dataset': '/content/drive/MyDrive/improve_FudanOCR/data/lmdb/adult_data/val',
    'batch': 128,
    'imageW': 128,
    'imageH': 128,
    'alphabet_path': '/content/lab_project/CCR-CLIP/data/radical_alphabet_27533_benchmark.txt',
    'decompose_path': '/content/lab_project/CCR-CLIP/data/decompose_27533_benchmark.txt',
    'max_len': 30,
    'lr': 1e-4,
    'exp_name': 'adult_with_notation',
}