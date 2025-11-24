config = {
    'exp_name' : 'kid_ver_5_with_adult_pre_train_model_1',
    'epoch' : 100,
    'lr' : 1,
    'batch' : 128,
    'val_frequency' : 100,
    'test' : False,
    'resume_model' : '/content/drive/MyDrive/improve_FudanOCR/final_model/final_model.pth',
    'train_dataset' : '/content/drive/MyDrive/improve_FudanOCR/lmdb/train',
    'validation_dataset': '/content/drive/MyDrive/improve_FudanOCR/lmdb/val',
    'inference_dataset': 'judge_data',
    'judge_dataset': '',
    'schedule_frequency' : 10,
    'imageH' : 32,
    'imageW' : 256,
    'encoder' : 'resnet',
    'decoder' : 'transformer',
    'alpha_path' : '/content/lab_project/data/decompose_v1.txt', #常見字?
    'radical_path': '/content/lab_project/data/all_bpmf.txt', #常見部首?
    'decompose_path': '/content/lab_project/data/bpmf_ids_v2.txt',   #文字檔ids?
    'pre-train_model': '/content/drive/MyDrive/improve_FudanOCR/pre_train_model/adult_pre_train_model.pth',
    'stn': False,
    'constrain': False,
    'char_len' : 60,
}

