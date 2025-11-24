# input: a image, the answer children writed
# output: the answer is correct or not 

# 1. read the image children answered (which type?)
# 2. preprocess the model and necessary variable
# 3. load the correct answer (which type? .txt?)
# 4. predict the image with model
# 5. compare to the correct answer
# 6. return True or False

from inference import inference
from inference import preprocessing
from config import config

# def judge():

if __name__ == "__main__":
    model, alphabet, clip_model, text_features, dataloader = preprocessing(config['judge_dataset'])
    results = inference(model, alphabet, clip_model, text_features, dataloader)
    print(results)