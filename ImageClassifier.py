
import numpy as np
import pathlib
from fastai.vision.all import *

from fastai.vision.data import ImageDataLoaders
from fastai.vision.all import *
from fastai.imports import *
import pandas as pd
# matplotlib inline
from datetime import datetime

def transfer_learn(learn:Learner, name:Path, device:torch.device=None):
    """_summary_
        Load model `name` from `self.model_dir` using `device`, defaulting to `self.dls.device`.
    Args:
        learn (Learner): the machine learning model that is to be retrained
        name (Path): the learner path ('.pth' file), for example: "model/stage1_SH_08_21.pth".
        device (torch.device, optional): _description_. Defaults to None.

    Returns:
        _type_: return the learner(the model)
    """
    
    if device is None: device = learn.dls.device
    learn.model_dir = Path(learn.model_dir)
    if (learn.model_dir/name).with_suffix('.pth').exists(): model_path = (learn.model_dir/name).with_suffix('.pth')
    else: model_path = name
    new_state_dict = torch.load(model_path, map_location=device)['model']
    learn_state_dict = learn.model.state_dict()
    for name, param in learn_state_dict.items():
        if name in new_state_dict:
            input_param = new_state_dict[name]
            print("input_param.shape: ", input_param.shape)
            print("param.shape: ", param.shape)
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                print('Shape mismatch at:', name, 'skipping')
        else:
            print(f'{name} weight of the model not in pretrained weights')
    learn.model.load_state_dict(learn_state_dict)
    return learn


class ImageClassifier:
    """the image classifier model.
    """

    def load_ML_model(self, model_path:str):
        """load the whole model, which is in a '.pkl' file, it can predict images immediately.

        Args:
            model_path (str): raletive path of the '.pkl' file. For example: "model/SH_Model_2022-08-21.pkl"
        """
        # model_path example: "model/SH_Model_2022-08-21.pkl"
        self.model = load_learner(model_path)
        

    def predict(self, img_path:str):
        """predict the image's, will print out the product label and the corresbonding confidence.

        Args:
            img_path (str): the relative path. For example: "test_images/4332405260311-2.jpg"

        Returns:
            return the predicted label and the probability.
        """
        #full_path = os.path.join(image_file_path,image_file_name)
        # print("image_path is: ", img_path)
        img = PILImage.create(img_path)
        # apply the model to the image
        pred_class, pred_idx, prob = self.model.predict(img)
        probability = f'{prob[pred_idx]:.04f}'
        predict_string = f'Prediction: {pred_class}\nProbability: {prob[pred_idx]:.04f}'   
        # print(predict_string)
        return pred_class, probability
    
    def prepare_the_training_set(self, training_path:str):
        """drop the corrupted images, print the information of the training set, and num of images droped

        Args:
            training_path (str): the relative path of the csv file that stores the path of the images and their labels.
                                For example: "clothes-categories/SHV2_3_training.csv"
        """
        self.PATH = ""
        self.TRAINING_PATH = training_path # example: "clothes-categories/SHV2_3_training.csv"
        self.training_data = pd.read_csv(self.TRAINING_PATH, names = ["image_name", "category_name"], index_col = False)
        df = self.training_data
        df = df.reset_index(drop = True)

        print("number of images: ", len(df))
        print("\n")
        print(df['category_name'].value_counts())
        print("\n")

        # Some items have corrupted images, here we drop them
        test_dict = {}

        for row in df.iterrows():
            test_dict[row[1]['image_name']] = row[1]['category_name']
        counter = 0
        for key, value in test_dict.items():
            try:
                img=Image.open(self.PATH + key)        

            except:
                self.training_data = self.training_data.drop(self.training_data[self.training_data['image_name'] == key].index)
                counter += 1
        
        print("number of bad images that have been hadeled:", counter)    

    def retrain_model(self, learner_path:str, epoch_num:int):
        """retrain the model with the given epoch_num, the learning rate is set

        Args:
            learner_path (str): relative path of the model that should be trained, '.pth' file,
                                For example: "models/stage1_SH_08_21.pth"
            epoch_num (int): the number of epoches to be trained.
        """
        # the dls
        self.training_data = ImageDataLoaders.from_df(self.training_data,
                                                        item_tfms=Resize(300),
                                                        batch_tfms=aug_transforms(size=224, min_scale=0.9),
                                                        valid_pct=0.1,
                                                        splitter = RandomSplitter(seed = 42),
                                                        number_workers = 0)
        # Here we get resnet34, which is pretrained and train it using our train set
        # learner_path example: "model/stage1_SH_08_21.pth"
        metrics = [accuracy]

        self.learn = cnn_learner(self.training_data, resnet34, metrics=accuracy, pretrained=True)
        
        # use transfer learning to load the model
        self.learn = transfer_learn(self.learn, learner_path)
        
        # for example: epoch_num = 5; learning_rate: lr_max=slice(1e-5, 1e-4)
        self.learn.fit_one_cycle(epoch_num, lr_max=slice(1e-5, 1e-4)) 

    def save_the_model(self):
        """ save the model automatically with the date in the name.
            the models are under 'models/' directory. 
        """
        PATH = "models/"
        MODEL_PATH = PATH + "SH_Model_" + str(datetime.date(datetime.now())) + ".pkl"
        self.learn.export(MODEL_PATH)

        LEARNER_PATH = "stage1_SH_" + str(datetime.date(datetime.now()))
        self.learn.save(LEARNER_PATH)

