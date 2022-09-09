from Labels import Labels
from ImageClassifier import ImageClassifier

def return_labels():
    IC = ImageClassifier()
    IC.load_ML_model("TrainedModels/ICModels/SH_model08_21.pkl")

    l = Labels()
    #Hardcoding the values for now
    #these values will be later set by labels predicted by models
    l.colorList=["red", "black", "white"]
    l.gender="Male"
    l.type, confidence=IC.predict("test_images/8777602065-2.jpg")
    l.size="L"

    return l

label = return_labels()