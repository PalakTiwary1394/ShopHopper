from Labels import Labels
from ImageClassifier import ImageClassifier
from convert_color import colordetectionprocess, num2color
from rembg import remove
from TextClassfierWithNER import Product, auto_detect

def return_labels():
    IC = ImageClassifier()
    IC.load_ML_model("TrainedModels/ICModels/SH_model08_21.pkl")

    input_path = "test_images/8777602065-2.jpg"
    bg_removed_path = "bg_removed/"+input_path.split('/')[-1]
    name = input_path.split('/')[-1] # image name
    try:
        with open(input_path, 'rb') as i:
            with open(bg_removed_path, 'wb') as o:
                input_img = i.read()
                output_img = remove(input_img)

                o.write(output_img)
    except:
        pass

    product = Product()
    try:
        NLP_product = auto_detect("The Swingarm is a lightweight frame with classic styling. Its innovative double-action hinge system delivers uncompromising fit and function. Available in two sizes. Made in Italy, Lifetime Warranty Blue Light Blocking, Melanin-Infused Lenses Lightweight Bio-Resin Impact-Resistant Frames Includes Repreve Microfiber Bag: Produced from Recycled Plastics")
        product.set_pattern(NLP_product.get_pattern())
        product.set_gender(NLP_product.get_gender())
        product.set_size(NLP_product.get_size())
    except:
        product.set_pattern(("pattern", "HARDCODED"))
        product.set_gender("male/HARDCODED")
        product.set_size(("large", "HARDCODED"))
        print("NLP Error")

    #Hardcoding the values for now
    #these values will be later set by labels predicted by models
    color_num = colordetectionprocess(bg_removed_path, name)
    product.set_color(set(num2color(color_num))) # num2color returns a list, for example:['Green', 'Green', 'Black', 'Purple', 'Green']
    
    # l.gender="Male"
    product_type, confidence = IC.predict(input_path)
    product.set_types(IC.predict(input_path))
    # l.type, confidence=IC.predict(input_path)

    # l.size="L"
    product.info()

    return product

label = return_labels()