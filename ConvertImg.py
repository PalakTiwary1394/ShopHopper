from PIL import Image
import os

root = "C:\\Users\\97902\\Desktop\\ShopHopper\\color\\ShopHopper\\ShopHopper\\images"
# 14160723985

i = 0

for path, subdirs, files in os.walk(root):
    for name in files:
        input_path = os.path.join(path, name)
        # print("Input path ", input_path)
        subfolders = input_path.split('\\')

        output_path= "D:\\ShopHopper\\png_imgs2"
        #output_path = 'SH_Dataset(bgrmvd)'
        try:
            output_path_foldr = output_path + '\\' + subfolders[-2]
            os.mkdir(output_path_foldr)
        except:
            pass
        output_path += '\\' + subfolders[-2] + '\\' + subfolders[-1]
        # print("Output path ", output_path)
        output_path = output_path[: -3] + 'png'
        # print("Output path ", output_path)
        # output_path += '\\' + subfolders[-1]

        try:
            im = Image.open(input_path)
            im.save(output_path)

            # print('done')
            i = i + 1
            if(i % 200 == 0):
                print('done')

            break
        except:
            pass