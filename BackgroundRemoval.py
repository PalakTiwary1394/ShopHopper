import os
from rembg import remove

root = "C:\\Users\\97902\\Desktop\\ShopHopper\\color\\ShopHopper\\ShopHopper\\png_imgs2"
# 14160723985
i = 0

for path, subdirs, files in os.walk(root):
    for name in files:
        input_path = os.path.join(path, name)
        # print("Input path ", input_path)
        subfolders = input_path.split('\\')

        output_path= "C:\\Users\\97902\\Desktop\\ShopHopper\\color\\ShopHopper\\ShopHopper\\bg_removed_png"
        #output_path = 'SH_Dataset(bgrmvd)'
        try:
            output_path_foldr = output_path + '\\' + subfolders[-2]
            os.mkdir(output_path_foldr)
        except:

            pass
        output_path += '\\' + subfolders[-2] + '\\' + subfolders[-1]
        # output_path += '\\' + subfolders[-1]

        try:
            with open(input_path, 'rb') as i:
                with open(output_path, 'wb') as o:
                    input_img = i.read()
                    output_img = remove(input_img)

                    o.write(output_img)
                    # print('done')

                    i = i + 1
                    if (i % 200 == 0):
                        print('done')

                    break
        except:
            pass