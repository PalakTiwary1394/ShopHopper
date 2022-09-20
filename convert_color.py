#################################################################
# This script can detect colors of images
# input: an image
# output: colors with highest frequencies


# ATTENTION: ************************
# in colormath. color_diff.py. delta_e_cie2000 function:
#   replace "return numpy.asscalar(delta_e)"  with  "return delta_e"

#################################################################
import pandas as pd
import numpy as np
import re
import time
import requests
from bs4 import BeautifulSoup
import multiprocessing as mp
from rembg import remove

# from pandarallel import pandarallel
start_time = time.time()

# the main_categories with all sub_categories dict
products_to_all = {}
# the main_categories relate to numbers dict
main_categories_map_to_num = {}
# all categories of product including the name of main_cat
all_cat = set()
# each specific item maps to the main_categories_number
specific_products_map_to_num = {}
# generate label_2nd class from specific labels
label3_to_label2_map = {}
# generate colorname_to_hex from our predetermined colors name json file
colorname_to_hex_map = {}


def parallelize_dataframe(df: pd.DataFrame, func, n_cores=mp.cpu_count()) -> pd.DataFrame:
    """This is a parallelize calculated function

    Args:
        df (pd.DataFrame): the dataframe of pandas, we are going to use.
        func (_type_): for each line calculation, we are going to implement.
        n_cores (_type_, optional): Defaults to mp.cpu_count().

    Returns:
        pd.DataFrame: After calculation, the return new dataframe.
    """
    df_split = np.array_split(df, n_cores)
    pool = mp.Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


#######################################################################################################
# Color Section
######################################################################################################

def color_text_cleaner(colors: str) -> str:
    """
    Input: colors(str), format {"Colours_1", "Colours_2"}
    Function: text processing and get the colour information separated by comma ','
    Output: Colours(str), format "colours_1, colours_2"
    """
    temp = (colors.lower().removeprefix("{").removesuffix("}"))
    res = re.sub('"', "", re.sub("/", ",", temp))
    return res


# reset df all "colors" colomn format

def reset_colors_format(df: pd.DataFrame):
    '''
    Input: panda.DataFrame
    Function: By using tc function above to reset all cells in 'colors' colomn
    Output: None
    '''
    for row in range(df.shape[0]):
        if not isinstance(df.loc[row]["colors"], str):
            df.loc[row, 'colors'] = ""
            continue
        e = df.loc[row]["colors"]
        c = color_text_cleaner(e)
        df.loc[row, 'colors'] = c


def requery_color_name(df: pd.DataFrame) -> None:
    '''
    Input: dataframe (panda.Dataframe)
    Function: check color cells inside the dataframe row by row, send requery to the colornames.org to get colour hex and save it as the set()
    Output: None
    '''
    for row in range(df.shape[0]):

        res = set()
        pattern = set()

        colors_str = df.loc[row]["colors"]
        if colors_str == "":
            continue
        colors_list = colors_str.split(",")

        # print(row, colors_list)
        for item in colors_list:
            # Method 1: if we can find the colors_name locally, then we just use this color name and hexCode
            if colorname_to_hex_map.get(item) != None:
                colorname = " ".join([e.capitalize() for e in item.split(" ")])
                res.add(colorname + " (" + colorname_to_hex_map.get(item) + ")")
            else:
                # Method 2: if we can not find the colors_name locally, we will send query to the website colorsname.org to see if they have the name. Else we will treat it as pattern info
                URL = "https://colornames.org/search/results/?type=exact&query="
                URL += item
                try:
                    page = requests.get(URL)
                    soup = BeautifulSoup(page.content, "html.parser")
                    results = soup.find_all("a", class_="button is-fullwidth freshButton")

                    # the first result which has the highest vote
                    content = results[0].find("span").text.strip()
                    res.add(content)
                except:
                    pattern.add(item)
        # res, pattern = df['colors'].apply(requery_func, axis=1)
        df.loc[row, "color_info"] = ",".join(res)
        df.loc[row, "pattern_info"] = ",".join(pattern)


from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from PIL import ImageColor


class Color:
    """
    Class Color: for encapsulate all related color infomation.
    """

    def __init__(self, color_string: str):
        assert color_string != None
        color_info = color_string.split(" (")
        assert len(color_info) == 2
        self.name = color_info[0]
        self.hex_code = color_info[1].removesuffix(")")

    def get_rgb(self) -> tuple:
        assert self.hex_code != None
        return ImageColor.getcolor(self.hex_code, "RGB")

    def get_info(self) -> str:
        return self.name + " " + self.hex_code


def difference(color_1: Color, color_2: Color) -> float:
    """
    Input: format("color_name": "color_hex_code"), for example, color_1 = (Color class)Navy (#000080), color_2 = (Color class)Blue (#0000ff)
    Function: calculate delta_e_cie2000 color distance
    Output: Float, the distance
    """
    assert color_1 != None and color_2 != None, "color_1 or color_2 is Nonetype"
    color1_set = color_1.get_rgb()
    color1_rgb = sRGBColor(color1_set[0], color1_set[1], color1_set[2])
    color2_set = color_2.get_rgb()
    color2_rgb = sRGBColor(color2_set[0], color2_set[1], color2_set[2])
    # convert from RGB to lab color space
    color1_lab = convert_color(color1_rgb, LabColor)
    color2_lab = convert_color(color2_rgb, LabColor)
    return delta_e_cie2000(color1=color1_lab, color2=color2_lab)


def get_closest_color(unknown_color: Color) -> int:
    """
    Input: (Color class) to_be_determined_color
    Function: By Comparing with colors_distance in the predetermined color list, we extract the closest main_color.
    Output: The cloest color in the list
    """
    assert unknown_color != None
    # sort dict {color_num: int : color_distacne: float}
    temp_distance_map = {}
    main_color = {1: 'Black (#000000)', 2: 'Deep Blue (#0000FF)', 3: 'Light Blue (#7eb1f7)', 4: 'Mid Blue (#009dff)',
                  5: 'Deep Brown (#783000)', 6: 'Light Brown (#d94800)', 7: 'Mid Brown (#bf6237)',
                  8: 'Deep Green (#008a29)', 9: 'Mid Green (#21cf2a)', 10: 'Light Green (#8aff73)',
                  11: 'Grey (#bdbdbd)', 12: 'Deep Orange (#ff8000)', 13: 'Light Pink (#e0baaf)', 14: 'Mid Pink (#ff4d6c)',
                  15: 'Deep Pink (#d63c57)', 16: 'Deep Purple (#800080)', 17: 'Mid Purple (#d126d1)', 18: 'Light Purple (#f05df0)',
                  19: 'Mid Red (#ff0000)', 20: 'Deep Red (#ab0000)', 21: 'White (#ffffff)',
                  22: 'Light Yellow (#ffff78)', 23: 'Mid Yellow (#ffff00)', 24: 'Deep Yellow (#b5b521)'}

    # main_color = {1: 'Black (#000000)', 2: 'Blue (#3d5bb3)', 3: 'Brown (#964b00)',
    #               4: 'Green (#00ff00)',
    #               5: 'Grey (#9c9c9c)', 6: 'Orange (#ff8000)', 7: 'Pink (#ffc0cb)', 8: 'Purple (#800080)',
    #               9: 'Red (#ff0000)', 10: 'Tan (#d2b48c)', 11: 'White (#ffffff)', 12: 'Yellow (#ffff00)'}

    for k, v in main_color.items():
        dist = difference(unknown_color, Color(v))
        temp_distance_map.update({k: dist})

    # print(temp_distance_map)
    sorted_dict = dict(sorted(temp_distance_map.items(), key=lambda item: item[1]))

    min1 = min(sorted_dict, key=sorted_dict.get)
    # print(min1)

    return min(sorted_dict, key=sorted_dict.get)


def from_color_info_to_color_num(color_info: str) -> list:
    """
    Input: (str) color_info
    Function: standarized from the color_info column to color_num
    Output: (str)all relative nums (in the relative order)
    """
    assert isinstance(color_info, str) and color_info != ""
    res = []
    color_info_list = color_info.split(",")

    for item in color_info_list:
        curr = Color(item)
        temp = get_closest_color(curr)
        res.append(get_closest_color(curr))


    return res


def colour_mapping(df_original: pd.DataFrame):
    """
    Input: pandas.DataFrame
    Function: By using color distance algorithm to standardize the color info, and map the res to our main pre-determined 14 colors.
    Output: None, but you can use the df.to_csv("temp.csv", index=False) to check the mapping result correctness
    """
    for row in range(df_original.shape[0]):
        color_info = df_original.loc[row]["color_info"]
        # print(color_info)
        # if color_info is empty
        if color_info == "":
            continue
        try:
            color_num_str = ",".join(str(n) for n in from_color_info_to_color_num(color_info))
            df_original.loc[row, "color_num"] = color_num_str
        except:
            print("Error: ", row, " color_info ", color_info)



'''
################################ code #########################################
'''
from sklearn.cluster import KMeans
from collections import Counter
import cv2


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def color_detect(input_path, name):
    """
        Input: input_path: image path including image name; name: image name
        Function: detect main colors in images, the number of colors can be changed
        Output: (list) list of detected colors
    """
    image = get_image(input_path)
    number_of_colors = 5
    modified_image = image.reshape(image.shape[0] * image.shape[1], 3)

    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)

    #print("Counts", counts)
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=False)}

    center_colors = clf.cluster_centers_  # to get the colors
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    # rgb_colors = [ordered_colors[i] for i in counts.keys()]
    data = [name, hex_colors[0], hex_colors[1], hex_colors[2], hex_colors[3], hex_colors[4]]

    # with open('colors_entireds_png.csv', 'a') as f:
    #     writer = csv.writer(f, lineterminator='\n')
    #     writer.writerow(data)
    return data


def list2str(data: list) -> str:
    """
        Input: (list) 5 colors formatted in Color class
        Function: convert "#123456" to "Unknown (#123456)"
        Output: (str) str of detected colors formatted in Color class
    """
    colorname = 'Unknown '
    colorstr = colorname + "(" + data[1] + ")," + colorname + "(" + data[2] + ")," + colorname + "(" + data[3] + ")," + colorname + "(" + data[4] + ")," + colorname + "(" + data[5] + ")"
    return colorstr


def colordetectionprocess(input_path: str, name: str) -> str:
    """
        Input: (str, str) image input_path, image name
        Function: process of color detection
        Output: (str) keys in main color dictionary
    """
    data = color_detect(input_path, name)
    # print(data)
    strcolor = list2str(data)
    color_info = [strcolor]
    # print(color_info)
    df = pd.DataFrame(color_info, columns=['color_info'])
    colour_mapping(df)
    color_num = df.loc[0, "color_num"]
    # print(color_num)

    return color_num


def num2color(num: str) -> list:
    """
    Input: (str) color_num, eg. 1, 2, 1
    Function: convert colornum to names of colors detected
    Output: (list) detected colors in list
    """
    # main_color = {1: 'Black (#000000)', 2: 'Blue (#3d5bb3)', 3: 'Brown (#964b00)', 4: 'Green (#00ff00)',
    #               5: 'Grey (#9c9c9c)', 6: 'Orange (#ff8000)', 7: 'Pink (#ffc0cb)', 8: 'Purple (#800080)',
    #               9: 'Red (#ff0000)', 10: 'Tan (#d2b48c)', 11: 'White (#ffffff)', 12: 'Yellow (#ffff00)'}

    main_color = {1: 'Black (#000000)', 2: 'Blue (#0000FF)', 3: 'Blue (#ADD8E6)', 4: 'Blue (#009dff)',
                  5: 'Brown (#783000)', 6: 'Brown (#d94800)', 7: 'Brown (#bf6237)',
                  8: 'Green (#008a29)', 9: 'Green (#21cf2a)', 10: 'Green (#8aff73)',
                  11: 'Grey (#9c9c9c)', 12: 'Orange (#ff8000)', 13: 'Pink (#e0baaf)',
                  14: 'Pink (#ff4d6c)',
                  15: 'Pink (#d63c57)', 16: 'Purple (#800080)', 17: 'Purple (#d126d1)',
                  18: 'Light Purple (#f05df0)',
                  19: 'Red (#ff0000)', 20: 'Red (#ab0000)', 21: 'White (#ffffff)',
                  22: 'Yellow (#ffff78)', 23: 'Yellow (#ffff00)', 24: 'Yellow (#b5b521)'}

    colorlist = num.split(',')
    colorstr = []
    for color in colorlist:
        color_int = int(color)
        val = main_color[color_int]
        colorname = val[:-10]
        colorstr.append(colorname)

    return colorstr


'''
################################ main #########################################
'''

if __name__ == "__main__":
    # image path including image name
    input_path = "test_images/8862193862-1.jpg"
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

    color_num = colordetectionprocess(bg_removed_path, name)

    colorresult = num2color(color_num)
    print(colorresult)


# black -> purple
# light blue -> grey
# light brown -> grey
# green -> grey
# grey -> tan
# pink -> red






# test example
# color_info = ['Navy (#000080), Blue (#0000ff)', 'Blue (#0000ff)', 'White (#de8e9c)']
# df = pd.DataFrame(color_info, columns=['color_info'])
# colour_mapping(df)
# df.to_csv("temp.csv", index=False)