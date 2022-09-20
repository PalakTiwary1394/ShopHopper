import os
from sklearn.cluster import KMeans
from collections import Counter
import cv2
import csv


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def color_detect(input_path, name):
    image = get_image(input_path)
    number_of_colors = 5  # 3
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
    #data = [name, hex_colors[0], hex_colors[1], hex_colors[2]]

    with open('colors_entireds_png.csv', 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(data)



if __name__ == "__main__":
    root = "C:\\Users\\97902\\Desktop\\ShopHopper\\color\\ShopHopper\\ShopHopper\\bg_removed_png"
    field_names = ["FILE_NAME", "COLOR1", "COLOR2", "COLOR3", "COLOR4", "COLOR5"]
    with open('colors_entireds_png.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(field_names)
        # writer.writerow(data)

        number = 0

    for path, subdirs, files in os.walk(root):
        for name in files:
            try:
                input_path = os.path.join(path, name)
                # print("Input path", input_path)
                color_detect(input_path, name)
                number += 1
                if number%200 == 0:
                    print("done")
            except:
                pass