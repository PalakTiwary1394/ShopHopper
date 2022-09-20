import os
import csv
import numpy as np
import pandas as pd
from scipy.spatial import distance
from statistics import mean
from statistics import median


def compare_distance(color1, color2):
    r1_hex = '0x' + color1[1:3]
    r2_hex = '0x' + color2[1:3]
    r1_int = int(r1_hex, 16)
    r2_int = int(r2_hex, 16)

    g1_hex = '0x' + color1[3:5]
    g2_hex = '0x' + color2[3:5]
    g1_int = int(g1_hex, 16)
    g2_int = int(g2_hex, 16)

    b1_hex = '0x' + color1[5:7]
    b2_hex = '0x' + color2[5:7]
    b1_int = int(b1_hex, 16)
    b2_int = int(b2_hex, 16)

    col1 = np.array((r1_int, g1_int, b1_int))
    col2 = np.array((r2_int, g2_int, b2_int))

    dist2 = distance.euclidean(col1, col2)
    return dist2


def filter_color(str):
    temp = ''
    output = []
    p = 0
    while p < len(str):
        if str[p] == '#':
            temp += str[p: p+7]
            p = p + 7

            output.append(temp)
            temp = ''
        else:
            p = p + 1

    return output


'''
Pre
'''
path1 = "C:\\Users\\97902\\Desktop\\ShopHopper\\color\\ShopHopper\\ShopHopper\\colors_entireds_png.csv" # path of prediction results
# path1 = "C:\\Users\\97902\\Desktop\\ShopHopper\\color\\ShopHopper\\ShopHopper\\colors_entiredb.csv"
path2 = "C:\\Users\\97902\\Desktop\\ShopHopper\\color\\ShopHopper\\ShopHopper\\(V1.8)all_products_data_set.csv" # path of ground truth

# PREDICTION
with open(path1, newline='') as file:
    result_list = list(csv.reader(file))
    for row in result_list:
        temp = row[0][: -6]
        row[0] = temp

    pred = np.array(result_list) # csv -> ndarray

pred1 = pd.DataFrame(pred, columns=['FIL', 'COLOR1', 'COLOR2', 'COLOR3', 'COLOR4', 'COLOR5'])
pred1 = pred1.iloc[1:, :]
print(pred1)



# Generate ground truth colors csv
# with open(path2, "r", newline='', encoding="utf8") as source:
#     reader = csv.reader(source)
#
#     with open("ground_truth.csv", "w", newline='', encoding="utf8") as result:
#         writer = csv.writer(result)
#         for r in reader:
#             writer.writerow((r[1], r[19]))


# GROUND TRUTH
path3 = "C:\\Users\\97902\\Desktop\\ShopHopper\\color\\ShopHopper\\ShopHopper\\BackgroundRemoval - Copy\\ground_truth.csv"

ground_truth = pd.read_csv(path3)

with open(path3, newline='') as file:
    list3 = list(csv.reader(file))
    for row in list3:
        if len(row[1]) > 1:
            temp = filter_color(row[1])
            row[1] = temp

    ground_truth = np.array(list3) # csv -> ndarray

gt = pd.DataFrame(ground_truth, columns=['FIL', 'GT_COLOR'])
gt['GT_COLOR'].replace('', np.nan, inplace=True)
gt.dropna(subset=['GT_COLOR'], inplace=True)
gt = gt.iloc[1:, :]
print(gt)


combine = pd.merge(pred1, gt, on='FIL')
combine['result'] = ''
# print(combine)


# Set the threshold

# unorderedset = {0}
# colors = ['#000000', '#75ae61', '#ffffff', '#0000ff', '#808080', '#00ff00', '#ffd700', '#000080', '#a94cd8', '#964b00',
#           '#ff0000', '#c0c0c0', '#a85772', '#ffc0cb', '#aa907d'] # colors appears most in results
# for i in colors:
#     for j in colors:
#         temp_dist = compare_distance(i, j)
#         unorderedset.add(temp_dist)
#
# list_dist = list(unorderedset)
# list_dist = list_dist[1:]
# print('mean: ' + str(mean(list_dist)))
# print('median' + str(median(list_dist)))

# threshold = mean(list_dist)
# threshold = median(list_dist)
threshold = 205

for index, row in combine.iterrows():
    for i in range(len(row['GT_COLOR'])):
        dist1 = compare_distance(row['GT_COLOR'][i], row['COLOR1'])
        if dist1 < threshold:
            row['result'] = 1
            break
        dist2 = compare_distance(row['GT_COLOR'][i], row['COLOR2'])
        if dist2 < threshold:
            row['result'] = 1
            break
        dist3 = compare_distance(row['GT_COLOR'][i], row['COLOR3'])
        if dist3 < threshold:
            row['result'] = 1
            break
        dist4 = compare_distance(row['GT_COLOR'][i], row['COLOR4'])
        if dist4 < threshold:
            row['result'] = 1
            break
        dist5 = compare_distance(row['GT_COLOR'][i], row['COLOR5'])
        if dist5 < threshold:
            row['result'] = 1
            break

    if row['result'] != 1:
        row['result'] = 0

# print(combine)

# temp = combine['GT_COLOR'].value_counts(ascending=False)
# print(temp)


print(combine['result'].value_counts(normalize=True))

# print(compare_distance('#540101', '#f55540'))