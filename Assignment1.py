# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 23:05:46 2020

@author: Wei Wang, Z5200638
"""

# ......IMPORT .........
import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from itertools import product


''' helper functions '''


def save_img(data: np.ndarray, title: str, filename: str):
    body = cv2.copyMakeBorder(data, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
    head = np.full((30, body.shape[1]), 255, dtype='uint8')
    cv2.putText(head, title, (30, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))

    img = cv2.vconcat((head, body))
    cv2.imwrite(filename, img)


def median_filter(data: np.ndarray, ksize: int) -> np.ndarray:

    # extend the borader of the image by 1
    border = ksize // 2
    extended = np.empty(
        (data.shape[0] + border * 2, data.shape[1] + border * 2), dtype='uint8')

    for line_number in range(len(data)):
        cur = data[line_number]
        head = [cur[0] for i in range(border)]
        tail = [cur[-1] for i in range(border)]
        extended[line_number + border] = np.concatenate([head, cur, tail])

    for i in range(border):
        extended[i] = extended[border]
        extended[-1 - i] = extended[-border - 1]

    # apply median convolution kernel to the original data
    row_range = range(border, extended.shape[0] - border)
    col_range = range(border, extended.shape[1] - border)
    for row, col in product(row_range, col_range):
        data[row - border, col - border] = np.median(
            extended[row - border: row + border + 1,
                     col - border: col + border + 1])

    return data


def task1(data: np.ndarray) -> (np.ndarray, float, list):
    '''
    This input data can only have one channel.

    Task1 will find the threshold of the image
    And then return the image processed into binary image, 
    threshold value and the values of threshold in each iteration
    '''

    # set initial thresholding value
    t = 127

    # record the value of t in each iteration
    t_history = [t]

    # non-trivial flag
    not_trivial = True

    # iteration, updating threshold value t
    while not_trivial is True:
        u0 = np.mean(np.ma.masked_less(data, t))
        u1 = np.mean(np.ma.masked_greater_equal(data, t))
        t_new = (u0 + u1) / 2
        if abs(t_new - t) < 0.01:
            not_trivial = False
        else:
            t = t_new
            # record t
            t_history.append(t)

    # update image pixels
    background = data < t
    foreground = data >= t
    data = np.where(background, 255, data)
    data = np.where(foreground, 0, data)

    return data, t, t_history


def task2(data: np.ndarray) -> (np.ndarray, int):
    '''
    Task 2 function will return the number of rice kernels and the labeled image
    '''
    # filter out the noise
    data = median_filter(data, 9)

    # create the record of the components
    labels = np.full(data.shape, 255, dtype='uint8')
    last_label = 0
    eq_label = {}

    # first pass
    row_range = range(1, data.shape[0] - 1)
    col_range = range(1, data.shape[1] - 1)
    for row, col in product(row_range, col_range):
        if data[row, col] != 255:
            # prefix n_ means neighborhood
            n_idx = [(row - 1, col - 1), (row - 1, col),
                     (row - 1, col + 1), (row, col - 1)]
            n_data = np.array([data[row, col] for row, col in n_idx])
            n_label = {labels[row, col]
                       for row, col in n_idx if labels[row, col] != 255}

            # create a new label if the neighborhoods are all 255
            if np.all(n_data != data[row, col]):
                eq_label[last_label] = {last_label}
                labels[row, col] = last_label
                last_label += 1
            else:
                labels[row, col] = min(n_label)

            # record the equivlence labels
            for r in n_label:
                eq_label[r] = eq_label[r].union(n_label)

    '''
    Compute records which are actually belongs to one component.

    For one label, we initialize last_group as the current label and
    we intialize the curr_group as the relating value of label.

    Each iteration will compare the last_group and the curr_group.
    If they are the same, it shows that we've found all labels which
    are acutally belongs to the same component.

    If not, take the symmetric difference of both groups as key of eq_record
    and then add their corresponding value to curr_group, then compare
    last_group and curr_group.
    '''
    equal_groups = []
    for k in eq_label:
        last_group = {k}
        curr_group = eq_label[k]
        diff = curr_group.symmetric_difference(last_group)

        while len(diff) > 0:
            last_group = curr_group
            for d in diff:
                curr_group = curr_group.union(eq_label[d])
            diff = curr_group.symmetric_difference(last_group)
        if curr_group not in equal_groups:
            equal_groups.append(curr_group)

    '''
    create key-value pairs where the keys are the labels
    and the values are the minimal label it should be
    '''
    eq_label = dict()
    for s in equal_groups:
        minor = min(s)
        for r in s:
            eq_label[r] = minor

    # second pass
    row_range = range(1, data.shape[0] - 1)
    col_range = range(1, data.shape[1] - 1)
    for row, col in product(row_range, col_range):
        if data[row, col] != 255:
            labels[row, col] = eq_label[labels[row, col]]

    return labels, len(equal_groups)


def task3(data: np.ndarray, min_area: int) -> (float, np.ndarray):
    '''
    Task 3 will process the result generated by task 2 and
    count the number of damaged kernels which got smaller area than
    'min_area'.

    It will return the percentage of damaged kernel and an image with labels
    replaced by 0
    '''

    # dictionary keeps the label and the area
    rice_labels = {k: 0 for k in (set(data.flat) - {255})}
    flatten_data = data.flatten()

    # count the label area
    for lb in rice_labels:
        rice_labels[lb] = np.ma.masked_not_equal(flatten_data, lb).count()

    damaged_kernels = set()
    whole_kernels = set()

    # seperate all labels into two groups - damaged, whole
    for lb in rice_labels:
        if rice_labels[lb] < min_area:
            damaged_kernels.add(lb)
        else:
            whole_kernels.add(lb)

    # remove damaged kernels from the image
    # and then replace labels with 0
    for lb in damaged_kernels:
        data = np.where(data == lb, 255, data)
    for lb in whole_kernels:
        data = np.where(data == lb, 0, data)

    return len(damaged_kernels) / len(rice_labels), data


my_parser = argparse.ArgumentParser()
my_parser.add_argument('-o', '--OP_folder', type=str,
                       help='Output folder name', default='OUTPUT')
my_parser.add_argument('-m', '--min_area', type=int, action='store', required=True,
                       help='Minimum pixel area to be occupied, to be considered a whole rice kernel')
my_parser.add_argument('-f', '--input_filename', type=str,
                       action='store', required=True, help='Filename of image ')
# Execute parse_args()
args = my_parser.parse_args()

# read arguments
output_folder = args.OP_folder
min_area = args.min_area

# full name of the image
fullname = args.input_filename.split('\\')[-1]
# file name without filename extension
filename = fullname.split('.')[0]


# create output folder if there were not
try:
    os.mkdir('./{}'.format(output_folder))
except FileExistsError:
    pass


# task 1
img = cv2.imread(args.input_filename, 0)
img, threshold, history = task1(img)
save_img(img, 'Threshold value = {}'.format(round(threshold, 1)),
         './{}/{}'.format(output_folder, filename + '_Task1.png'))

plt.bar([i for i in range(1, len(history) + 1)], history)
plt.title('Value of threshold in each iteration')
plt.savefig('./{}/{}'.format(output_folder,
                             filename + '_Task1_threshold_change.png'))


# task 2
img_median_filtered = median_filter(np.copy(img), ksize=9)
save_img(img_median_filtered, 'Median filter with ksize = 9', 
         './{}/{}'.format(output_folder, filename + '_Task2_Median_Filter.png'))
img, rice_kernel_cnt = task2(img)

title = ''
if rice_kernel_cnt == 1:
    title = '1 rice kernel found'
elif rice_kernel_cnt > 1:
    title = '{} rice kernels found'.format(rice_kernel_cnt)
else:
    title = 'No rice kernel found'

save_img(img, title,
         './{}/{}'.format(output_folder, filename + '_Task2.png'))


# task 3
damage_percent, img = task3(img, min_area)
save_img(img, '{}% of kernel damaged, min_area = {}'.format(round(damage_percent, 4) * 100, min_area),
         './{}/{}'.format(output_folder, filename + '_Task3.png'))
