#!/usr/bin/env python3

import codecs
import csv
from datetime import datetime

PATH = '/Users/louisliu/vagrant/github/kkv_data_game/'

label_train_cut = PATH + 'log/train/log-00000_cut'
label_train = PATH + 'log/train/log-00000'


def datetime_to_slot(dt):
    weekday_map = [2, 3, 4, 5, 6, 0, 1]

    dt_align = dt.replace(hour=0, minute=0, second=0)
    s = int(dt.timestamp() - dt_align.timestamp())

    # print(dt.weekday())
    offset = weekday_map[dt.weekday()] * 4

    if s >= 1 * 60 * 60 and s < 9 * 60 * 60:
        slot = 0
    elif s >= 9 * 60 * 60 and s < 17 * 60 * 60:
        slot = 1
    elif s >= 17 * 60 * 60 and s < 21 * 60 * 60:
        slot = 2
    else:
        slot = 3

    return offset + slot


def main():

    csvreader = csv.reader(codecs.open(label_train_cut, 'r', encoding='utf-8'))
    header = next(csvreader)
    orig_label = list(csvreader)

    print(header)

    for item in orig_label:
        dt = datetime.strptime(item[7], '%Y-%m-%d %H:%M:%S')
        print('%s %s %s %d' % (item[3], item[7], item[8], datetime_to_slot(dt)))

    # print(orig_label[0])


if __name__ == '__main__':
    main()
