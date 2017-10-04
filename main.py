#!/usr/bin/env python3

import codecs
import csv
from datetime import datetime, timedelta

from auc import auc

PATH = '/Users/louisliu/vagrant/github/kkv_data_game/'
SLOT_COUNT = 7 * 4
OUTPUT_FILENAME = PATH + 'out.ans'

label_train_cut = PATH + 'log/train/log-00000_cut'
label_train = PATH + 'log/train/log-00000'


ans_train = PATH + 'ans/ans-00000'
ans_test = PATH + 'init_submit.csv'

ans_header = [
    'userId',
    '624ans1', '624ans2', '624ans3', '624ans4',
    '625ans1', '625ans2', '625ans3', '625ans4',
    '626ans1', '626ans2', '626ans3', '626ans4',
    '627ans1', '627ans2', '627ans3', '627ans4',
    '628ans1', '628ans2', '628ans3', '628ans4',
    '629ans1', '629ans2', '629ans3', '629ans4',
    '630ans1', '630ans2', '630ans3', '630ans4'
]


class AnswerReader:
    def __init__(self, ans_file):
        print(ans_file)
        reader = csv.DictReader(codecs.open(ans_file, 'r', encoding='utf-8'))
        self.data = list(reader)

    @property
    def start_index(self):
        return int(self.data[0]['userId'])

    @property
    def entry_count(self):
        return int(self.data[-1]['userId']) - int(self.data[0]['userId']) + 1

    def compare(self, ans2):
        assert(self.entry_count == ans2.entry_count)
        assert(self.start_index == ans2.start_index)

        guess = []
        truth = []
        guess_l = [[] for i in range(SLOT_COUNT)]
        truth_l = [[] for i in range(SLOT_COUNT)]

        for row_index in range(0, self.entry_count):
            for column_index in range(1, SLOT_COUNT + 1):
                guess.append(int(ans2.data[row_index][ans_header[column_index]]))
                truth.append(int(self.data[row_index][ans_header[column_index]]))
                guess_l[column_index - 1].append(int(ans2.data[row_index][ans_header[column_index]]))
                truth_l[column_index - 1].append(int(self.data[row_index][ans_header[column_index]]))

        print('score: %f' % auc(guess, truth))

        for i in range(SLOT_COUNT):
            print('\tscore[%s] is %f' % (ans_header[i + 1], auc(guess_l[i], truth_l[i])))


def datetime_to_slot(dt):
    weekday_map = [2, 3, 4, 5, 6, 0, 1]

    dt_align = dt.replace(hour=0, minute=0, second=0)
    s = int(dt.timestamp() - dt_align.timestamp())

    # print(dt.weekday())
    offset = weekday_map[dt.weekday()] * 4

    if s < 1 * 60 * 60:
        offset -= 4
        slot = 3
    elif s < 9 * 60 * 60:
        slot = 0
    elif s < 17 * 60 * 60:
        slot = 1
    elif s < 21 * 60 * 60:
        slot = 2
    else:
        slot = 3

    ret = offset + slot

    if ret < 0:
        assert(ret == -1)
        ret += 28

    return ret


def write_out_answer(output_filename, stat, start_index):
    f = open(output_filename, 'w')

    for item in ans_header[0:-1]:
        f.write('%s,' % item)
    f.write('%s\n' % ans_header[-1])

    counter = 0
    for row in stat:
        user_id = start_index + counter
        f.write('%d,' % user_id)
        for item in row[0:-1]:
            f.write('%d,' % item)
        f.write('%d\n' % row[-1])
        counter += 1

    f.close()


def processing(stat):
    threshold = [0 for i in range(0, SLOT_COUNT)]

    for i in range(0, len(threshold)):
        if (i % 4) == 0:
            threshold[i] = 60 * 60
        elif (i % 4) == 1:
            threshold[i] = 30 * 60
        elif (i % 4) == 2:
            threshold[i] = 32 * 60
        elif (i % 4) == 3:
            threshold[i] = 30 * 60

    threshold[0] = 70 * 60
    threshold[1] = 30 * 60
    threshold[2] = 26 * 60
    threshold[3] = 30 * 60
    threshold[4] = 50 * 60
    threshold[16] = 40 * 60
    threshold[22] = 60 * 60
    threshold[26] = 60 * 60
    threshold[27] = 60 * 60

    # print(threshold)

    for i in range(0, len(stat)):
        for j in range(0, len(stat[i])):
            if stat[i][j] > threshold[j]:
                stat[i][j] = 1
            else:
                stat[i][j] = 0


def get_weight(dt):
    target = '2017-06-24 01:00:00'
    target_dt = datetime.strptime(target, '%Y-%m-%d %H:%M:%S')

    x = (target_dt - dt).days / 7
    weight = (25 - x) / 25
    # print('%d %f' % (x, weight))

    # assert(weight > 0)

    return weight * weight


def human_learning(is_train, start_index=0):

    if is_train:
        ans_train = PATH + 'ans/ans-%05d' % start_index
        ans_obj = AnswerReader(ans_train)
        end_index = start_index + 1
    else:
        ans_obj = AnswerReader(ans_test)
        start_index = 60
        end_index = 100

    stat = [[0 for i in range(SLOT_COUNT)] for j in range(ans_obj.entry_count)]

    for i in range(start_index, end_index):
        if is_train:
            input_filename = '%s-%05d' % (PATH + 'log/train/log', i)
        else:
            input_filename = '%s-%05d' % (PATH + 'log/test/log', i)
        print(input_filename)
        reader = csv.DictReader(codecs.open(input_filename, 'r', encoding='utf-8'))
        for session in reader:
            dt = datetime.strptime(session['sessionStartTime'], '%Y-%m-%d %H:%M:%S')
            weight = get_weight(dt)
            watch = int(session['sessionLength'])
            dt2 = dt + timedelta(seconds=watch)
            timeslot = datetime_to_slot(dt)
            timeslot2 = datetime_to_slot(dt2)
            if watch > 4 * 60 * 60:
                continue
            if timeslot == timeslot2:
                stat[int(session['userId']) - ans_obj.start_index][timeslot] += (weight * watch)
            else:
                stat[int(session['userId']) - ans_obj.start_index][timeslot] += (weight * watch / 2)
                weight2 = get_weight(dt2)
                stat[int(session['userId']) - ans_obj.start_index][timeslot2] += (weight2 * watch / 2)

    processing(stat)
    write_out_answer(OUTPUT_FILENAME, stat, ans_obj.start_index)

    if is_train:
        ans_obj2 = AnswerReader(OUTPUT_FILENAME)
        ans_obj.compare(ans_obj2)


def main():
    is_train = False

    if is_train:
        for i in range(60):
            human_learning(is_train, i)
    else:
        human_learning(is_train)


if __name__ == '__main__':
    main()
