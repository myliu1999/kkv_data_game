#!/usr/bin/env python3

import codecs
import csv
from datetime import datetime

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

        hit_count = 0
        for row_index in range(0, self.entry_count):
            for column_index in range(1, SLOT_COUNT + 1):
                if self.data[row_index][ans_header[column_index]] == ans2.data[row_index][ans_header[column_index]]:
                    hit_count += 1

        hit_rate = hit_count / (SLOT_COUNT * self.entry_count) * 100
        print('hit_rate is %f' % hit_rate)


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
    threshold = 15
    for i in range(0, len(stat)):
        stat[i] = [1 if val >= (60 * threshold) else 0 for val in stat[i]]


def get_weight(dt):
    one_week_before = '2017-06-17 00:00:00'
    two_week_before = '2017-06-10 00:00:00'
    three_week_before = '2017-06-03 00:00:00'
    four_week_before = '2017-05-27 00:00:00'
    target_1st = datetime.strptime(one_week_before, '%Y-%m-%d %H:%M:%S')
    target_2nd = datetime.strptime(two_week_before, '%Y-%m-%d %H:%M:%S')
    target_3rd = datetime.strptime(three_week_before, '%Y-%m-%d %H:%M:%S')
    target_4th = datetime.strptime(four_week_before, '%Y-%m-%d %H:%M:%S')
    if (target_1st < dt):
        weight = 0.66
    elif (target_2nd < dt):
        weight = 0.33
    elif (target_3rd < dt):
        weight = 0
    else:
        weight = 0
    # print('%s %s %s %d' % (target_dt, dt, target_dt - dt, target_dt < dt))

    return weight


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
            # print('%s %s %s %d' % (session['userId'], session['sessionStartTime'], session['sessionLength'], datetime_to_slot(dt)))
            # print('%d %d %d' % (int(session['userId']), datetime_to_slot(dt), int(session['sessionLength'])))
            weight = get_weight(dt)
            if weight > 0:
                stat[int(session['userId']) - ans_obj.start_index][datetime_to_slot(dt)] += (weight * int(session['sessionLength']))

    processing(stat)
    write_out_answer(OUTPUT_FILENAME, stat, ans_obj.start_index)

    ans_obj2 = AnswerReader(OUTPUT_FILENAME)
    ans_obj.compare(ans_obj2)


def main():
    is_train = False
    human_learning(is_train)
    # for i in range(60):
    #    human_learning(is_train, i)


'''
    ans_obj = AnswerReader(ans_train)

    stat = [[0 for i in range(SLOT_COUNT)] for j in range(ans_obj.entry_count)]

    for i in range(0, 1):
        input_filename = '%s-%05d' % (PATH + 'log/train/log', i)
        print(input_filename)
        reader = csv.DictReader(codecs.open(input_filename, 'r', encoding='utf-8'))
        for session in reader:
            dt = datetime.strptime(session['sessionStartTime'], '%Y-%m-%d %H:%M:%S')
            # print('%s %s %s %d' % (session['userId'], session['sessionStartTime'], session['sessionLength'], datetime_to_slot(dt)))
            # print('%d %d %d' % (int(session['userId']), datetime_to_slot(dt), int(session['sessionLength'])))
            if is_valid_dt(dt):
                stat[int(session['userId']) - ans_obj.start_index][datetime_to_slot(dt)] += int(session['sessionLength'])

    processing(stat)
    write_out_answer(OUTPUT_FILENAME, stat, ans_obj.start_index)

    ans_obj2 = AnswerReader(OUTPUT_FILENAME)
    ans_obj.compare(ans_obj2)
'''


if __name__ == '__main__':
    main()
