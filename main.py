#!/usr/bin/env python3

import codecs
import csv
from datetime import datetime, timedelta
import math

import numpy as np
from numpy import array

from auc import auc
import ml

PATH = '/Users/louisliu/vagrant/github/kkv_data_game/'
SLOT_COUNT = 7 * 4
FEATURE_COUNT = 33
OUTPUT_FILENAME = PATH + 'out.ans'
OUTPUT_FILENAME2 = PATH + 'out2.ans'

ans_train = PATH + 'public/data-000.csv'
ans_test = PATH + 'sample.csv'

ans_header = [
    'user_id',
    'time_slot_0', 'time_slot_1', 'time_slot_2', 'time_slot_3',
    'time_slot_4', 'time_slot_5', 'time_slot_6', 'time_slot_7',
    'time_slot_8', 'time_slot_9', 'time_slot_10', 'time_slot_11',
    'time_slot_12', 'time_slot_13', 'time_slot_14', 'time_slot_15',
    'time_slot_16', 'time_slot_17', 'time_slot_18', 'time_slot_19',
    'time_slot_20', 'time_slot_21', 'time_slot_22', 'time_slot_23',
    'time_slot_24', 'time_slot_25', 'time_slot_26', 'time_slot_27'
]


class AnswerReader:
    def __init__(self, ans_file):
        print(ans_file)
        reader = csv.DictReader(codecs.open(ans_file, 'r', encoding='utf-8'))
        self.data = list(reader)

    @property
    def start_index(self):
        return int(self.data[0]['user_id'])

    @property
    def entry_count(self):
        return int(self.data[-1]['user_id']) - int(self.data[0]['user_id']) + 1

    def compare(self, ans2):
        assert(self.entry_count == ans2.entry_count)
        assert(self.start_index == ans2.start_index)

        guess = []
        truth = []
        guess_l = [[] for i in range(SLOT_COUNT)]
        truth_l = [[] for i in range(SLOT_COUNT)]

        for row_index in range(0, self.entry_count):
            for column_index in range(1, SLOT_COUNT + 1):
                guess.append(float(ans2.data[row_index][ans_header[column_index]]))
                truth.append(float(self.data[row_index][ans_header[column_index]]))
                guess_l[column_index - 1].append(float(ans2.data[row_index][ans_header[column_index]]))
                truth_l[column_index - 1].append(float(self.data[row_index][ans_header[column_index]]))

        print('score: %f' % auc(guess, truth))

        for i in range(SLOT_COUNT):
            print('\tscore[%s] is %f' % (ans_header[i + 1], auc(guess_l[i], truth_l[i])))

    def compare_single(self, column_index, column):
        guess = []
        truth = []

        for row_index in range(0, self.entry_count):
            guess.append(column[row_index])
            truth.append(int(self.data[row_index][ans_header[column_index]]))

        assert(len(guess) == len(truth))
        # print(truth)

        return auc(guess, truth)


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


def write_out_answer(output_filename, predict, start_index):
    assert(len(predict) == SLOT_COUNT)
    f = open(output_filename, 'w')

    for item in ans_header[0:-1]:
        f.write('%s,' % item)
    f.write('%s\n' % ans_header[-1])

    entry_count = len(predict[0])
    for i in range(entry_count):
        user_id = start_index + i
        f.write('%d,' % user_id)
        for j in range(SLOT_COUNT - 1):
            f.write('%f,' % predict[j][i])
        f.write('%f\n' % predict[-1][i])

    f.close()


def column(matrix, i):
    return [row[i] for row in matrix]


def cal_threshold(stat):
    # print(column(stat, 0))

    threshold = [29 for i in range(SLOT_COUNT)]

    return threshold


def get_offset(dt):
    target = '2017-08-14 01:00:00'
    target_dt = datetime.strptime(target, '%Y-%m-%d %H:%M:%S')

    x = (target_dt - dt).days / 7
    # weight = (25 - x) / 25
    # print('%d %f' % (x, weight))

    # assert(weight > 0 and weight <= 1)

    if int(x) >= FEATURE_COUNT:
        print(x)
        print(dt)

    assert(int(x) < FEATURE_COUNT)

    return int(x)


def get_x_data(is_train, index, ans_obj):
    if is_train:
        count = len(ans_obj)
        entry_count = sum(obj.entry_count for obj in ans_obj)
        start_index = ans_obj[0].start_index
    else:
        count = 30
        entry_count = ans_obj.entry_count
        start_index = ans_obj.start_index

    x_data = [[0 for i in range(FEATURE_COUNT * SLOT_COUNT)] for y in range(entry_count)]

    for i in range(index, index + count):
        if is_train:
            input_filename = '%s-%03d.csv' % (PATH + 'public/data', i)
        else:
            input_filename = '%s-%03d.csv' % (PATH + 'public/data', i)

        print(input_filename)

        reader = csv.DictReader(codecs.open(input_filename, 'r', encoding='utf-8'))
        for session in reader:
            dt = datetime.strptime(session['event_time'], '%Y-%m-%d %H:%M:%S.%f')
            offset = get_offset(dt)
            if offset < 0:
                continue
            watch = int(session['played_duration'])
            timeslot = datetime_to_slot(dt)
            if watch < 300:
                continue

            if x_data[int(session['user_id']) - start_index][offset + timeslot * FEATURE_COUNT] < 1:
                x_data[int(session['user_id']) - start_index][offset + timeslot * FEATURE_COUNT] += 0.05

    return x_data


def gen_x_data(is_train, train_start_index=1, train_count=36):
    ans_obj_train = [None for i in range(train_count)]
    for i in range(train_start_index, train_start_index + train_count):
        ans_train = PATH + 'public/label-%03d.csv' % i
        ans_obj_train[i - 1] = AnswerReader(ans_train)

    if is_train is False:
        ans_obj_test = AnswerReader(ans_test)
        test_start_index = 46

    m_train = sum(ans_obj.entry_count for ans_obj in ans_obj_train)
    if is_train is False:
        m_test = ans_obj_test.entry_count

    if is_train is False:
        # prepare test set
        test_set_x_orig = get_x_data(False, test_start_index, ans_obj_test)
        test_set_x = array(test_set_x_orig).T
        np.savetxt('x_data_test.txt', test_set_x)
    else:
        # prepare training set
        train_set_x_orig = get_x_data(True, train_start_index, ans_obj_train)
        train_set_x = array(train_set_x_orig).T
        np.savetxt('x_data_train_%d_%d.txt' % (train_start_index, train_count), train_set_x)


def machine_learning(is_train, train_start_index=1, train_count=36):
    ans_obj_train = [None for i in range(train_count)]
    for i in range(train_start_index, train_start_index + train_count):
        ans_train = PATH + 'public/label-%03d.csv' % i
        ans_obj_train[i - 1] = AnswerReader(ans_train)

    if is_train is False:
        ans_obj_test = AnswerReader(ans_test)
        test_start_index = 46

    m_train = sum(ans_obj.entry_count for ans_obj in ans_obj_train)
    if is_train is False:
        m_test = ans_obj_test.entry_count

    # prepare training set
    train_set_x_orig = get_x_data(True, train_start_index, ans_obj_train)
    train_set_x = array(train_set_x_orig).T

    # prepare test set
    if is_train is False:
        test_set_x_orig = get_x_data(False, test_start_index, ans_obj_test)
        test_set_x = array(test_set_x_orig).T
    else:
        test_set_x_orig = get_x_data(True, train_start_index, [ans_obj_train[0]])
        test_set_x = array(test_set_x_orig).T

    num_iterations = 4000 * 40
    learning_rate = 0.005 * 4
    print_cost = True
    print('num_iterations = %d, learning_rate = %f' % (num_iterations, learning_rate))
    predict_train = [[] for i in range(SLOT_COUNT)]
    predict_test = [[] for i in range(SLOT_COUNT)]
    iw, ib = None, None
    for model_no in range(1, SLOT_COUNT + 1):
        print(ans_header[model_no])
        train_set_y_orig = [[int(ans_obj_train[j].data[i][ans_header[model_no]]) for j in range(len(ans_obj_train)) for i in range(ans_obj_train[j].entry_count)]]
        train_set_y = array(train_set_y_orig)

        d = ml.model(train_set_x, train_set_y, test_set_x, None, num_iterations, learning_rate, print_cost, iw, ib)

        iw = d['w']
        ib = d['b']

        predict_train[model_no - 1] = d['Y_prediction2_train'][0]
        predict_test[model_no - 1] = d['Y_prediction2_test'][0]

        # score = ans_obj_train.compare_single(model_no, predict_train[model_no - 1])
        # print('[%s] score: %f' % (ans_header[model_no], score))

    if is_train is False:
        write_out_answer(OUTPUT_FILENAME2, predict_test, ans_obj_test.start_index)
    else:
        write_out_answer(OUTPUT_FILENAME2, predict_test, ans_obj_train[0].start_index)
        ans_obj_predict = AnswerReader(OUTPUT_FILENAME2)
        ans_obj_train[0].compare(ans_obj_predict)


def main():
    is_train = False
    machine_learning(is_train)
    # gen_x_data(is_train)

    # x_data = [[1, 2, 3], [2, 3, 4]]
    # np.savetxt('test.txt', x_data)
    # new_data = np.loadtxt('test.txt')
    # print(new_data)


if __name__ == '__main__':
    main()
