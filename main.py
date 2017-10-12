#!/usr/bin/env python3

import codecs
import csv
from datetime import datetime, timedelta
import math

from numpy import array

from auc import auc
import ml

PATH = '/Users/louisliu/vagrant/github/kkv_data_game/'
SLOT_COUNT = 7 * 4
FEATURE_COUNT = 25
OUTPUT_FILENAME = PATH + 'out.ans'
OUTPUT_FILENAME2 = PATH + 'out2.ans'

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


def write_out_answer2(output_filename, predict, start_index):
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


def processing(stat, threshold):
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

    # assert(weight > 0 and weight <= 1)

    return weight * weight


def get_offset(dt):
    target = '2017-06-24 01:00:00'
    target_dt = datetime.strptime(target, '%Y-%m-%d %H:%M:%S')

    x = (target_dt - dt).days / 7
    # weight = (25 - x) / 25
    # print('%d %f' % (x, weight))

    # assert(weight > 0 and weight <= 1)

    return int(x)


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
            watch = int(session['sessionLength']) / 60
            dt2 = dt + timedelta(seconds=watch)
            timeslot = datetime_to_slot(dt)
            timeslot2 = datetime_to_slot(dt2)
            if watch == 0:
                continue
            if watch < 10:
                continue
            if watch >= 4 * 60:
                watch = 4 * 60
            if timeslot == timeslot2:
                stat[int(session['userId']) - ans_obj.start_index][timeslot] += (weight * watch)
            else:
                pass
                #stat[int(session['userId']) - ans_obj.start_index][timeslot] += (weight * watch / 3)
                #weight2 = get_weight(dt2)
                #stat[int(session['userId']) - ans_obj.start_index][timeslot2] += (weight2 * watch / 3)

    processing(stat, cal_threshold(stat))
    write_out_answer(OUTPUT_FILENAME, stat, ans_obj.start_index)

    if is_train:
        ans_obj2 = AnswerReader(OUTPUT_FILENAME)
        ans_obj.compare(ans_obj2)


def get_x_data(is_train, index, ans_obj):
    if is_train:
        count = len(ans_obj)
        entry_count = sum(obj.entry_count for obj in ans_obj)
        start_index = ans_obj[0].start_index
    else:
        count = 40
        entry_count = ans_obj.entry_count
        start_index = ans_obj.start_index

    x_data = [[0 for i in range(FEATURE_COUNT * SLOT_COUNT)] for y in range(entry_count)]

    for i in range(index, index + count):
        if is_train:
            input_filename = '%s-%05d' % (PATH + 'log/train/log', i)
        else:
            input_filename = '%s-%05d' % (PATH + 'log/test/log', i)

        print(input_filename)

        reader = csv.DictReader(codecs.open(input_filename, 'r', encoding='utf-8'))
        for session in reader:
            dt = datetime.strptime(session['sessionStartTime'], '%Y-%m-%d %H:%M:%S')
            offset = get_offset(dt)
            watch = int(session['sessionLength'])
            timeslot = datetime_to_slot(dt)
            if watch < 300:
                continue

            if x_data[int(session['userId']) - start_index][offset + timeslot * FEATURE_COUNT] < 1:
                x_data[int(session['userId']) - start_index][offset + timeslot * FEATURE_COUNT] += 0.05

    return x_data


def machine_learning(is_train, train_start_index=0, train_count=60):
    ans_obj_train = [None for i in range(train_count)]
    for i in range(train_start_index, train_start_index + train_count):
        ans_train = PATH + 'ans/ans-%05d' % i
        ans_obj_train[i] = AnswerReader(ans_train)

    if is_train is False:
        ans_obj_test = AnswerReader(ans_test)
        test_start_index = 60

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

    num_iterations = 4000
    learning_rate = 0.005 * 2
    print_cost = True
    print('num_iterations = %d, learning_rate = %f' % (num_iterations, learning_rate))
    predict_train = [[] for i in range(SLOT_COUNT)]
    # if is_train is False:
    predict_test = [[] for i in range(SLOT_COUNT)]
    for model_no in range(1, SLOT_COUNT + 1):
        print(ans_header[model_no])
        train_set_y_orig = [[int(ans_obj_train[j].data[i][ans_header[model_no]]) for j in range(len(ans_obj_train)) for i in range(ans_obj_train[j].entry_count)]]
        train_set_y = array(train_set_y_orig)

        d = ml.model(train_set_x, train_set_y, test_set_x, None, num_iterations, learning_rate, print_cost)

        predict_train[model_no - 1] = d['Y_prediction2_train'][0]
        predict_test[model_no - 1] = d['Y_prediction2_test'][0]

        # score = ans_obj_train.compare_single(model_no, predict_train[model_no - 1])
        # print('[%s] score: %f' % (ans_header[model_no], score))

    if is_train is False:
        write_out_answer2(OUTPUT_FILENAME2, predict_test, ans_obj_test.start_index)
    else:
        write_out_answer2(OUTPUT_FILENAME2, predict_test, ans_obj_train[0].start_index)
        ans_obj_predict = AnswerReader(OUTPUT_FILENAME2)
        ans_obj_train[0].compare(ans_obj_predict)


def main():
    is_human_learning = False
    is_train = False

    if is_human_learning:
        if is_train:
            for i in range(60):
                human_learning(is_train, i)
        else:
            human_learning(is_train)
    else:
        machine_learning(is_train)


if __name__ == '__main__':
    main()
