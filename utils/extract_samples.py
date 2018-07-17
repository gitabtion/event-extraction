import re
import os
import random
from lxml import etree


def main():
    closed_set = set()
    train_set = open('./data/train_set.txt', 'w', encoding='utf-8')
    ver_set = open('./data/ver_set.txt', 'w', encoding='utf-8')
    test_set = open('./data/test_set.txt', 'w', encoding='utf-8')
    types = {'Life': '1', 'Movement': '2', 'Transaction': '3', 'Business': '4',
             'Conflict': '5', 'Contact': '6', 'Personnel': '7', 'Justice': '8'}

    paths = ['/Users/abtion/workspace/dataset/ace/Chinese/bn/adj/',
             '/Users/abtion/workspace/dataset/ace/Chinese/nw/adj/',
             '/Users/abtion/workspace/dataset/ace/Chinese/wl/adj/']
    tests, verifies = _get_test_and_verify_list(paths)

    for path in paths:
        files = os.listdir(path)
        random.shuffle(files)
        for file in files:
            if re.search(r'\.apf\.xml', file):
                full_txt_file = file[:-7] + 'sgm'
                full_txt = _get_text(path, full_txt_file)
                full_txt = re.sub(r'[\s]+', '', full_txt)

                root = etree.parse(path + file).getroot()
                for doc in root.findall('document'):
                    for event in doc.findall('event'):
                        type = event.get('TYPE')
                        for event_mention in event.findall('event_mention'):
                            for ldc in event_mention.findall('ldc_scope'):
                                seq = ldc.find('charseq').text
                                seq = re.sub(r'[\s]+', '', seq)
                                full_txt = full_txt.replace(seq, '')
                                temp_text = (types[type] + '\t' + seq + '。\n')
                                _output(file, verifies, tests, train_set,
                                        ver_set, test_set, temp_text, closed_set)

                full_txt_seqs = re.split(r'。', full_txt)
                for seq in full_txt_seqs:
                    seq = re.sub(r'^\W+', '', seq)
                    seq = re.sub(r'\W+$', '', seq)
                    if seq != '':
                        temp_text = '0' + '\t' + seq + '。\n'
                        _output(file, verifies, tests, train_set,
                                ver_set, test_set, temp_text, closed_set)

    train_set.close()
    ver_set.close()
    test_set.close()


def _get_test_and_verify_list(paths):
    """
    获取测试集和验证集的文件名列表
    共633个文件，其中：
    测试集： 66个文件,从test_set_name.txt中取
    训练集： 567个文件
    验证集： 在训练集中随机取33个文件
    :param paths: 路径列表
    :return: 测试集和验证集的文件名列表
    """
    files = []
    verifies = []
    for path in paths:
        files.extend(os.listdir(path))
    random.shuffle(files)

    test_set_names = open('./data/test_set_name.txt', encoding='utf-8').read()
    tests = test_set_names.split('\n')
    for file in files:
        if re.search(r'\.apf\.xml', file) and len(verifies) < 33 and file[:-8] not in tests:
            verifies.append(file)
        if len(verifies) >= 33:
            break
    return tests, verifies


def _output(file, verifies, tests, train_set, ver_set, test_set, seq, closed_set):
    """
    将单行文本输出至对于文件
    :param file: 当前读取的文件名
    :param verifies: 验证集文件名列表
    :param tests: 测试集文件名列表
    :param train_set: 训练集文件
    :param ver_set: 验证集文件
    :param test_set: 测试集文件
    :param seq: 单行样本
    :param closed_set: 已输出的样本集合
    :return: None
    """
    if seq not in closed_set:
        if seq[1:] not in closed_set or seq[0] != '0':

            if file in verifies:
                ver_set.write(seq)
            elif file[:-8] in tests:
                test_set.write(seq)
            else:
                train_set.write(seq)
            closed_set.add(seq)
            closed_set.add(seq[1:])


def _get_text(path, file):
    """
    解析全文XML，获取所需文本
    :param path: 文件路径
    :param file: 文件名
    :return: string类型的文本
    """
    body = etree.parse(path + file).getroot().find('BODY')
    temp_str = ''
    if path[-7:-5] == 'bn':
        return temp_str.join(body.xpath('//TURN/text()'))
    if path[-7:-5] == 'nw':
        return temp_str.join(body.xpath('//TEXT/text()'))
    if path[-7:-5] == 'wl':
        return body.xpath('//POST[1]/text()')[2]


if __name__ == '__main__':
    main()
