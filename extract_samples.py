import re
import os
from lxml import etree


def main():
    output = open('samples/samples.txt', 'w', encoding='utf-8')
    types = {'Life': '1', 'Movement': '2', 'Transaction': '3', 'Business': '4',
             'Conflict': '5', 'Contact': '6', 'Personnel': '7', 'Justice': '8'}

    paths = ['/Users/abtion/workspace/dataset/ace/Chinese/bn/adj/',
             '/Users/abtion/workspace/dataset/ace/Chinese/nw/adj/',
             '/Users/abtion/workspace/dataset/ace/Chinese/wl/adj/']
    for path in paths:
        files = os.listdir(path)
        for file in files:
            if re.search(r'\.apf\.xml', file):
                full_txt_file = file[:-7] + 'sgm'
                full_txt = _get_text(path, full_txt_file)

                root = etree.parse(path + file).getroot()
                for doc in root.findall('document'):
                    for event in doc.findall('event'):
                        type = event.get('TYPE')
                        polarity = event.get('POLARITY')
                        for event_mention in event.findall('event_mention'):
                            for ldc in event_mention.findall('ldc_scope'):
                                seq = ldc.find('charseq').text
                                full_txt.replace(seq, '')
                                seq = seq.replace('\n', '')
                                seq = seq.replace(' ', '')
                                temp_text = (types[type] + '\t' + seq + '。\n')
                                output.write(temp_text)

                full_txt = full_txt.replace('\n', '')
                full_txt = full_txt.replace(' ', '')
                full_txt_seqs = re.split(r'。', full_txt)
                for seq in full_txt_seqs:
                    seq = re.sub(r'^[：，“”"「」\s]', '', seq)
                    if seq != '':
                        temp_text = '0' + '\t' + seq + '。\n'
                        output.write(temp_text)


def _get_text(path, file):
    body = etree.parse(path + file).getroot().find('BODY')
    if path[-7:-5] == 'bn':
        return body.xpath('//text()')[-6]
    if path[-7:-5] == 'nw':
        return body.xpath('//text()')[-3]
    if path[-7:-5] == 'wl':
        return body.xpath('//text()')[-4]


if __name__ == '__main__':
    main()
