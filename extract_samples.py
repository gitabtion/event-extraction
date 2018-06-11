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
                full_txt = ''
                full_txt_file = file[:-7]+'sgm'
                # f = open(path+'/'+full_txt_file,encoding='utf-8').read
                # etree.fromstring(f).getroot()
                full_txt_root = etree.parse(path+'/'+full_txt_file).getroot()
                for body in full_txt_root.findall('BODY'):
                    for text in body.findall('TEXT'):
                        for turn in text.findall('TURN'):
                            full_txt = turn.text

                root = etree.parse(path+'/'+file).getroot()
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
                                temp_text = seq+'\t' + \
                                    types[type]+'\n'
                                output.write(temp_text)

                full_txt = full_txt.replace('\n', '')
                full_txt = full_txt.replace(' ', '')
                full_txt_seqs = re.split('[^：，\w]', full_txt)
                for seq in full_txt_seqs:
                    seq = seq.replace('\n', '')
                    seq = seq.replace(' ', '')
                    if seq != '':
                        temp_text = seq + '\t' + '0' + '\n'
                        output.write(temp_text)


if __name__ == '__main__':
    main()
