import numpy as np
import pandas as pd
import os

untransform_txt="generate_dev.txt"


def load_lines_from_txt(directory):# load lines from text
    file = open(directory, 'r')
    lines = []
    while 1:
        line = file.readline()
        if not line:
            break
        lines.append(line)
    return lines
def load_labels(lines):
    original_labels=[]
    for line in lines:
        original_label=[]
        if line=="/n":  #the final line
            continue
        split_line=line.split()
        for word in split_line:
            word_part=word.split("/")
            original_label.append(word_part[-1])
        original_labels.append(original_label)
    return original_labels
def transform_into_all_s(original_labels):
    for line_index in range(len(original_labels)):
        for label_index in range(len(original_labels[line_index])):
            if original_labels[line_index][label_index]=="rel" or original_labels[line_index][label_index]=="O":
                continue
            else:
                original_labels[line_index][label_index]="S"+original_labels[line_index][label_index][1:]
    return original_labels
def change_into_BIE(all_s_labels):
    for line_index in range(len(all_s_labels)):
        label_index=0
        while(label_index<len(all_s_labels[line_index])):
            if original_labels[line_index][label_index]=="rel" or original_labels[line_index][label_index]=="O":
                label_index +=1
                continue
            else:
                word_length=1
                word=all_s_labels[line_index][label_index][2:]
                while(all_s_labels[line_index][label_index+word_length][2:]==word):
                    word_length+=1

                if word_length ==1:
                    label_index +=1
                    continue
                if word_length>1:
                    all_s_labels[line_index][label_index]="B"+all_s_labels[line_index][label_index][1:]
                    all_s_labels[line_index][label_index+word_length-1]="E"+all_s_labels[line_index][label_index+word_length-1][1:]
                    for move_pos in range(1,word_length-1):
                        all_s_labels[line_index][label_index+move_pos]="I"+all_s_labels[line_index][label_index+move_pos][1:]

                    label_index +=word_length
                    continue
    return all_s_labels


if __name__=="__main__":
    lines=load_lines_from_txt(untransform_txt)
    original_labels=load_labels(lines)
    all_s_labels=transform_into_all_s(original_labels)
    BIE_labels=change_into_BIE(all_s_labels)

    f_pred = open("transformed_"+untransform_txt, 'w')
    for line_index in range(len(lines)):
        BIE_sentence=[]
        if lines[line_index] == "/n":  # the final line
            continue
        split_line = lines[line_index].split()
        for word_index in range(len(split_line)):
            word_part = split_line[word_index].split("/")
            BIE_sentence.append("/".join([word_part[0],word_part[1],BIE_labels[line_index][word_index]]))
        BIE_sentence=" ".join(BIE_sentence)
        f_pred.write(BIE_sentence)
        f_pred.write('\n')
    f_pred.write('\n')
    f_pred.close()
    os.system("python "+"calc_f1.py "+"transformed_generate_dev.txt "+"cpbdev.txt")
