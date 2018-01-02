#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import sys
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional,Dropout
from keras.models import Sequential
from keras.models import Model
#import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from gensim.models import Word2Vec
import gensim
import pickle
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
#global variables
train_txt_directory="cpbtrain.txt"
dev_txt_directory = "cpbdev.txt"
test_txt_directory="cpbtest.txt"
word2vec_directory = "word2vec/news_12g_baidubaike_20g_novel_90g_embedding_64.bin"
window_size=2   # the window of rel_verb
maxlen = 45     #maximum length for one line.
feature_num=3  #unchanged

word_size = 64 # the dimension of word vector
num_1=200    # the number of units in first LSTM layer
num_2=300    # the number of units in second LSTM layer
num_3=200   # the number of units in third LSTM layer
max_epochs=60

batch_size = 1000 #feed into LSTM

def load_lines_from_txt(directory):# load lines from text
    file = open(directory, 'r')
    lines = []
    while 1:
        line = file.readline()
        if not line:
            break
        lines.append(line)
    return lines

def load_all_labels(lines):      # construct the set of all labels
    labels = []
    for line in lines:
        split_line = line.split()
        for word in split_line:
            label_candidate = word.split("/")[-1]
            if label_candidate not in labels:
                labels.append(label_candidate)
    for label_index in range(len(labels)):
        if labels[label_index]=="rel" or labels[label_index]=="O":
            continue
        else:
            labels[label_index]="S"+labels[label_index][1:]
    new_labels=[]
    for label in labels:
        if label not in new_labels:
            new_labels.append(label)
    print "num of labels:",len(new_labels)
    return new_labels

def generate_dataset_DF(lines,test=False):  # generate a dataframe coutaining each line
    data = []  # 生成训练样本
    label = []
    verbs = []
    distances = []


    for line in lines:
        flag,x, y, z, distance = get_single_data(line,test=test)
        if flag:
            data.append(x)
            label.append(y)
            verbs.append(z)
            distances.append(distance)

    d_temp = pd.DataFrame(index=range(len(data)))
    d_temp['data'] = data
    d_temp['label'] = label
    d_temp['verbs'] = verbs
    d_temp['distance'] = distances

    d_temp = d_temp[d_temp['data'].apply(len) <= maxlen]
    d_temp.index = range(len(d_temp))

    return d_temp

def get_single_data(line,test=False):          # generate a single line data
    split_line = line.split()
    collect_word = []

    collect_label = []
    collect_distance = []
    flag=1  #1 means it is ok
    rel_index=0
    for word in split_line:
        word_part = word.split("/")
        if len(word_part)==3 and word_part[-1]=="rel":
            rel_index=split_line.index(word)
            break
    if len(split_line)>maxlen:
        begin_pos=max(0,rel_index-maxlen/2)
        end_pos=min(begin_pos+maxlen,len(split_line)-1)
        rel_index=rel_index-begin_pos
        split_line=split_line[begin_pos:end_pos]

    if test==True:
        for word in split_line:
            word_part = word.split("/")
            word_original, word_label = word_part[0], "O"
            collect_word.append(word_original)
            collect_label.append(word_label)
        collect_label[rel_index]="rel"
    else:
        for word in split_line:
            word_part = word.split("/")
            word_original, word_label = word_part[0], word_part[-1]
            collect_word.append(word_original)
            collect_label.append(word_label)


    verbs = []
    if "rel" not in collect_label: ##error line
        flag=0
        print "error line is:",line
        return flag, collect_word, collect_label, verbs, collect_distance
    else:
        rel_index = collect_label.index("rel")
        verb = collect_word[rel_index]
        indexes = np.arange(len(collect_word))
        indexes = indexes - rel_index
        collect_distance = list(indexes)
        collect_distance = map(lambda x: 1 if abs(x) <= 2 else 0, collect_distance)
        verbs = [verb] * len(collect_label)

        for label_index in range(len(collect_label)):
            if collect_label[label_index] == "rel" or collect_label[label_index] == "O":
                continue
            else:
                collect_label[label_index] = "S" +collect_label[label_index][1:]

        return flag,collect_word, collect_label, verbs, collect_distance



def create_label_dict(labels):      # create dict of unique labels
    label_dict = {}
    for single_label_index in range(len(labels)):
        label_dict[labels[single_label_index]] = single_label_index
    return label_dict
def create_word_dict(d,dev_d,test_d):            # create a dict of unique words
    chars = []  # 统计所有字，跟每个字编号
    for i in d['data']:
        chars.extend(i)
    for i in dev_d['data']:
        chars.extend(i)
    for i in test_d['data']:
        chars.extend(i)

    chars = pd.Series(chars).value_counts()
    chars[:] = range(1, len(chars) + 1)
    print "finish create word dict"
    return chars

def process_dataframe(d_temp,labels):  # prepare the dataFrame further, make it ready for models
    pre_vector = list([0.] * (len(labels) - 1))
    pre_vector.append(1.0)

    d_temp['original_sentences'] = map(lambda x: np.array(list(chars[x]) + [0] * (maxlen - len(x))),d_temp['data'])
    d_temp['processed_verb'] = map(lambda x: np.array(list(chars[x]) + [0] * (maxlen - len(x))),d_temp['verbs'])
    d_temp['y'] = map(lambda x: np.array(
        list(map(lambda y: np_utils.to_categorical(y, len(labels)), tag[x].values.reshape((-1, 1)))) + [
            np.array(pre_vector)] * (maxlen - len(x))),d_temp['label'])
    d_temp['processed_distance'] =map(lambda x: np.array(list(x) + [0] * (maxlen - len(x))),d_temp['distance'])
    return d_temp

def load_word2Vec_model(vocab):
    print "="*40
    print "loading word2Vec moel"
    w2vModel = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_directory, binary=True)

    vocab.index=map(lambda x: x.decode('utf-8'), vocab.index)
    vocabSize=len(vocab)

    embeddingDim = w2vModel.vector_size
    embeddingUnknown = [0 for i in range(embeddingDim)]
    embeddingWeights = np.zeros((vocabSize + 1, embeddingDim))

    num_unknown_word=0
    for word, index in vocab.iteritems():
        if word in w2vModel:
            e = w2vModel[word]
        else:
            num_unknown_word += 1
            e = embeddingUnknown
        embeddingWeights[index, :] = e
    print "num unknown words:",num_unknown_word
    print "finishing loading word2Vec"
    return embeddingDim,embeddingWeights




def handle_label(all_data_dict,d):
    print "="*40
    print "computing class weight"
    only_labels = []
    for element in list(d['label']):
        only_labels.extend(element)
        only_labels.extend(["O"] * (maxlen - len(element)))
    # check missing labels
    for element in labels:
        if element not in only_labels:
            only_labels.append(element)
    labels_df = pd.DataFrame(only_labels, columns=['all_labels'])
    labels_df['num_label'] = labels_df['all_labels'].apply(lambda x: tag[x])

    label_counts = dict(labels_df['all_labels'].value_counts())
    label_weight = compute_class_weight("balanced",np.array(labels),np.array(list(labels_df['all_labels'])))

    one_hot_weight_dict={}
    for label_index in range(len(labels)):
        weight=label_weight[label_index]
        tag_value=tag[labels[label_index]]
        key=np_utils.to_categorical(tag_value, len(labels))
        one_hot_weight_dict[tag_value]=weight
    #return one_hot_weight_dict
    print "finishing computing class weight"
    all_data_dict['one_hot_weight_dict']=one_hot_weight_dict
    return all_data_dict

def generate_weight_array(all_data_dict,d,dev_d):
    print "="*40
    print "generating weight array for umbalanced class"
    one_hot_weight_dict=all_data_dict['one_hot_weight_dict']
    weight_array=np.zeros((len(d.index),maxlen))
    for row_index in range(len(d.index)):
        original_labels=d['label'].iloc[row_index]
        weight_array[row_index]=np.array(list(map(lambda x:one_hot_weight_dict[tag[x]],original_labels))+[one_hot_weight_dict[0]]*(maxlen-len(original_labels)))

    all_data_dict["train"]["weight_array"]=weight_array

    weight_array=np.zeros((len(dev_d.index),maxlen))
    for row_index in range(len(dev_d.index)):
        original_labels=dev_d['label'].iloc[row_index]
        weight_array[row_index]=np.array(list(map(lambda x:one_hot_weight_dict[tag[x]],original_labels))+[one_hot_weight_dict[0]]*(maxlen-len(original_labels)))

    all_data_dict["dev"]["weight_array"]=weight_array
    return all_data_dict

def load_data_3DF():
    print "="*40
    lines = load_lines_from_txt(train_txt_directory)
    d = generate_dataset_DF(lines)
    labels = load_all_labels(lines)

    dev_lines = load_lines_from_txt(dev_txt_directory)
    dev_d=generate_dataset_DF(dev_lines)

    test_lines=load_lines_from_txt(test_txt_directory)
    test_d=generate_dataset_DF(test_lines,test=True)
    print "finish load 3DF"
    return labels,d,dev_d,test_d

def process_3DF(d,dev_d,test_d):
    print "=="*20
    print "processing train data"
    d = process_dataframe(d, labels)
    print "finish processing train data"

    print "processing dev data"
    dev_d = process_dataframe(dev_d, labels)
    print "finish processing dev data"

    print "processing test data"
    test_d = process_dataframe(test_d,labels)
    print "finishing processing test data"
    return d,dev_d,test_d

def prepare_data_for_model(d, dev_d, test_d):
    all_data_dict={}
    print "="*40
    print "preparing data for model"

    main_input_data = np.array(list(d['original_sentences']))
    aux_input_data=np.array(list(d['processed_verb']))
    aux2_input_data=np.array(list(d['processed_distance'])).reshape(-1,maxlen,1)
    main_output_data=np.array(list(d['y'])).reshape((-1,maxlen,len(labels)))
    train_data_dict={"main_input_data":main_input_data,"aux_input_data":aux_input_data,"aux2_input_data":aux2_input_data,"main_output_data":main_output_data}
    all_data_dict["train"]=train_data_dict

    main_input_data = np.array(list(dev_d['original_sentences']))
    aux_input_data=np.array(list(dev_d['processed_verb']))
    aux2_input_data=np.array(list(dev_d['processed_distance'])).reshape(-1,maxlen,1)
    main_output_data=np.array(list(dev_d['y'])).reshape((-1,maxlen,len(labels)))
    dev_data_dict = {"main_input_data": main_input_data, "aux_input_data": aux_input_data,
                       "aux2_input_data": aux2_input_data, "main_output_data": main_output_data}
    all_data_dict["dev"] = dev_data_dict

    main_input_data = np.array(list(test_d['original_sentences']))
    aux_input_data=np.array(list(test_d['processed_verb']))
    aux2_input_data=np.array(list(test_d['processed_distance'])).reshape(-1,maxlen,1)
    main_output_data=np.array(list(test_d['y'])).reshape((-1,maxlen,len(labels)))
    test_data_dict = {"main_input_data": main_input_data, "aux_input_data": aux_input_data,
                       "aux2_input_data": aux2_input_data, "main_output_data": main_output_data}
    all_data_dict["test"] = test_data_dict
    print "finish preparing data for model"
    return all_data_dict

def test_model(dev_d):

    test_row=[4,5]
    d2=dev_d.iloc[test_row] #d2 is for testing the result of d2 row

    test_main_input_data = np.array(list(d2['original_sentences']))
    test_aux_input_data = np.array(list(d2['processed_verb']))
    test_aux2_input_data = np.array(list(d2['processed_distance'])).reshape(-1, maxlen, 1)

    prediction = model.predict(
        {'main_input': test_main_input_data, 'aux_input': test_aux_input_data, 'aux2_input': test_aux2_input_data})

    count=0
    for single_line in prediction:
        sorted_proba_index=np.argsort(single_line,axis=1)
        labels_index=sorted_proba_index[:,-1]
        pred_labels=map(lambda x:labels[x],labels_index)

        word_single_line=d2.iloc[count]["data"]
        num_word_single_line=len(word_single_line)

        result=""
        for word_index in range(num_word_single_line):
            result=result+word_single_line[word_index]+"/"+pred_labels[word_index]+" "

        print result
        count +=1

def generate_prediction(prediction):
    result=np.empty((len(prediction),maxlen),dtype=object)
    for single_line_index in range(len(prediction)):

        sorted_proba_index=np.argsort(prediction[single_line_index],axis=1)
        labels_index=sorted_proba_index[:,-1]
        pred_labels=map(lambda x:labels[x],labels_index)
        result[single_line_index]=np.array(pred_labels)
    return result

def generate_txt_single_line(lines,line_index,prediction,dev_flag=False,test_flag=False):
    result=''
    temp_result=[]
    split_line = lines[line_index].split()
    begin_pos=0
    end_pos=len(split_line) - 1

    rel_index = 0
    for word in split_line:
        word_part = word.split("/")
        temp_result.append("/".join([word_part[0],word_part[1],'O']))
        if len(word_part) == 3 and word_part[-1] == "rel":
            rel_index = split_line.index(word)

    if len(split_line)>maxlen:
        begin_pos=max(0,rel_index-maxlen/2)
        end_pos=min(begin_pos+maxlen,len(split_line)-1)
        rel_index=rel_index-begin_pos
        split_line=split_line[begin_pos:end_pos]

    for changed_index in range(begin_pos,end_pos,1):
        if(changed_index-begin_pos)>=45:
            print changed_index-begin_pos
        temp_result[changed_index]="/".join(list(temp_result[changed_index].split("/")[:-1])+[prediction[line_index][changed_index-begin_pos]])
    result=temp_result
    return result

def generate_txt(dev_flag=False,test_flag=False):
    if dev_flag==True:

        lines = load_lines_from_txt(dev_txt_directory)
        lines.pop()
        test_main_input_data = np.array(list(all_data_dict["dev"]['DF']['original_sentences']))
        test_aux_input_data = np.array(list(all_data_dict["dev"]['DF']['processed_verb']))
        test_aux2_input_data = np.array(list(all_data_dict["dev"]['DF']['processed_distance'])).reshape(-1, maxlen, 1)

        prediction = model.predict(
            {'main_input': test_main_input_data, 'aux_input': test_aux_input_data, 'aux2_input': test_aux2_input_data})

        prediction=generate_prediction(prediction)
        pred_lines=map(lambda x:generate_txt_single_line(lines,lines.index(x),prediction,dev_flag=False,test_flag=False),lines)

        f_pred = open('generate_dev.txt', 'w')
        for pred_line in pred_lines:
            f_pred.write(" ".join(pred_line))
            f_pred.write('\n')
        f_pred.write('\n')
        f_pred.close()

    if dev_flag==False and test_flag==False:  #train set

        lines = load_lines_from_txt(train_txt_directory)
        lines.pop()
        test_main_input_data = np.array(list(all_data_dict["train"]['DF']['original_sentences']))
        test_aux_input_data = np.array(list(all_data_dict["train"]['DF']['processed_verb']))
        test_aux2_input_data = np.array(list(all_data_dict["train"]['DF']['processed_distance'])).reshape(-1, maxlen, 1)

        prediction = model.predict(
            {'main_input': test_main_input_data, 'aux_input': test_aux_input_data, 'aux2_input': test_aux2_input_data})

        prediction=generate_prediction(prediction)
        pred_lines=map(lambda x:generate_txt_single_line(lines,lines.index(x),prediction,dev_flag=False,test_flag=False),lines)

        f_pred = open('generate_train.txt', 'w')
        for pred_line in pred_lines:
            f_pred.write(" ".join(pred_line))
            f_pred.write('\n')
        f_pred.write('\n')
        f_pred.close()

    if test_flag==True:
        lines = load_lines_from_txt(test_txt_directory)
        lines.pop()
        test_main_input_data = np.array(list(all_data_dict["test"]['DF']['original_sentences']))
        test_aux_input_data = np.array(list(all_data_dict["test"]['DF']['processed_verb']))
        test_aux2_input_data = np.array(list(all_data_dict["test"]['DF']['processed_distance'])).reshape(-1, maxlen, 1)

        prediction = model.predict(
            {'main_input': test_main_input_data, 'aux_input': test_aux_input_data, 'aux2_input': test_aux2_input_data})

        prediction=generate_prediction(prediction)
        pred_lines=map(lambda x:generate_txt_single_line(lines,lines.index(x),prediction,dev_flag=False,test_flag=False),lines)

        f_pred = open('generate_test.txt', 'w')
        for pred_line in pred_lines:
            f_pred.write(" ".join(pred_line))
            f_pred.write('\n')
        f_pred.write('\n')
        f_pred.close()


def construct_model(embeddingWeights,embeddingDim,all_data_dict,continue_train=False,continue_epoch=None,continue_model=None):      # construct keras LSTM model
    if continue_train==True:
        model=continue_model
        history = model.fit({'main_input': all_data_dict['train']['main_input_data'],
                             'aux_input': all_data_dict['train']['aux_input_data'],
                             'aux2_input': all_data_dict['train']['aux2_input_data']},
                            {'main_output': all_data_dict['train']['main_output_data']},
                            epochs=continue_epoch,
                            validation_data=({'main_input': all_data_dict['dev']['main_input_data'],
                                              'aux_input': all_data_dict['dev']['aux_input_data'],
                                              'aux2_input': all_data_dict['dev']['aux2_input_data']},
                                             {'main_output': all_data_dict['dev']['main_output_data']},
                                             all_data_dict['dev']["weight_array"]),
                            batch_size=batch_size,
                            sample_weight=all_data_dict['train']["weight_array"])
        return model,history

    main_input = Input(shape=(maxlen,), name='main_input')
    embedded_main = Embedding(output_dim = embeddingDim, input_dim = vocabSize + 1, weights = [embeddingWeights],mask_zero=True)(main_input)

    aux_input = Input(shape=(maxlen,), name='aux_input')
    embedded_aux = Embedding(output_dim = embeddingDim, input_dim = vocabSize + 1, weights = [embeddingWeights],mask_zero=True)(aux_input)

    aux2_input = Input(shape=(maxlen, 1), name='aux2_input')
    aux2_layer2 = LSTM(1, return_sequences=True, input_shape=(all_data_dict['train']['aux2_input_data'].shape[1], all_data_dict['train']['aux2_input_data'].shape[2]))(
        aux2_input)

    x = keras.layers.concatenate([embedded_main, embedded_aux, aux2_layer2])

    blstm1 = Bidirectional(LSTM(num_1, return_sequences=True), merge_mode='sum')(x)
    blstm2 = Bidirectional(LSTM(num_2, return_sequences=True), merge_mode='sum')(blstm1)
    drop_layer=Dropout(0.1)(blstm2)
    blstm3 = Bidirectional(LSTM(num_3, return_sequences=True), merge_mode='sum')(drop_layer)
    main_output = TimeDistributed(Dense(len(labels), activation='softmax'), name='main_output')(blstm3)
    model = Model(inputs=[main_input, aux_input, aux2_input], outputs=[main_output])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'],sample_weight_mode="temporal")
    print(model.summary())

    #plot_model(model, to_file='./model_plot.png', show_shapes=True, show_layer_names=True)
    checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
    history = model.fit({'main_input': all_data_dict['train']['main_input_data'],
                         'aux_input': all_data_dict['train']['aux_input_data'],
                         'aux2_input': all_data_dict['train']['aux2_input_data']},
                        {'main_output': all_data_dict['train']['main_output_data']},
                        epochs=max_epochs,
                        validation_data=({'main_input': all_data_dict['dev']['main_input_data'],
                         'aux_input': all_data_dict['dev']['aux_input_data'],
                         'aux2_input': all_data_dict['dev']['aux2_input_data']},
                        {'main_output': all_data_dict['dev']['main_output_data']},
                        all_data_dict['dev']["weight_array"]),
                        batch_size=batch_size,
                        sample_weight=all_data_dict['train']["weight_array"])
    return model,history

if __name__=="__main__":



    reload(sys)
    sys.setdefaultencoding('utf8')
    andy=False

    if andy:
        labels, d, dev_d, test_d=load_data_3DF()
        label_dict=create_label_dict(labels)

        tag = pd.Series(label_dict)

        chars=create_word_dict(d,dev_d,test_d)

        d, dev_d, test_d=process_3DF(d,dev_d,test_d)
        all_data_dict=prepare_data_for_model(d, dev_d, test_d)

        vocab=chars  # stored in pd.Series type all words and corresponding index
        vocabSize = len(vocab)

        embeddingDim, embeddingWeights=load_word2Vec_model(vocab)
        all_data_dict =one_hot_weight_dict=handle_label(all_data_dict,d)
        all_data_dict=generate_weight_array(all_data_dict,d,dev_d)

        all_data_dict["train"]['DF']=d
        all_data_dict["dev"]['DF']=dev_d
        all_data_dict['test']['DF']=test_d
        all_data_dict["labels"] = labels
        print "dumping data"
        pickle.dump(all_data_dict,open("./all_data_dict.pkl",'w'))
        print "finish dumping"

        model,history=construct_model(embeddingWeights,embeddingDim,all_data_dict)
        model.save('./my_model.h5')
        #test_model(dev_d)
        #save all data processed


    else:  
        print "="*40
        print "loading data"
        all_data_dict=pickle.load(open("all_data_dict.pkl","r"))
        print "finish loading data"
        d=all_data_dict["train"]['DF']
        dev_d=all_data_dict["dev"]['DF']
        test_d=all_data_dict['test']['DF']
        labels=all_data_dict["labels"]

        label_dict = create_label_dict(labels)
        tag = pd.Series(label_dict)
        chars = create_word_dict(d, dev_d, test_d)
        vocab=chars  # stored in pd.Series type all words and corresponding index
        vocabSize = len(vocab)

        embeddingDim, embeddingWeights = load_word2Vec_model(vocab)
        # use pre-trained model,可以改变continue_epoch
        model = load_model('./my_model.h5')

        train_times = 100
        # for times in range(train_times):
        #
        #     print times
        #     generate_txt(dev_flag=False, test_flag=True)
        #     generate_txt(dev_flag=True, test_flag=False)
        #     model, history = construct_model(embeddingWeights, embeddingDim, all_data_dict, continue_train=True,
        #                            continue_epoch=5,
        #                            continue_model=model)
        #     model.save('./my_model.h5')


        #model,history= construct_model(embeddingWeights, embeddingDim, all_data_dict, continue_train=True,
        #                        continue_epoch=2,
        #                        continue_model=model)

        #plot_history_graph(history)
        #  save model to your computer  训练完之后将model保存到本地 下一次训练的时候可以load这次训练的结果然后继续训练
        # model.save('./my_model.h5')

    #  load model from your computer
    #model = load_model('./my_model.h5')

    #想再训练的话：
    #model=construct_model(embeddingWeights,embeddingDim,all_data_dict,continue_train=True,continue_epoch=5,continue_model=model)

    #生成dev 的预测txt
    #generate_txt(dev_flag=True, test_flag=False)

    #生成test 的预测txt
    #generate_txt(dev_flag=False,test_flag=True)
    # train for beginning
    #model = construct_model(embeddingWeights, embeddingDim, all_data_dict)
