import json
import numpy as np
import jieba
import string
import timeit
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
import os

# 自定义打印方法
def print_format(str, a):
    print(str + '\n{0}\n'.format(a))

# # cut_list
# def cut_list(list_input, isSpecialHandle=True):
    # list_new = []
    # for sentence in list_input:
        # if isSpecialHandle:
            # list_new.append(sentence.replace('?','').split())
        # else:
            # list_new.append(sentence.split())
    # return list_new



# 分词
def cut(input_list):
    list_new = []
    for q in input_list:
        list_new.append(q.replace('?','').split(' '))
    return list_new


# #handle_one_sentence
# def handle_one_sentence(sentence):
    # return sentence.replace('?','').split(' ')


def get_least_numbers_big_data(alist, k):
    max_heap = []
    length = len(alist)
    if not alist or k <= 0 or k > length:
        return
    k-=1
    for ele in alist:
        if len(max_heap) <= k:
            heapq.heappush(max_heap, ele)
        else:
            heapq.heappushpop(max_heap, ele)

    # return list(map(lambda x:x, max_heap))
    return max_heap

# ==============================第一部分：对于训练数据的处理：读取文件和预处理=======================

# 文本的读取： 需要从文本中读取数据，此处需要读取的文件是dev-v2.0.json，并把读取的文件存入一个列表里（list）
def read_corpus():
    #解析json数据
    #Tips1:答案字典“answers”可能会有空的情况 此时应该取plausible_answers节点
    qlist = []
    alist = []

     with open(file_path, 'r') as path:
        json_data = json.load(path)
        #print(json_data)
        
    qlist = [] # 问题列表
    alist = [] # 答案列表
    
    data = json_data['data']
    
    for eachdata in data:
        for eachqas in eachdata['paragraphs']:
            for qa in eachqas['qas']:
                 if(len(qa['answers']))>0: # 有很多问题没有答案，所以需要判断去掉没有答案的数据=
                    qlist.append(qa['question'])
                    alist.append(qa['answers'][0]['text'])
    assert len(qlist) == len(alist)  # 确保长度一样
    return qlist, alist


qlist, alist = read_corpus('train-v2.0.json')
qlist_new = [q for l in cut(qlist) for q in l] # 分词
dif_word_total = len(qlist_new) # 所有单词的个数

word_dict = Counter(qlist_new) # Counter({单词：词频})
word_total = len(dict(word_dict)) 
word_total_unique = [word for word in dict(word_dict)]  # 单词

print ("一共出现了 %d 个单词"%dif_word_total) # 903411
print ("共有 %d 个不同的单词"%word_total) #51979 



# 低频词库
low_frequency_words = []
for (k,v) in  word_dict.items():
    if v < 2:
        low_frequency_words.append(k)
 


def text_preprocessing(input_list):
    """预处理"""
    
    stop_words = set(stopwords.words('english')) # 停用词
    stemmer = PorterStemmer() # 词干提取
    input_list = cut(input_list)  # 分词  
    new_list = [] #保存处理完的qlist\alist
    
    for l in input_list:
        l_list = '' # 保存句子
        for word in l:
            word = word.lower()      # 1.转换小写
            word = stemmer.stem(word) # 2.词干提取
            word = ''.join(c for c in word if c not in string.punctuation)  # 3.去除所有标点符号
            
            if word.isdigit(): # 4. 处理数字
                word = word.replace(word,'#number')
            
            if word not in stop_words and word not in low_frequency_words: # 5.去停用词 6.过滤低频词
                l_list += word + ' '
        new_list.append(l_list)
    return new_list
  

qlist = text_preprocessing(qlist)   # 预处理后的问题列表


# =====================================glove 方式  开始============================================

embeddings_index = {}
glovefile = open("glove.6B.200d.txt", "r", encoding="utf-8")

for line in glovefile:
    values = line.split()
    word = values[0] # 单词
    coefs = np.asarray(values[1:], dtype='float16') # 词向量
    embeddings_index[word] = coefs # embeddings_index={单词：词向量}
glovefile.close()

embedding_dim = 200
# 获取单词word对应的词向量
def get_embedding_matrix_glove(word):
    embedding_vector = embeddings_index.get(word) # 单词对应的词向量
    if embedding_vector is not None:
        return embedding_vector[:embedding_dim]
    return np.zeros(embedding_dim)

word2id, id2word = {}, {}  # 单词-id字典 ，id-单词字典
emd = []
for word in word_total_unique:
    if word not in word2id:
        word2id[word] = len(word2id) 
        id2word[len(id2word)] = word
        emd.append(get_embedding_matrix_glove(word))
emd = np.asarray(emd)

dict_related = {word:[] for word in word_total_unique}  #{'When': [],'did': [],'Beyonce': [],
emd_csr_matrix = scipy.sparse.csr_matrix(emd)

test_count = 0

for key in dict_related.keys():

    word_index = word2id[key]


    result = list(cosine_similarity(emd_csr_matrix[word_index], emd_csr_matrix)[0])

    top_values = sorted(get_least_numbers_big_data(result, 10), reverse=True)

    top_idxs = []
    len_result = len(result)
    dict_visited = {}
    for value in top_values:
        for i in range(len_result):
            if value == result[i] and i not in dict_visited and word_index != i:
                top_idxs.append(i)
                dict_visited[i] = True

    top_idxs = top_idxs[:10]

    word_total_unique = np.array(word_total_unique)
    dict_related[key] = list(word_total_unique[top_idxs])


# print("dict_related", dict_related)

file_store_path = 'related_words.txt'
if os.path.exists(file_store_path):
    os.remove(file_store_path)



with open(file_store_path, mode='w', encoding='utf-8') as file:
    for item in dict_related.items():
        r_l = " ".join(word for word in item[1])
        output = '{0},{1}'.format(item[0], r_l)
        file.write(output + "\n")
    file.close()
