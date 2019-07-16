import sqlite3 as lite
import os
import csv
import numpy as np
from keras.utils import to_categorical
import question_features as qf
from stop_words import get_stop_words
from gensim.models import KeyedVectors
import gc
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from unicodedata import normalize, category
import re
from nltk.stem import SnowballStemmer
import pickle

proj_dir = os.path.abspath(os.path.join('.'))
elem_dev_queries = os.path.abspath(os.path.join(proj_dir,'data','Elem-Dev.csv'))
elem_dev_expl = os.path.abspath(os.path.join(proj_dir,'data','Elem-Dev-Expl.csv'))
all_dev_expl = os.path.abspath(os.path.join(proj_dir,'../all_dev_qae.csv'))


MAX_NUM_WORDS = 5000
MAX_SEQUENCE_LENGTH = 160

tokenizer_path = "ranking/tensorflow_ranking/examples/data/tokenizer.pickle"


def remove_stops(string):
    ostop_words = ["!!", "?!", "??", "!?", "`", "``", "''", "-lrb-", "-rrb-", "-lsb-", "-rsb-", ",", ".", ":", ";",
                   "\"",
                   "'", "?", "<", ">", "{", "}", "[", "]", "+", "-", "(", ")", "&", "%", "$", "@", "!", "^", "#", "*",
                   "..", "...", "'ll", "'s", "'m", "a", "about", "above", "after", "again", "against", "all", "am",
                   "an",
                   "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
                   "between", "both", "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do",
                   "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
                   "had",
                   "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
                   "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
                   "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
                   "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
                   "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
                   "she'll", "she's", ",should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
                   "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
                   "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
                   "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
                   "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
                   "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
                   "yours", "yourself", "yourselves", "###", "return", "arent", "cant", "couldnt", "didnt", "doesnt",
                   "dont", "hadnt", "hasnt", "havent", "hes", "heres", "hows", "im", "isnt", "its", "lets", "mustnt",
                   "shant", "shes", "shouldnt", "thats", "theres", "theyll", "theyre", "theyve", "wasnt", "were",
                   "werent", "whats", "whens", "wheres", "whos", "whys", "wont", "wouldnt", "youd", "youll", "youre",
                   "youve"]

    istop_words = get_stop_words('en')

    stop_words = list(set(ostop_words).union(set(istop_words)))

    string = string.replace(",","")
    string = string.replace("'s", "")
    return_str = " ".join([word for word in string.split(" ") if word not in stop_words])
    return return_str


def clean_text(text, remove_stopwords=True, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = ''.join([c for c in normalize('NFD', u'' + text) if category(c) != 'Mn'])
    text = text.strip().replace('/', ' ').replace('?', '').lower().split(' ')
    stops = [u'all', u'just', u"don't", u'being', u'over', u'both', u'through', u'yourselves', u'its', u'before', \
             u'o', u'don', u'hadn', u'herself', u'll', u'had', u'should', u'to', u'only', u'won', u'under', u'ours', \
             u'has', u"should've", u"haven't", u'do', u'them', u'his', u'very', u"you've", u'they', u'not', u'during', \
             u'now', u'him', u'nor', u"wasn't", u'd', u'did', u'didn', u'this', u'she', u'each', u'further', u"won't", \
             u'where', u"mustn't", u"isn't", u'few', u'because', u"you'd", u'doing', u'some', u'hasn', u"hasn't",
             u'are', \
             u'our', u'ourselves', u'out', u'what', u'for', u"needn't", u'below', u're', u'does', u"shouldn't",
             u'above', \
             u'between', u'mustn', u't', u'be', u'we', u'who', u"mightn't", u"doesn't", u'were', u'here', u'shouldn', \
             u'hers', u"aren't", u'by', u'on', u'about', u'couldn', u'of', u"wouldn't", u'against', u's', u'isn', u'or', \
             u'own', u'into', u'yourself', u'down', u"hadn't", u'mightn', u"couldn't", u'wasn', u'your', u"you're",
             u'from', \
             u'her', u'their', u'aren', u"it's", u'there', u'been', u'whom', u'too', u'wouldn', u'themselves', u'weren', \
             u'was', u'until', u'more', u'himself', u'that', u"didn't", u'but', u"that'll", u'with', u'than', u'those',
             u'he', \
             u'me', u'myself', u'ma', u"weren't", u'these', u'up', u'will', u'while', u'ain', u'can', u'theirs', u'my', \
             u'and', u've', u'then', u'is', u'am', u'it', u'doesn', u'an', u'as', u'itself', u'at', u'have', u'in',
             u'any', \
             u'if', u'again', u'no', u'when', u'same', u'how', u'other', u'which', u'you', u"shan't", u'shan', u'needn', \
             u'haven', u'after', u'most', u'such', u'why', u'a', u'off', u'i', u'm', u'yours', u"you'll", u'so', u'y',
             u"she's", \
             u'the', u'having', u'once']

    # Correct the misstyped words"
    # text = [w if w.strip().lower() not in correction_dict else correction_dict[w.strip().lower()] for w in text]

    # Optionally, remove stop words
    if remove_stopwords:
        text = [w for w in text if w not in stops]

    text = " ".join(text)

    text = re.sub("’s", "", text)
    text = re.sub("’", "", text)
    text = re.sub("\"", "", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split(' ')
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a string
    return text


def prep_data(X, Y=[],wi=False):
    # Clean the text
    X_clean = []
    for x in X:
        X_clean.append(clean_text(x, remove_stopwords=True, stem_words=False))

    if wi:
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS) #num_words=MAX_NUM_WORDS
        tokenizer.fit_on_texts(X_clean)

        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)

    sequences = tokenizer.texts_to_sequences(X_clean)
    word_index = tokenizer.word_index
    X_seq = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    if Y:
        labels = to_categorical(np.asarray(Y))
    else:
        labels = []
    return X_clean, X_seq, labels, word_index


def add_embeddings():

    train_file_wScore = "ranking/tensorflow_ranking/examples/data/train_with_scores_ids.csv"
    train_fixed_wEmbeddings = "ranking/tensorflow_ranking/examples/data/train-fixed-wEmbeddings.txt"
    train_var_wEmbeddings = "ranking/tensorflow_ranking/examples/data/train-var-wEmbeddings.txt"
    train_avg_wEmbeddings_fixed_300 = "ranking/tensorflow_ranking/examples/data/train-avg-wEmbeddings-fixed-300.txt"

    model = KeyedVectors.load_word2vec_format('ranking/tensorflow_ranking/embeddings/numberbatch-en-17.06.txt', binary=False)
    #vec = model['person']

    with open(train_file_wScore, 'r') as myfile:
        reader = csv.reader(myfile,delimiter='\t')

        str_row = []
        X = []
        all_ids = []
        for row in reader :

            ids = row[0]
            sequence = row[3]+" SPR '"+row[4]+" SPR "+" ".join(row[5].split(" ")[1:])+" SPR "+row[6]
            # sequence = sequence.replace""

            sequence = sequence.translate(str.maketrans('', '', string.punctuation))

            X.append(sequence)

            str_row.append([ids,sequence])
            all_ids.append(ids)

    myfile.close()

    print("Lenth Sequences : ",len(X))
    # X = asarray(np.asmatrix(str_row)[:,1])

    X_clean,X_seq, labels, word_index = prep_data(X)

    for id,x,Xs in zip(all_ids[233615:],X_clean[233615:],X_seq[233615:]) :
        print("\n\n")

        row4_fixed = id
        row4_var = id
        Vectors_Matrix = np.zeros((1,300))

        actual_len = len(x.split(" "))
        len_zeros = len(Xs) - actual_len

        #Add the Zeros for fixed Sizes one
        for i in range(len_zeros):
            row4_fixed = row4_fixed +' 0:0'

        # len_zeros=
        for j in range(actual_len):
            wd = x.split(" ")[j]
            feat_idx = j+len_zeros

            if len(wd.strip()) < 1:
                pass
            elif wd == 'spr':
                # row = row+" " + str(word_index[wd]) + ":" + str(np.average(np.ones((1, 300), dtype=np.int)))
                row4_fixed = row4_fixed + " " + str(Xs[feat_idx]) + ":" + str(np.average(np.ones((1, 300), dtype=np.int)))
                row4_var = row4_var + " " + str(Xs[feat_idx]) + ":" + str(np.average(np.ones((1, 300), dtype=np.int)))
                Vectors_Matrix += Xs[feat_idx]
            else:
                try:
                    vec=model[wd]
                except:
                    vec=np.zeros((1,300))
                avg = np.average(vec)
                # row = row+" " + str(word_index[wd]) + ":" + str(avg)

                row4_fixed = row4_fixed+" " + str(Xs[feat_idx]) + ":" + str(avg)
                row4_var = row4_var+ " " + str(Xs[feat_idx]) + ":" + str(avg)
                Vectors_Matrix += vec

        print("4_fixed : \t",row4_fixed)
        print("4_var   : \t",row4_var)

        # Vectors_Matrix = np.average(np.asanyarray(Vectors_Matrix),axis=0)
        avg_vec = Vectors_Matrix/actual_len
        context_vec = id

        z = 1
        for val in avg_vec[0] :
            context_vec += " "+str(z)+":"+str(val)
            z+=1

        print(context_vec)

        eff = open(train_fixed_wEmbeddings,"a+")
        eff.write(row4_fixed+"\n")
        eff.close()

        evf = open(train_var_wEmbeddings, "a+")
        evf.write(row4_var+"\n")
        evf.close()

        eavf = open(train_avg_wEmbeddings_fixed_300, "a+")
        eavf.write(context_vec+"\n")
        eavf.close()

def create_file_for_embedding():
    # model = KeyedVectors.load_word2vec_format('numberbatch-en-17.04b.txt', binary=False)
    #Load the scores :
    scores_file = "ranking/tensorflow_ranking/examples/data/train_scores_ids.txt"
    dev_scores_file = "ranking/tensorflow_ranking/examples/data/dev_scores_ids.txt"
    train_file_wScore = "ranking/tensorflow_ranking/examples/data/train_with_scores_ids.csv"
    dev_file_wScore = "ranking/tensorflow_ranking/examples/data/dev_with_scores_ids.csv"

    scores = []
    with open(scores_file) as f:

        lines = f.readlines()
        i=0
        for line in lines:
            scores.append(line.strip())
            i+=1
        del lines
        gc.collect()

    print("Lenth : ",len(scores))

    # print(scores)
    #
    db_file = "qae_sqlite/qae.sqlite"
    select_dev_query = "SELECT edq.q_id, edq.e_id, dq.q_string,dq.answer,dq.other_choices, e.explanation FROM dev_data \
                        AS edq LEFT JOIN (SELECT DISTINCT * FROM dev_question) AS dq ON edq.q_id=dq.q_id, (SELECT DISTINCT * FROM explanation) AS e ON edq.e_id=e.e_id ; "
                        # ASC LIMIT 20000 OFFSET 0;"

    select_train_query = "SELECT jq.q_id, jq.e_id, q.q_string,q.answer,q.other_choices, e.explanation FROM training_data \
                            AS jq  JOIN question AS q ON jq.q_id=q.q_id, explanation AS e ON jq.e_id=e.e_id ; "

    con = lite.connect(db_file)
    curr_list = []

    with con:
        cur = con.cursor()

        cur.execute(select_train_query)
        data = cur.fetchall()
        print("Retrieved : ",len(data))
        i=0
        max = 0
        for d in data:
            # if i> 1082317:
            #     break
            d=list(d)

            if d[4] :
                d[4] = d[4].replace(","," ")
                # l = len(d[2].split(" ")) + len(d[3].split(" ")) + len(d[4].split(" ")) + len(d[5].split(" "))
                #
                # if l> max:
                #     max=l
                new_d = [scores[i]]

                new_d.extend([remove_stops(val.lower()) for val in d])
                curr_list.append(new_d)

            i+=1
    con.close()

    # print("\n\nFIRST 20 : \n")
    # for i in range(20):
    #     print(curr_list[i])
    #
    # print("\n\nlast : \n")
    # for i in range(1,20):
    #     print(curr_list[-i])

    #SAVE THEM TO FILE
    with open(train_file_wScore, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,delimiter='\t')
        wr.writerows(curr_list)

if __name__ == "__main__":

    #IMPORT DEV.CSV TO DEV_QUESTIONS

    # with open(elem_dev_queries) as f:
    #     reader = list(csv.reader(f, delimiter='\t'))
    #
    #     dict_answ_mappings = {'A':3,'B':4,'C':5,'D':6,'E':7}
    #
    #     con = lite.connect('qae_sqlite/qae.sqlite')
    #
    #     with con:
    #         cur = con.cursor()
    #         i=0
    #         for row in reader:
    #             if i< 258:
    #                 i += 1
    #                 continue
    #
    #             answ_idx = dict_answ_mappings[row[2].strip()]
    #             q_id = row[0].strip()
    #             answer = row[answ_idx]
    #             answer = answer.replace("'s","")
    #             q_string = row[1]
    #             q_string = q_string.replace("'s","")
    #             q_string = q_string.replace("s'", "s")
    #
    #             other_choices_str = " "
    #             other_keys = set(list(dict_answ_mappings.values())).difference(set([answ_idx]))
    #
    #             for ok in other_keys:
    #                 if len(row) > int(ok):
    #                     other_choices_str += "," + row[ok]
    #
    #             other_choices_str = other_choices_str.replace("'s", "")
    #             other_choices_str = other_choices_str.replace("'s", "")
    #             other_choices_str = other_choices_str.replace("s'", "s")
    #
    #
    #             #Process the query
    #             question = qf.Question(q_string)
    #             question.process()
    #
    #             focus = ",".join(question.get_focus())
    #             pentree = question.get_pentree()
    #             scores = [str(s[-1]) for s in question.all_word_features()]
    #
    #             insert_query = "INSERT INTO dev_question(`q_id`, `q_string`, `answer`, `other_choices`,`focus`,`scores`,`q_pen_tree`) " \
    #                            "VALUES('" + q_id + "', '" + q_string + "', '" + answer + "', '" + other_choices_str.replace("'s","").strip(" ").strip(",") + "', '" + focus + "', '"+",".join(scores)+"', '"+pentree+"');"
    #
    #             # update_query = "UPDATE dev_question SET `q_id` = '" +  + "' `answer` = '" ++ " `q_string` = '" + \
    #             #                 + "' `other_choices`= '" + + "';"
    #             print(i,"\t",insert_query)
    #             cur.execute(insert_query)
    #
    #
    #             con.commit()
    #
    #             i+=1
    #         # con.commit()
    #         con.close()


    #POPULATE DEV_EXP JOIN TABLE
    # con = lite.connect('qae_sqlite/qae.sqlite')
    #
    # with con:
    #     cur = con.cursor()
    #
    #     with open(all_dev_expl) as f:
    #         reader = list(csv.reader(f, delimiter='\t'))
    #
    #         i=0
    #         for row in reader:
    #             # if i< 148:
    #             #     i += 1
    #             #     continue
    #
    #             q_id = row[0].strip()
    #             e_id = row[1].strip()
    #
    #             try:
    #                 insert_query = "INSERT INTO Dev_Q_E_Join(`q_id`, `e_id`, `truth_value`) " \
    #                                "VALUES('" + q_id + "', '" + e_id + "', '0' );"
    #
    #                 print("\n",i,insert_query)
    #                 cur.execute(insert_query)
    #
    #             except:
    #                 pass
    #
    #             if i%10:
    #                 con.commit()
    #
    #             i+=1
    #
    #     con.commit()
    # con.close()

    # with open(elem_dev_expl) as f:
    #     reader = list(csv.reader(f, delimiter='\t'))
    #     con = lite.connect('qae_sqlite/qae.sqlite')
    #
    #     with con:
    #         cur = con.cursor()
    #         i=0
    #         for row in reader:
    #             q_id = row[0].strip()
    #             e_id = row[1]
    #
    #             update_query = "UPDATE Dev_Q_E_Join SET `truth_value` = '1' WHERE q_id='" + q_id + "' AND `e_id` = '" + e_id + "';"
    #             print("\n",i,"\t",update_query)
    #
    #             cur.execute(update_query)
    #             con.commit()
    #
    #             i+=1
    #
    #     con.commit()
    #     con.close()


    #POPULATE DEV_EXP JOIN TABLE
    # con = lite.connect('qae_sqlite/qae.sqlite')
    #
    # with con:
    #     cur = con.cursor()
    #
    #     with open(all_dev_expl) as f:
    #         reader = list(csv.reader(f, delimiter='\t'))
    #
    #         i=0
    #         for row in reader:
    #             # if i< 148:
    #             #     i += 1
    #             #     continue
    #
    #             q_id = row[0].strip()
    #             e_id = row[1].strip()
    #
    #             try:
    #                 insert_query = "INSERT INTO Dev_Q_E_Join(`q_id`, `e_id`, `truth_value`) " \
    #                                "VALUES('" + q_id + "', '" + e_id + "', '0' );"
    #
    #                 print("\n",i,insert_query)
    #                 cur.execute(insert_query)
    #
    #             except:
    #                 pass
    #
    #             if i%10:
    #                 con.commit()
    #
    #             i+=1
    #
    #     con.commit()
    # con.close()

    # ## DUMP PENTREES
    # # select_question_query = "SELECT eq.q_id, eq.e_id, q.q_pen_trees, e.e_pen_trees FROM Q_E_Join AS eq JOIN question AS q ON eq.q_id=q.q_id, explanation AS e ON eq.e_id=e.e_id ORDER BY eq.q_id;"
    # select_dev_query = "SELECT edq.q_id, edq.e_id, dq.q_pen_trees, e.e_pen_trees FROM Dev_Q_E_Join AS edq JOIN dev_question AS dq ON edq.q_id=dq.q_id, explanation AS e ON edq.e_id=e.e_id ORDER BY edq.q_id;"
    #
    #
    # con = lite.connect('qae_sqlite/qae.sqlite')
    # curr_list = []
    #
    # with con:
    #     cur = con.cursor()
    #
    #     cur.execute(select_dev_query)
    #     data = cur.fetchall()
    #
    #     for d in data:
    #         print(d)
    #         curr_list.append([val for val in d])
    # con.close()
    #
    # with open("data/dev_pen_trees.csv","w") as dev_file:
    #     wr = csv.writer(dev_file, quoting=csv.QUOTE_ALL)
    #     wr.writerows(curr_list)


    ## CREATE THE EMBEDDINGS DATASET
    # create_training_embedded()


    ##CREATE
    add_embeddings()
