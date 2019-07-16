#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:04:28 2017
@author: mulang
"""
import csv
import json
import os
import sentence_aggregation as sa
from sentence_aggregation import Graphlet
from sentence_aggregation import Graph
import question_features as qf
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
import gc
import sqlite3 as lite
import time
import parse_utils as pu
import re

proj_dir = os.path.abspath(os.path.join('.'))

elem_train_expl = os.path.abspath(os.path.join(proj_dir,'data','Elem-Train-Expl.csv'))
elem_dev_expl = os.path.abspath(os.path.join(proj_dir,'data','Elem-Dev-Expl.csv'))
negative_train_sample = os.path.abspath(os.path.join(proj_dir,"data","negative_examples_train-ids.csv"))
all_Explanations = os.path.abspath(os.path.join(proj_dir,"data","expl-tablestore-cleaned.csv"))

elem_train_queries = os.path.abspath(os.path.join(proj_dir,'data','Elem-Train.csv'))
elem_dev_queries = os.path.abspath(os.path.join(proj_dir,'data','Elem-Dev.csv'))

graphlets_folder = os.path.abspath(os.path.join(proj_dir,"data","graphlets"))

all_feature_file = os.path.abspath(os.path.join(proj_dir,'data','qae_features.json'))


def load_questions():
    elem_dev_quex = {}
    elem_train_quex = {}
    elem_train_negs = []

    explanations_file = "data/expl-tablestore-cleaned.csv"
    expl_sents = {}
    with open(explanations_file, encoding='utf-8') as f:
        reader = list(csv.reader(f, delimiter='\t'))

        for row in reader :
            expl_sents[row[0].strip()] = row[1]
            print(expl_sents[row[0].strip()])


    # # NOW LOAD THE NEGATIVES DATASET
    # with open(negative_train_sample, encoding='utf-8') as f:
    #     reader = list(csv.reader(f, delimiter='\t'))
    #     elem_train_negs = {}
    #
    #     list_equal_id = []
    #     prev_id = ""
    #
    #     for row in reader :
    #
    #         if prev_id == row[0] :
    #             pass
    #
    #         elif prev_id == "" :
    #             prev_id = row[0]
    #
    #         else :
    #             elem_train_negs[prev_id] = list_equal_id
    #             prev_id = row[0]
    #             list_equal_id = []
    #
    #         new_row = ["","","","",""]
    #         new_row[0] = row[0]                         # Question ID
    #         new_row[1] = row[1]                         # Question ID
    #         new_row[4] = expl_sents[row[1].strip()]     # Explanation Itself
    #         list_equal_id.append(new_row)
    #
    # # TRAIN QAE
    # with open(elem_train_expl) as f:
    #     reader = list(csv.reader(f, delimiter='\t'))
    #     curr_id = ""
    #
    #     # saved_neg_idxs = []
    #
    #     for row in reader:
    #         new_row = ["", "", "", "", ""]
    #         new_row[0] = row[0]             # Question ID
    #         new_row[1] = row[1]             # Explanation ID
    #         new_row[4] = row[2]             # Explanation Itself
    #
    #         if row[0] == curr_id:
    #             elem_train_quex[curr_id].append(new_row)
    #
    #         else:
    #             curr_id = row[0]
    #             elem_train_quex[curr_id] = [new_row]
    #
    #             try:
    #                 neg_smpl = elem_train_negs[curr_id]
    #                 elem_train_quex[curr_id].extend(neg_smpl)
    #             except:
    #                 print(neg_smpl)
    #
    #     # TRAIN QAE
    #     with open(elem_train_expl) as f:
    #         reader = list(csv.reader(f, delimiter='\t'))
    #         curr_id = ""
    #
    #         for row in reader:
    #             new_row = ["", "", "", "", ""]
    #             new_row[0] = row[0]  # Question ID
    #             new_row[1] = row[1]  # Explanation ID
    #             new_row[4] = row[2]  # Explanation Itself
    #
    #             if row[0] == curr_id:
    #                 elem_train_quex[curr_id].append(new_row)
    #
    #             else:
    #                 curr_id = row[0]
    #                 elem_train_quex[curr_id] = [new_row]
    #
    #                 try:
    #                     neg_smpl = elem_train_negs[curr_id]
    #                     elem_train_quex[curr_id].extend(neg_smpl)
    #                 except:
    #                     print(neg_smpl)


    #  DEV QAE
    with open(elem_dev_expl) as f:
        reader = list(csv.reader(f, delimiter='\t'))
        curr_id = ""
        for row in reader:
            new_row = ["", "", "", "", ""]
            new_row[0] = row[0]
            new_row[1] = row[1]
            new_row[4] = row[2]

            if row[0] == curr_id:
                elem_dev_quex[curr_id.strip()].append(new_row)

            else:

                curr_id = row[0]
                elem_dev_quex[curr_id.strip()] = [new_row]

                print(type(expl_sents))

                for key in expl_sents.keys() :
                    inner_row = []
                    if new_row[1] == key:
                        pass
                    else:
                        inner_row.append(curr_id)
                        inner_row.append(key)
                        inner_row.append("")
                        inner_row.append("")
                        inner_row.append(expl_sents[key])

                        elem_dev_quex[curr_id.strip()].append(inner_row)



    # Add Train Queries and Answers
    # with open(elem_train_queries) as f:
    #     reader = list(csv.reader(f, delimiter='\t'))
    #
    #     dict_answ_mappings = {'A':3,'B':4,'C':5,'D':6,'E':7}
    #
    #     for row in reader:
    #         answ_idx = dict_answ_mappings[row[2].strip()]
    #         key = row[0].strip()
    #
    #         if key not in elem_train_quex:
    #             continue
    #
    #         for e in elem_train_quex[key]:
    #             e[2] = row[1]
    #             e[3] = row[answ_idx]
    #
    #         other_keys = set(list(dict_answ_mappings.values())).difference(set([key]))
    #
    #         other_choices_str = " "
    #         for ok in other_keys:
    #             if len(row) > int(ok):
    #                 other_choices_str += "," + row[ok]
    #
    #         other_choices_str = other_choices_str.strip().strip(",")
    #
    #         for e in elem_train_quex[key]:
    #             e.append(other_choices_str)

            #Print

    # Add Dev Queries and Answers
    with open(elem_dev_queries) as f:
        reader = list(csv.reader(f, delimiter='\t'))

        dict_answ_mappings = {'A': 3, 'B': 4, 'C': 5, 'D': 6, 'E': 7}

        for row in reader:
            answ_idx = dict_answ_mappings[row[2].strip()]
            key = row[0].strip()

            if key not in elem_dev_quex:
                continue

            for e in elem_dev_quex[key]:
                e[2] = row[1]
                e[3] = row[answ_idx]

            other_keys = set(list(dict_answ_mappings.values())).difference(set([key]))

            other_choices_str = " "
            for ok in other_keys:
                # print(ok)
                if len(row) > int(ok) :
                    other_choices_str += ","+row[ok]

            other_choices_str = other_choices_str.strip().strip(",")

            for e in elem_dev_quex[key]:
                e.append(other_choices_str)


    questions_full = [['Q_ID','Expl_ID','Question','Answer','Explanation','Other_Choices']]
    k = 1
    for key,value in elem_dev_quex.items():
        for v in value :
            print(str(k)+" : ",v)
            questions_full.append(v)
            k += 1

    with open('data/all_dev_qae.csv', 'w+') as mycsv:
        csvWriter = csv.writer(mycsv, delimiter='\t')
        csvWriter.writerows(questions_full)

    return questions_full


def generate_qae_features(questions_full):

    all_graphlets = sa.load_Graphlets(graphlets_folder)


    full_features = {}
    feature_per_query = []
    previous_id = ""

    p = 1

    for line in questions_full:

        if p == 1:
            p+=1
            continue

        print("\n\n",line)
        q_id = line[0]

        if q_id == previous_id :
            pass
        elif len(previous_id)>1:
            full_features[previous_id] = feature_per_query
            features_per_query = []
            previous_id = q_id

        else :
            previous_id = q_id

        exp_id = line[1]
        q_string = line[2]
        answ_string = line[3]

        other_choices_words = line[-1]

        question = qf.Question(q_string)
        question.process()

        answer = qf.Question(answ_string)
        answer.process()

        expl_graphlet = all_graphlets[exp_id]

        tag_level_features = {}

        graphlet_words = [w.strip() for w in expl_graphlet.__str__().split(",")]

        intsect_q = set(question.get_focus()).intersection(set(graphlet_words))
        intsect_all_q = set(q_string.split(" ")).intersection(graphlet_words)

        intsect_a = set(answer.get_focus()).intersection(set(graphlet_words))

        tag_level_features["numFocusQ"] = len(intsect_q)
        tag_level_features["numFocusA"] = len(intsect_a)

        massFocusQ = 0.0
        massFocusA = 0.0

        if len(intsect_q)>0 :
            for w in list(intsect_q):
                massFocusQ += float(qf.get_word_rating(w)[2])

        # if len(intsect_a)>0 :
        #     for w in list(intsect_a):
        #         massFocusA += float(qf.get_word_rating(w)[2])

        tag_level_features["massFocusQ"] = massFocusQ
        # tag_level_features["massFocusA"] = massFocusA

        numRepeatedFocus = 0
        numOtherAnswerF = 0

        for focus in question.get_focus():
            qfocus_in_graphlets = [fw for fw in graphlet_words if fw == focus]
            if len(qfocus_in_graphlets) > 1 :
                numRepeatedFocus += 1

        for focus in answer.get_focus():
            afocus_in_graphlets = [fw for fw in graphlet_words if fw == focus]
            if len(afocus_in_graphlets) > 1:
                numRepeatedFocus += 1

        # other_choices_words = other_choices_words.replace(", "," ")
        # other_choices_words = other_choices_words.replace(" , ", " ")
        # other_choices_words = other_choices_words.replace(" ,", " ")
        # other_choices_words = other_choices_words.replace(",", " ")
        #
        # for other_word in other_choices_words.split(" "):
        #     rating = 0.0
        #     b = float(qf.get_word_rating(other_word)[2])
        #
        #     if b == 0 :
        #         pass
        #     else:
        #         rating = b
        #
        #     if rating > 2.9 and rating < 4.3 :
        #         ofocus_in_graphlets = [fw for fw in graphlet_words if fw == other_word]
        #         if len(ofocus_in_graphlets)>1:
        #             numOtherAnswerF += 1


        minConcShared = 6.0
        if len(intsect_all_q)>0:
            for iw in intsect_all_q:
                sc = float(qf.get_word_rating(iw)[2])

                if sc < minConcShared :
                    minConcShared = sc

        if minConcShared == 6.0 :
            minConcShared = 0.0

        tag_level_features["numRepeatedFocus"] = numRepeatedFocus
        tag_level_features["numOtherAnswerF"] = numOtherAnswerF
        tag_level_features["minConcShared"] = minConcShared


        # BRIDGE FEATURES
        # massMaxBridgeScore = 0.0
        # massMinBridgeScore = 0.0

        # bridge_intersect = intsect_q.intersection(intsect_a)
        # if len(bridge_intersect)>0 :
        #     for bw in bridge_intersect:
        #         sc = float(qf.get_word_rating(bw)[2])
        #
        #         if massMaxBridgeScore < sc :
        #             massMaxBridgeScore = sc
        #
        #         elif massMinBridgeScore > sc :
        #             massMinBridgeScore = sc
        #
        # massDeltaBridgeScore = massMaxBridgeScore - massMinBridgeScore
        #
        # tag_level_features["massMaxBridgeScores"] = massMaxBridgeScore
        # tag_level_features["massMinBridgeScore"] = massMinBridgeScore
        # tag_level_features["massDeltaBridgeScore"] = massDeltaBridgeScore


        all_f = {}
        # all_f["question_id"] = q_id
        all_f["explation_id"] = exp_id
        all_f["tag_level"] = tag_level_features
        all_f["graphlet_level"] = expl_graphlet.get_nugget_features()

        feature_per_query.append(all_f)

        # DUMP FEATURES PER FILE
        with open("data/qae_sep/all_features.txt", 'a+') as fp:
            fp.write(str(all_f) + "\n\n")

        p += 1

    # DUMP FEATURES TO JSON
    with open(all_feature_file, 'w') as fp:
        json.dump(full_features, fp)


def parallel_generate_qae_features(line):#,nugget_features):
    "select j.Q_ID, j.E_ID, q.q_string, q.answer, q.focus, q.other_choices, e.focus from Q_E_Join as j " \
    "inner join question as q on (q.q_id = j.q_id) inner join explanation as e on (e.e_id = j.e_id) "

    q_id = line[0]
    exp_id = line[1]
    q_string = line[2]
    answ_string = line[3]
    q_focus_words = line[4]
    other_choices_words = line[5]
    exp_focus = line[6]

    # try:
    answer_focus = [w for w in answ_string if w not in pu.get_stop_words()]

    tag_level_features = {}

    graphlet_words = [w.strip() for w in exp_focus.split(",")]
    q_focus = [w.strip() for w in q_focus_words.split(",")]

    intsect_q = set(q_focus).intersection(set(graphlet_words))
    intsect_all_q = set(q_string.split(" ")).intersection(graphlet_words)

    intsect_a = set(answer_focus ).intersection(set(graphlet_words))

    tag_level_features["numFocusQ"] = len(intsect_q)
    tag_level_features["numFocusA"] = len(intsect_a)

    massFocusQ = 0.0
    massFocusA = 0.0

    if len(intsect_q)>0 :
        for w in list(intsect_q):
            massFocusQ += float(qf.get_word_rating(w)[2])

    tag_level_features["massFocusQ"] = massFocusQ

    if len(intsect_a)>0 :
        for w in list(intsect_a):
            massFocusA += float(qf.get_word_rating(w)[2])

    numRepeatedFocus = 0
    numOtherAnswerF = 0

    for focus in q_focus_words.split(","):
        qfocus_in_graphlets = [fw for fw in graphlet_words if fw == focus]
        if len(qfocus_in_graphlets) > 1 :
            numRepeatedFocus += 1

    for focus in answer_focus :
        afocus_in_graphlets = [fw for fw in graphlet_words if fw == focus]
        if len(afocus_in_graphlets) > 1:
            numRepeatedFocus += 1

    other_choices_words = other_choices_words.replace(", "," ")
    other_choices_words = other_choices_words.replace(" , ", " ")
    other_choices_words = other_choices_words.replace(" ,", " ")
    other_choices_words = other_choices_words.replace(",", " ")

    for other_word in other_choices_words.split(" "):
        rating = 0.0
        b = float(qf.get_word_rating(other_word)[2])

        if b == 0 :
            pass
        else:
            rating = b

        if rating > 2.9 and rating < 4.3 :
            ofocus_in_graphlets = [fw for fw in graphlet_words if fw == other_word]
            if len(ofocus_in_graphlets)>1:
                numOtherAnswerF += 1


    minConcShared = 6.0
    if len(intsect_all_q)>0:
        for iw in intsect_all_q:
            sc = float(qf.get_word_rating(iw)[2])

            if sc < minConcShared :
                minConcShared = sc

    if minConcShared == 6.0 :
        minConcShared = 0.0

    tag_level_features["numRepeatedFocus"] = numRepeatedFocus
    tag_level_features["numOtherAnswerF"] = numOtherAnswerF
    tag_level_features["minConcShared"] = minConcShared


    #BRIDGE FEATURES
    massMaxBridgeScore = 0.0
    massMinBridgeScore = 0.0

    bridge_intersect = intsect_q.intersection(intsect_a)
    if len(bridge_intersect)>0 :
        for bw in bridge_intersect:
            sc = float(qf.get_word_rating(bw)[2])

            if massMaxBridgeScore < sc :
                massMaxBridgeScore = sc

            elif massMinBridgeScore > sc :
                massMinBridgeScore = sc

    massDeltaBridgeScore = massMaxBridgeScore - massMinBridgeScore
    #
    tag_level_features["massMaxBridgeScores"] = massMaxBridgeScore
    tag_level_features["massMinBridgeScore"] = massMinBridgeScore
    tag_level_features["massDeltaBridgeScore"] = massDeltaBridgeScore


    all_f = {}
    all_f["question_id"] = q_id
    all_f["explation_id"] = exp_id
    all_f["tag_level"] = tag_level_features

    curr_graphlet = all_graphlets[exp_id]

    all_f["graphlet_level"] = curr_graphlet.get_nugget_features() #expl_graphlet.get_nugget_features()




    # DUMP FEATURES PER FILE
    with open("data/qae_sep/all_qae_features.txt", 'a+') as fp:
        fp.write(str(all_f)+"\n")

    return  str(all_f).replace("'","\\\'")

    # except:
    #     print("Question with Errors : ",str(all_f)+"\n")

def process_questions(q_id_and_str,con):
    q_id = q_id_and_str[0]
    q_str = q_id_and_str[1].strip().replace("'s","")
    question = qf.Question(q_str)
    question.process()

    scores = [str(s[-1]) for s in question.all_word_features()]

    update_query = "UPDATE explanation SET `focus` = '"+",".join(question.get_focus())+"', `scores` = '"+ ",".join(scores) +\
                   "', `e_pen_trees` = '"+ question.get_pentree()+ "' WHERE explanation.e_id='"+str(q_id)+"'"

    exp_update_query = "UPDATE explanation SET `focus` = '"+",".join(question.get_focus())+ "' WHERE explanation.e_id='"+str(q_id)+"'"

    print(exp_update_query)

    cur = con.cursor()

    cur.execute(update_query)

    con.commit()

def clean_tree(id,tree,con):
    tree_clean = ""

    if not tree:
        tree = "(S1 )"

    # if tree.find("((dep")>-1:
    # tree_clean = re.sub(r'\((\(.*\))\) ', r'\1 ', tree).strip()
    # tree_clean = re.sub(r'\((\(.*\))\)', r'\1', tree_clean)
    # tree_clean = re.sub(r' (\(.*\))\)', r' \1', tree_clean)

    tree_clean = tree.replace("((dep ","(dep (")
    # tree_clean = tree_clean.replace(" (dep ", " (dep (")

    tree_clean = tree_clean.replace("()", "").strip()
    tree_clean = tree_clean.replace(" ( )", "").strip()
    tree_clean = re.sub(r'([(]S1$)', r'\1 ', tree_clean)
    tree_clean = tree_clean.strip()+")"
    tree_clean = tree_clean.replace("(S1)", "(S1 )")
    print("\n", tree, "\t", tree_clean)

    # update_questions_query =  "UPDATE question SET `q_pen_trees` = '" +tree_clean + "' WHERE question.q_id='" +id+ "'"
    update_dev_query = "UPDATE dev_question SET `q_pen_trees` = '" + tree_clean + "' WHERE dev_question.q_id='" + id + "'"
    # update_query = "UPDATE explanation SET `e_pen_trees` = '" +tree_clean + "' WHERE explanation.e_id='" +id+ "'"

    cur = con.cursor()

    # cur.execute(update_query)
    cur.execute(update_dev_query)
    con.commit()

    return tree_clean


if __name__ == "__main__":

    # if len(sys.argv) == 0:
    #     sent_id = sys.argv[0]
    #     sentence_aggregation.display_graphlets(sent_id,graphlets_folder)
    # else :
    #     sentence_aggregation.display_graphlets(graphlets_folder)

    # load_questions()

   # generate_qae_features()


    # questions_full = []  # load_questions()
    #
    # with open('data/all_train_qae.csv') as f:
    #     reader = list(csv.reader(f, delimiter='\t'))
    #
    #     i=0
    #     for row in reader:
    #         if i== 0:
    #             i=i+1
    #             continue
    #         questions_full.append(row)
    #
    # print("\n\n Am done with that Num Queries : ", len(questions_full[1:]),"\n\n")

    # generate_qae_features(questions_full)
    # stopwords = pu.get_stop_words()
    #
    # all_graphlets = sa.load_Graphlets(graphlets_folder)

    #FETCH AND PREOCESS THE QUESTIONS [for focus and pentrees]
    #
    # con = lite.connect('qae.sqlite')
    #
    # with con:
    #     cur = con.cursor()
    #
    #     start = time.time()
        # gc.collect()
        #
        # # sql_string = "select question.q_id, question.q_string from question"
        # sql_string = "select exp.e_id, exp.e_pen_trees from explanation as exp"
        #
        # cur.execute(sql_string)
        # data = cur.fetchall()

        # curr_list = []
        # for d in data:
        #     curr_list.append(list(d))
        #
        # first = 0
        # num_cpus = 6

        # for key,expl_graphlet in all_graphlets.items():
        #
        #     graphlet_words = list(
        #     set([w.strip().replace("'s","") for w in expl_graphlet.__str__().split(",") if w.strip() not in stopwords]))
        #
        #     question = qf.Question(expl_graphlet.raw_sentence)
        #     question.process()
        #
        #     focus_db = question.get_focus()
        #     if not focus_db:
        #         focus_db=[""]
            # else:
            #     focus_db=focus_db.split(",")

        #     focus_f = list(set(focus_db+graphlet_words))
        #
        #     update_query = "UPDATE explanation SET `focus` = '" + ",".join(
        #         focus_f).strip(",") + "' WHERE explanation.e_id='" + key + "'"
        #
        #     print("\n\n",update_query)
        #
        #     cur.execute(update_query)
        #
        #     con.commit()
        #
        # end = time.time()
        # print("\n\n Total Time : ", end - start)

    con = lite.connect('qae_sqlite/qae.sqlite')

    with con:
        cur = con.cursor()

        start = time.time()
    gc.collect()

    # sql_string = "select question.q_id, question.q_string from question"
    # sql_string = "select exp.e_id, exp.e_pen_trees from explanation as exp"
    # q_sql_string = "select q.q_id, q.q_pen_trees from question as q ;"
    dev_sql_string = "select dq.q_id, dq.q_pen_trees from dev_question as dq ;"

    cur.execute(dev_sql_string)
    data = cur.fetchall()

    curr_list = []
    for d in data:
        curr_list.append(list(d))


    for entry in curr_list:
        key = entry[0]
        tree = entry[1]

        tree_clean = clean_tree(key,tree,con)





    # PARALLEL PROCESS THE FEATURES
    # con = lite.connect('qae.sqlite')
    #
    # with con:
    #     cur = con.cursor()
    #
    #     print(cur)
    #
    #     limit_value = 26000 #mp.cpu_count() - 1
    #     offset_value = 260001
    #
    #     start = time.time()
    #
    #     gc.collect()
    #
    #     sql_string = "select j.Q_ID, j.E_ID, q.Q_String, q.Answer, q.Other_Choices, e.explanation from Q_E_Join as j " \
    #                  "inner join question as q on (q.Q_ID = j.Q_ID) inner join explanation as e on (e.E_ID = j.E_ID) "
    #     sql_string += "LIMIT " + str(limit_value) + " OFFSET " + str(offset_value)
    #
    #     cur.execute(sql_string)
    #     data = cur.fetchall()
    #
    #     curr_list = []
    #     for d in data:
    #         curr_list.append(list(d))
    #
    #     start = 24519
    #     num_cpus = 6
    #
    #     while start < 26000 :
    #         # if start < 5217:
    #         #     start = start + num_cpus
    #         #     continue
    #
    #         last = start + num_cpus
    #         pool = Pool(processes=num_cpus,maxtasksperchild=1000)
    #
    #         time.sleep(1)
    #
    #         pool.map(parallel_generate_qae_features, data[start:last])
    #
    #         offset_value += limit_value
    #
    #         time.sleep(2)
    #         pool.close()
    #         start += num_cpus
    #
    #     end = time.time()
    #     print("\n\n Total Time : ",end - start)



    # PROCESS THE FEATURES

    # all_graphlets = sa.load_Graphlets(graphlets_folder)
    # con = lite.connect('qae_sqlite/qae.sqlite')
    # curr_list = []
    #
    # with con:
    #     cur = con.cursor()
    #     sql_string = "select j.Q_ID, j.E_ID, q.q_string, q.answer, q.focus, q.other_choices, e.focus from Q_E_Join as j " \
    #                          "inner join question as q on (q.q_id = j.q_id) inner join explanation as e on (e.e_id = j.e_id) " \
    #                  " ORDER BY j.Q_ID ASC, j.E_ID ASC "
    #     sql_string += "LIMIT 100000 OFFSET 0"
    #
    #
    #     cur.execute(sql_string)
    #     data = cur.fetchall()
    #
    #     for d in data:
    #         # print(d)
    #         curr_list.append(list(d))
    # con.close()
    #
    # i = 14447
    # num_cpus = 6
    #
    # start = time.time()
    #
    # while i < 100000 :
    #     last = i+num_cpus
    #
    #     pool = Pool(processes=num_cpus, maxtasksperchild=1)
    #
    #     time.sleep(1)
    #
    #     pool.map(parallel_generate_qae_features, data[i:last])
    #
    #     time.sleep(1)
    #
    #     pool.close()
    #     pool.join()
    #
    #     if i%10 == 0:
    #         print("Now at : ", i)
    #
    #     i+=num_cpus
    #
    # end = time.time()
    # print("\n\n Total Time : ", end - start)
