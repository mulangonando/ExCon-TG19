#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 10:23:57 2018

@author: mulang

Handle Most nltk stuff
"""
from pntl.tools import Annotator
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import re

acc_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ','NN', 'NNS', 'NNP', 'NNPS', 'WP', 'PRP', 'PRP$', 'DT']
def get_stop_words():
    stop_words = ["!!", "?!", "??", "!?", "`", "``", "''", "-lrb-", "-rrb-", "-lsb-", "-rsb-", ",", ".", ":", ";", "\"",
                  "'", "?", "<", ">", "{", "}", "[", "]", "+", "-", "(", ")", "&", "%", "$", "@", "!", "^", "#", "*",
                  "..", "...", "'ll", "'s", "'m", "a", "about", "above", "after", "again", "against", "all", "am", "an",
                  "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
                  "between", "both", "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do",
                  "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
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
    return stop_words

def is_noun(tag):
    if tag.startswith('NN'):
        return True
    else :
        return tag in ['NN', 'NNS', 'NNP', 'NNPS', 'WP', 'PRP', 'PRP$', 'DT']

def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None



def extract_annotations(query):
    query = query.replace('?', '').strip()
    query = query.replace('/', ' / ')
    query = query.replace(' - ', ' ')
    query = query.replace('-', '')
    query = query.lower()

    annotator = Annotator()
    annotations = annotator.get_annoations(query.strip().split(" "), dep_parse=True)

    srls = annotations['srl']
    poss = annotations['pos']
    ners = annotations['ner']
    chunks = annotations['chunk']
    dep_parse = annotations['dep_parse']
    syntax_tree = annotations['syntax_tree']

    # print(syntax_tree)

    bag_of_words = []
    pos_seq = []
    ner_seq = []
    chunk_seq = []

    for pos in poss:
        bag_of_words.append(pos[0])
        pos_seq.append(pos[1])

    for ner in ners:
        ner_seq.append(ner[1])

    for chunk in chunks:
        chunk_seq.append(chunk[1])

    dep_seq = []
    dep_ref = {}

    for dep in dep_parse.decode("utf-8").split('\n'):
        # print(dep)
        positions = []
        try:
            dep_rel = str(dep)[:dep.index("(")].strip()
            pair = dep[dep.index('(') + 1:dep.index(')')]
            word1, word2 = pair.split(',')

            word1_num = word1.split('-')
            word2_num = word2.split('-')

            positions.append(dep_rel)
            positions.append(re.sub('[^0-9]','', word1_num[1].strip()))
            positions.append(re.sub('[^0-9]','', word2_num[1].strip()))
            positions.append(word1_num[0].strip())
            positions.append(word2_num[0].strip())

            dep_seq.append(positions)

            dep_ref[str(word1_num[1])+","+str(word2_num[1])] = dep_rel

        except ValueError:
            pass

    return (bag_of_words, pos_seq, ner_seq, chunk_seq,dep_ref,dep_seq, srls, syntax_tree)


def list_extractor(dep_seq, bag_of_words,pos_seq):

    # Init Lemmatizer
    wlzer = WordNetLemmatizer()

    focus = []
    list = []
    list_idx = []
    list_head = None

    # EXtract Lists first
    # Traverse the dependency tree
    anchor_from = []
    anchor_to = []
    preconj = []
    begin_list = False
    conj_and_or = []
    conj_idx = []

    # print(dep_seq)

    for dep in dep_seq:
        # print(dep)
        if anchor_from and anchor_to and anchor_from[0] == anchor_to[0]:

            pos1 = penn_to_wn(pos_seq[anchor_to[0] - 1])
            pos2 = penn_to_wn(pos_seq[anchor_from[1] - 1])
            if not pos1:
                pos1 = 'n'
            if not pos2:
                pos2 = 'n'

            focus.append(wlzer.lemmatize(bag_of_words[anchor_to[0] - 1],pos1))
            list.append(wlzer.lemmatize(bag_of_words[anchor_from[1] - 1],pos2))
            list.append(wlzer.lemmatize(bag_of_words[anchor_to[1] - 1],pos1))

            anchor_to = []
            anchor_from = []
        # try:
        if (dep[0] == 'prep_from'):
            anchor_from = [int(dep[1]), int(dep[2])]
        elif (dep[0] == 'prep_to'):
            anchor_to = [int(dep[1]), int(dep[2])]
        elif (dep[0] == 'preconj'):
            preconj = [int(dep[1]), int(dep[2])]

        elif (dep[0] == 'conj_and' or dep[0] == 'conj_or' or dep[0] == 'conj_nor' or dep[0] == 'appos'):
            if len(preconj) > 0:  # and preconj[0] == dep[]
                list.append(wlzer.lemmatize(bag_of_words[preconj[0] - 1],penn_to_wn(pos_seq[preconj[0] - 1])))
                list_idx.append(preconj[0] - 1)

                if (int(dep[1]) - 1) not in list_idx:
                    list.append(wlzer.lemmatize(bag_of_words[int(dep[1]) - 1].strip(0),penn_to_wn(pos_seq[int(dep[1]) - 1])))
                    list_idx.append(int(dep[1]) - 1)

                if (int(dep[2]) - 1) not in list_idx:

                    if int(dep[2]) - int(dep[1]) > 2 :
                        comp_noun = ""
                        for i in range(int(dep[1])+1,int(dep[2])):

                            comp_noun += " "+wlzer.lemmatize(bag_of_words[i],penn_to_wn(pos_seq[i]))
                            list_idx.append(i)
                        list.append(comp_noun.strip(" "))

                    else:
                        list.append(wlzer.lemmatize(bag_of_words[int(dep[2]) - 1],penn_to_wn(pos_seq[int(dep[2]) - 1])))
                        list_idx.append(int(dep[2]) - 1)

            else:
                pos1 = penn_to_wn(pos_seq[int(dep[1]) - 1])
                pos2 = penn_to_wn(pos_seq[int(dep[2]) - 1])

                if begin_list == False:

                    if not pos1:

                        pos = penn_to_wn(pos_seq[int(dep[1])-2])

                    if not pos1:
                        conj_and_or.append(wlzer.lemmatize(dep[3]))
                    else:
                        conj_and_or.append(wlzer.lemmatize(dep[3],pos1))
                        conj_idx.append(int(dep[1])-1)
                    begin_list = True

                    # if dep[0] == 'conj_or':
                    list_head = int(dep[1])-2

                elif int(dep[1])-1 < conj_idx[0]:
                    list_head = int(dep[1])-1

                if not pos2:
                    conj_and_or.append(wlzer.lemmatize(dep[4]))
                    conj_idx.append(int(dep[2]) - 1)
                else:
                    conj_and_or.append(wlzer.lemmatize(dep[4],penn_to_wn(pos_seq[int(dep[2])-1])))
                    conj_idx.append(int(dep[2]) - 1)
        # except:
        #     pass

    if list_idx and not list_head and not preconj:
        list_head = list_idx[0]
        list_idx = list_idx[1:]
        list = list[1:]

    list.extend(conj_and_or)
    list_idx.extend(conj_idx)

    return (list_head,list_idx,list, focus)


def PTB_generator(syntax_tree):

    print("Trying something")

