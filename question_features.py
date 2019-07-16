#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 10 04:04:28 2019
@author: Mulang'
"""
import csv
from nltk.tree import ParentedTree
from nltk import Tree
import parse_utils as pu
from nltk.stem import WordNetLemmatizer
import penTreeFormat as ptf

# Let's first define a minimalistic list of STOP WORDS
# OBTAINED FROM THE Stanford CoreNLP Stopwords list

def get_word_rating(w):
    crt_ratings = {}

    with open("data/Concreteness_ratings_Brysbaert_et_al_BRM.csv") as f:
        ratings_list = []
        reader = csv.reader(f)
        for row in reader:
            ratings_list.append(row)

        i = 0
        while i < len(ratings_list):
            if i > 0:
                crt_ratings[ratings_list[i][0]] = ratings_list[i][1:]
            i = i + 1

    if w not in crt_ratings:
        return [0,0,0,0]

    return crt_ratings[w]


def traverse_tree(tree, type_sects):
    '''
    Take the parts closes to the needed leaf and extracts it.
    :param tree:
    :param queue:
    :return:
    '''
    phrase_labels_list = ['NNS', 'NN', 'NP']

    # print("\n\not Tree : ",tree)

    if type(tree) == ParentedTree:

        children = [t for t in tree]

        if tree.label() in phrase_labels_list:  # and tree.parent().label() in parents_list:
            parent = tree.parent()
            if parent.label() == "WHNP":

                # first ruleprint("Subs : ", subs)
                try:
                    left = tree.left_sibling()
                    if left.label() == "WHNP" or left.label() == "WP" or left.label() == "WDT":
                        type_phrase = " ".join(tree.leaves())
                        # if parent.right_sibling().label() == "PP":
                        type_phrase = type_phrase + " " + " ".join(parent.right_sibling().leaves())

                        # type_phrase = type_phrase + " " + " ".join(parent.right_sibling().leaves())
                        type_sects.append(type_phrase)

                except AttributeError:

                    type_phrase = " ".join(tree.leaves())
                    # if parent.right_sibling().label() == "PP":

                    if parent.right_sibling():
                        type_phrase = type_phrase + " " + " ".join(parent.right_sibling().leaves())

                    # type_phrase = type_phrase + " " + " ".join(parent.right_sibling().leaves())
                    type_sects.append(type_phrase)
                # Second Rule
                # elif (left.label() =='SQ'or left.label()=='S') and  (left.left_sibling().label() == "WHNP"):
                #     print("Got Second")
                #     is_found = False
                #     for child in left:
                #         if child.label() == 'VBZ':
                #             is_found = True
                #             break
                #     if is_found:
                #         type_phrase = tree.leaves()
                #         type_sects.append(type_phrase)

            elif (parent.label() == 'NP' and parent.left_sibling() and parent.left_sibling().label() == 'VBZ'):
                type_phrase = tree.leaves()
                type_sects.append(type_phrase)

            # Rule 3
            elif parent.label() == "NP" and parent.right_sibling() and parent.right_sibling().label() == "SBAR":
                right = parent.right_sibling()

                for child in right:
                    if child.label() == "WHNP":
                        for c in child:
                            if c.label() == "WDT":
                                type_sects.append(tree.leaves())
                            break
                    break

            # Rule 4
            elif parent.label() == "NP" and tree.right_sibling() and tree.right_sibling().label() == "PP" \
                    and parent.right_sibling() and parent.right_sibling().label() == "VP":

                type_sects.append(tree.leaves())

        len_children = len(children)
        i = 1
        while i <= len_children:
            traverse_tree(children[-i], type_sects)
            i = i + 1

    return type_sects


def get_types(pase_str):
    '''
    :param pase_str: A sring of the parsetree from the stanford parser
    :return: A list of the subquestions
    '''

    synt_tree = Tree.fromstring(pase_str)
    # synt_tree.pretty_print()

    type_list = []
    res_queue = traverse_tree(ParentedTree.convert(synt_tree), type_list)

    return res_queue


class Q_Word:
    '''
        The class represents a word in the question
        Modeled as word and the features
    '''

    def __init__(self, str, pos, conc=None, abstract=None, tag=None, score=None):
        self.word_str = str
        self.pos = pos
        self.conc = conc
        self.abstract_level = abstract
        self.tag = tag
        self.score = score
        self.next = None
        self.prev = None

    def word_features(self):
        w_feat = [self.conc, self.abstract_level, self.tag, self.score]
        return w_feat

    def __str__(self):
        return self.word_str

    def whole_q_word(self):
        whole_qword = [self.word_str, self.pos, self.conc, self.abstract_level, self.tag, self.score]
        return whole_qword

    def set_conc(self, conc):
        self.conc = conc

    def set_tag(self, tag):
        self.tag = tag

    def set_tag(self, score):
        self.score = score

    def get_conc(self):
        return self.conc

    def get_tag(self):
        return self.tag

    def get_pos(self):
        return self.pos

    def get_score(self):
        return self.score


class Question:
    def __init__(self, raw):
        self.head = None
        self.tail = None
        self.raw = raw
        self.list = []
        self.focus = []
        self.stops = []
        self.type_list = []
        self.srls = []
        self.pentree = ""

    def get_pentree(self):
        return self.pentree

    def get_focus(self):
        return  self.focus

    def add(self, str, pos):
        '''
            Adding at the tail
            Traverse first
        '''
        q_word = Q_Word(str, pos)
        if self.head == None:
            self.head = q_word
            self.tail = q_word
        else:
            # curr = self.head
            # while (curr.next != None):
            #     curr = curr.next
            self.tail.next = q_word
            self.tail = q_word
        return self.tail

    def search(self, key):
        '''

        :param key: the word to be retrieved
        :return: the Q_Word
        '''
        p = self.head
        if p != None:
            while p.next != None:
                if (p.__str__() == key):
                    return p
                p = p.next
            if (p.__str__() == key):
                return p
        return None

    def get(self, idx):
        '''

        :param idx: the pos of the word to retrieve
        :return: the Q_Word at idx
        '''
        curr = self.head
        i = 1
        while (curr.next != None and i < idx):
            curr = curr.next
            i = i + 1

        if i == idx:
            return curr

        else:
            return None

    def remove(self, p):
        tmp = p.prev
        p.prev.next = p.next
        p.prev = tmp

    def __str__(self):
        s = ""
        p = self.head
        if p != None:
            while p.next != None:
                s += " " + p.__str__()
                p = p.next
            s += " " + p.__str__()
        return s

    def all_word_features(self):
        all_features = []
        p = self.head
        if p != None:
            while p.next != None:
                all_features.append(p.word_features())
                p = p.next
            all_features.append(p.word_features())

        return all_features

    def all_word_and_features(self):
        all_words_and_features = []
        p = self.head
        if p != None:
            while p.next != None:
                all_words_and_features.append(p.whole_q_word)
                p = p.next
            all_words_and_features.append(p.whole_q_word)

        return all_words_and_features

    def score_function(self, curr):
        if curr.word_str in self.list:
            curr.score = 21
            curr.tag = "List"
        elif curr.tag == "Focus":
            self.focus.append(curr.word_str)
            curr.score = 20

        elif curr.tag == "ST":
            curr.score = 0

        else:
            oldRange = 1  # (OldMax - OldMin)
            newRange = (10 - 2)
            oldMin = 0
            newMin = 2

            if curr.abstract_level == 'Abstract':
                oldRange = (3 - 0)
                oldMin = 0

            elif curr.abstract_level == 'Concrete':
                oldRange = (5 - 4.3)
                oldMin = 4.3

            oldValue = curr.conc
            if oldValue == None:
                oldValue = 0

            newValue = (((oldValue - oldMin) * newRange) / oldRange) + newMin

            if curr.abstract_level == "Concrete":
                curr.score = round(12 - newValue, 1)

            elif curr.tag == "ATYPE":
                curr.score = round(newValue + 1, 1)
            else:
                curr.score = round(newValue, 1)

    def process(self):
        '''
        Here we process the qustion: Actually get the features and save them
        :return: the head pointer.
        FOCUS WORD EXTRACTOR

        TEXT AGGREGATION GRAPHS

        '''

        (bag_of_words, pos_seq, ner_seq, chunk_seq,dep_ref, dep_seq, srls, syntax_tree) = pu.extract_annotations(self.raw)
        # print("Dependencies : \n\n ", dep_seq)
        self.srls = srls

        # Get the List and Focus Words

        (_,_, list, focus) = pu.list_extractor(dep_seq, bag_of_words,pos_seq)

        self.list.extend(list)
        self.focus.extend(focus)

        # Extract ANSWER TYPE
        type_list = get_types(syntax_tree)

        # Load the Concretenes dataset
        crt_ratings = {}

        with open("data/Concreteness_ratings_Brysbaert_et_al_BRM.csv") as f:
            ratings_list = []
            reader = csv.reader(f)
            for row in reader:
                ratings_list.append(row)

            i = 0
            while i < len(ratings_list):
                if i > 0:
                    crt_ratings[ratings_list[i][0]] = ratings_list[i][1:]
                i = i + 1

        ## Do we add it to the List or create data first

        for i in range(len(bag_of_words)):

            q_word = self.add(bag_of_words[i], pos_seq[i])

            if q_word.word_str in pu.get_stop_words():
                q_word.tag = "ST"
                q_word.conc = float(0.0)

            else:

                wordnet_lemmatizer = WordNetLemmatizer()

                w_pos = pu.penn_to_wn(q_word.get_pos())
                if not w_pos:
                    w_pos = 'n'

                word_lemma = wordnet_lemmatizer.lemmatize(q_word.word_str,w_pos)

                if word_lemma in crt_ratings:
                    q_word.conc = float(crt_ratings[word_lemma][1])
                else:
                    q_word.conc = float(0.0)

                # Abstraction Level
                if q_word.conc < 3.0:
                    q_word.abstract_level = 'Abstract'
                elif q_word.conc < 4.3:
                    q_word.abstract_level = 'Focus'
                    q_word.tag = 'Focus'
                elif q_word.conc > 4.2:
                    q_word.abstract_level = 'Concrete'

                # Stop Words, Examples
                if ner_seq[i] == 'B-LOC' or ner_seq[i] == 'I-LOC' or ner_seq[i] == 'E-LOC' or ner_seq[i] == 'MISC':
                    q_word.tag = 'Ex'
                # else:
                # 	q_word.abstract_level = 'Concrete'

                for typ in type_list:
                    # print(typ)
                    if q_word.word_str in typ:
                        q_word.tag = "ATYPE"
                        q_word.score = 1
                        break

        # SCORE
        max_score = 0
        curr = self.head

        while curr.next:
            self.score_function(curr)
            curr = curr.next

        self.score_function(curr)


        #GET Penn Tree Format
        self.pentree = ptf.getPenTreeBankSpecial(syntax_tree,dep_seq,bag_of_words,pos_seq)

if __name__ == "__main__":

    # query = "Which of these will most likely increase a plant population in a habitat?"
    # query = "What collective name is given to mammals,birds,amphibians, and reptiles ?"
    # query = "Who was neither president nor vice president?"
    query = "A meter stick is a common instrument for measuring length or distance"
    q = Question(query)
    q.process()

    print("The Question : ", q.__str__())
    print("List : ", q.list)

    for q_feat in q.all_word_features():
        # all_word_featuresfor f in q_feat:
        print("Featues : ", [f for f in q_feat])
