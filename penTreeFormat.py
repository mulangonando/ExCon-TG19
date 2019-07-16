#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 10:23:57 2018

@author: mulang

Handle Most nltk stuff
"""
from pntl.tools import Annotator
import nltk
from collections import deque as DQ
from collections import OrderedDict
from nltk.tree import ParentedTree
from nltk import Tree
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import string
import  parse_utils as pu

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



class NoVPException:
    def __init__(self): pass


# Makes the nltk tree from a string passed to it.
def maketree(tree_string):
    t = nltk.ParentedTree.fromstring(tree_string)
    return t


def getroot(subtree):
    crt = subtree
    while crt.parent() != None:
        crt = crt.parent()
    return crt


def lowest_common_subtree_phrases(t, word_list):
    lcst = lowest_common_subtree(t, word_list)
    return get_phrases(lcst)


def get_phrases(tree):
    """ This gets all of the phrase labels from the tree. """
    phrases = []

    def recurse(t):
        try:
            phrases.append(t.label())
        except AttributeError:
            return
        for child in t:
            recurse(child)

    recurse(tree)
    return phrases


def find_subtree_phrases(t, phrases):
    """ This is the best function here. """
    subtrees = []

    def recurse(t, phrases):
        try:
            t.label()
        except AttributeError:
            return
        if t.label() in phrases:
            subtrees.append(t)
        for child in t:
            recurse(child, phrases)

    recurse(t, phrases)
    return subtrees


def get_nearest_clause(tree, start, end=None):
    clauses = ['S', 'SBAR', 'SQ', 'SBARQ']

    if not end:
        subtree = getsmallestsubtrees(tree)[start]
    else:
        subtrees = getsmallestsubtrees(tree)

        try:
            subtree, subtree2 = subtrees[start], subtrees[end]
        except Exception:
            return None

    crt = subtree
    while (not crt is tree) and (not crt.label() in clauses) and ((not end) or not subtree2 in tree):
        crt = crt.parent()
    return crt


# To test if subtree1 dominates subtree2, we go up the tree from subtree2 until we either reach:
# the root, in which case we return false; or subtree1, in which case we return true.
def dominates(t, subtree1, subtree2):
    # The root dominates everything.
    if subtree1 == t.root(): return True

    crt = subtree2
    try:
        while crt != t.root() and crt.parent() != None:
            if crt == subtree1:
                return True
            crt = crt.parent()
    except AttributeError:
        return False
    return False


# Implementing the notion of C-Command; it might help.
# If A dominates B or B dominates A, there is no c-command.
# If A's first branching parent dominates B, then we have c-command.
def ccommands(t, subtree1, subtree2):
    if dominates(t, subtree1, subtree2) or dominates(t, subtree2, subtree1): return False
    if subtree1 is subtree2: return False

    crt = subtree1
    while len(crt) == 1 or (len(crt) == 2 and hasadverb(
            crt)):  # TODO: Changed it to get parent if there is only the word plus an adverb.
        crt = crt.parent()

    return dominates(t, crt, subtree2)


def hasadverb(subtree):
    for child in subtree:
        if not type(child) is str:
            if child.label() == 'RB':
                return True
    return False


def generate_local_structure_from_subtree(t, subtree):
    SENTENCE_PHRASES = ['S', 'SBAR', 'SQ', 'SBARQ', 'SINV']

    crt = subtree
    while not crt is t and not crt.label() in SENTENCE_PHRASES:
        crt = crt.parent()

    return crt


def has_phrases_between_trees(subtree1, subtree2, phrases):
    crt_phrase = subtree2

    while crt_phrase != subtree1 and crt_phrase.parent() != None:
        crt_phrase = crt_phrase.parent()
        if crt_phrase.label() in phrases:
            return True
        elif 'PRD' in phrases and crt_phrase.label().endswith('PRD'):
            return True

    return False


# Goes through the tree and gets the tuple indexes for each word in the tree,
# thus excluding the positions of the pos tags and phrase markers.
# This allows us to map the words by index in their list to their location
# in the tree.
#
# i.e.
# t = maketree(word_list's_tree)
# word_tree_positions = getwordtreepositions(t)
# word_list[4] <==> word_tree_positions[3]
#
# IMPORTANT - if the word_list contains the word 'ROOT' it is not mapped to, so we subtract by 1.
def getwordtreepositions(t):
    tree_pos_list = []
    for pos in t.treepositions():
        if isinstance(t[pos], str):
            tree_pos_list.append(pos)
    return tree_pos_list


# This will allow us to use the trees that correspond to the words, i.e. (VBZ is) instead of just 'is'
def getsmallestsubtrees(t):
    return [subtree for subtree in t.subtrees(lambda t: t.height() == 2)]


def pos_word_tuples(t):
    return [(subtree.label(), subtree[0]) for subtree in t.subtrees(lambda t: t.height() == 2)]


def get_smallest_subtree_positions(t, subtree_list=None):
    subtree_positions = []
    if not subtree_list:
        for subtree in t.subtrees(lambda t: t.height() == 2):
            subtree_positions.append(subtree.treeposition())
    else:
        for subtree in subtree_list:
            subtree_positions.append(subtree.treeposition())
    return subtree_positions


def lowest_common_subtree(t, word_list):
    positions = get_smallest_subtree_positions(t)

    head_idx = 0
    head = word_list[head_idx]
    tree_words = t.leaves()
    for i in range(0, len(tree_words)):
        if tree_words[i] == head:
            head_idx = i

    subtree = t[positions[head_idx]]
    while subtree.parent() != None:
        broke = False
        for word in word_list:
            if not word in subtree.leaves():
                subtree = subtree.parent()
                broke = True
                break
        if broke: continue
        while subtree.label() != 'VP' and len(subtree.parent().leaves()) == len(word_list):
            subtree = subtree.parent()
        return subtree
    return subtree


# This sequentially creates trees for all possible combinations of a VP and its children
def get_linear_phrase_combinations(t, phrase):
    vp_combos = []
    for position in phrase_positions_in_tree(t, phrase):
        vp_combos.append(t[position])

        vp = vp_combos[-1].copy(deep=True)
        vp_copy = vp.copy(deep=True)
        for child in reversed(vp):
            vp_copy.remove(child)
            if len(vp_copy):
                vp_combos.append(vp_copy.copy(deep=True))

    return vp_combos


def phrase_positions_in_tree(t, phrase):
    subtree_vp_positions = []

    compare = lambda t: t.label() == phrase
    if phrase == 'predicative':
        compare = lambda t: t.label().endswith('PRD')

    for subtree in t.subtrees(compare):
        subtree_vp_positions.append(subtree.treeposition())

    return subtree_vp_positions


def get_phrase_length(t):
    get_phrase_length.length = 0

    def recurse(tree):
        if type(tree) is nltk.ParentedTree:
            for child in tree:
                recurse(child)
        else:
            get_phrase_length.length += 1

    recurse(t)
    return get_phrase_length.length


def get_nearest_phrase(t, idx, phrases):
    positions = get_smallest_subtree_positions(t)
    try:
        crt_node = t[positions[idx - 1]]
    except IndexError:
        print('IndexError Encountered in get nearest phrase')
        return None

    while not crt_node.label() in phrases:

        if crt_node.parent() == None:
            return crt_node

        crt_node = crt_node.parent()

    return crt_node


def get_nearest_vp(t, idx):
    """DEPRECATED"""
    positions = get_smallest_subtree_positions(t)
    crt_node = t[positions[idx - 1]]

    while crt_node.label() != 'VP' and crt_node.label() != 'SINV':  # TODO: maybe change this to just VP!

        if crt_node.parent() == None:
            print('WARNING - NO VP IN THIS SENTENCE!')
            return crt_node

        crt_node = crt_node.parent()

    return crt_node


def get_nearest_vp_exceptional(t, idx, trigger, sentnum):
    """Returns the start and end indexes of the VP in the sentence."""
    vps = []

    def find_vps_recursive(tree):  # Need to save indexes of the VPs
        for child in tree:
            if type(child) != str and type(child) != unicode:
                if child.label() == 'VP':
                    vps.append(child)
                find_vps_recursive(child)

    find_vps_recursive(t)

    if len(vps) >= 1:
        trig_idx = getsmallestsubtrees(t)[idx].treeposition()
        to_remove = []
        for vp in vps:
            if sentnum == trigger.sentnum and vp.treeposition() >= trig_idx[:-1]:  # Don't include the last 0
                to_remove.append(vp)  # Get rid of VPs that include the trigger

            # elif not True in map(is_noun, [node.label() for node in getsmallestsubtrees(vp.parent())]):
            #     to_remove.append(vp)

        for badvp in to_remove:
            vps.remove(badvp)

        if len(vps) == 0:
            raise NoVPException

        ret = t[max([vp.treeposition() for vp in vps])]
        retleaves = ret.leaves()

        start, end, cursor = None, None, 0
        for i, word in enumerate(t.leaves()):
            if retleaves[cursor] == word:
                if start == None:
                    start = i
                    end = i
                else:
                    end = i
                cursor += 1
                if cursor == len(retleaves):
                    break
            else:
                if start and end and end - start != len(retleaves):
                    start, end = None, None
                elif start and end:
                    break

        return retleaves, start + 1, end + 2  # Return the right-most VP, increment because of ROOT

    if len(vps) == 0:
        raise NoVPException


def get_closest_constituent(t, word_list):
    head_idx = 0
    head = word_list[head_idx]
    tree_words = t.leaves()
    for i in range(0, len(tree_words)):
        if tree_words[i] == head:
            try:
                if tree_words[i + 1] == word_list[1]:
                    head_idx = i
                    break
            except IndexError:
                head_idx = i
                break

    positions = getsmallestsubtreepositions(t)
    crt_node = t[positions[head_idx - 1]]

    while not contain_eachother(crt_node.leaves(), word_list):
        if crt_node.parent() == None:
            return crt_node

        crt_node = crt_node.parent()

    return crt_node


def contain_eachother(lst, check_lst):
    if len(lst) < len(check_lst): return False
    for w in check_lst:
        if w not in lst:
            return False
    return True


# This is just to print the sentence.
def printfrompositions(t, tree_pos_list):
    for pos in tree_pos_list:
        print(t[pos], '\n')


def traverse_specific(t):
    try:
        t.label()
    except AttributeError:
        return
    else:

        if t.height() == 2:  # child nodes
            # print(t.parent())
            return

        for child in t:
            traverse_specific(child)


def traverse_annotation(tree, annotation, count=0):
    if tree is None:
        print("")
    elif isinstance(tree, ParentedTree):

        print("")
    else:
        # It's a leaf Build the array for this tree
        value_pair = {count, tree.label()}
        entry = [value_pair]
        count += 1

        hierarchy_count = 0
        curr = tree.parent()

        while curr is not None:
            value_pair = {hierarchy_count, curr.label()}
            ++hierarchy_count

        annotation.append(entry)


# Gets the tree and it's Parent [Positions]
def tree_parent_pairs(tree, pos):
    print(tree.treeposition())
    return [tree.parent().treeposition(), tree.treeposition(), pos]


def clean_string(s, l):
    '''
    Cleans String to Add

    :param s:
    :param l:
    :return:
    '''
    for w in l:

        if len(s.split()) > len(w.split()) and len(w.split()) > 1 and w.strip() in s.strip():
            s = s.replace(w, "")
            # if s.strip(' ').split(' ')[-2] == w.strip(' ').split(' ')[-2]:
            #     s = s.strip(' ')[:-(len(w))]

        elif s == w:
            s = ''
            break

        else:
            j = 0

            if s.strip() in w:
                w = w.replace(s.strip(), "")

            '''Thi while loop should '''
            while j <= len(w.strip(' ').split(' ')) and len(w) > 1 and w.strip() in s.strip():

                if w.strip(' ').split(' ')[-j] == s.strip(' ').split(' ')[-j]:
                    s = s.strip(' ')[:(len(w.strip().split()[-j]))]
                j += 1
    if len(s.strip(' ').split(' ')) == 1:
        if not l:
            l.append(s)
        s = ''
    return s, l


def sameClause(left, par):
    '''

    Function to check if the left sibling is still in the same clause [Avoid going left till root]

    :param left: the left sibling to check if still in the same clause
    :param par: the parent of current clause
    :return: true or false
    '''
    phrase_labels_list = ['ROOT', 'SQ', 'SBARQ', 'SBAR', 'CC', 'SINV', 'S', 'WHNP', 'WHADVP', 'WHPP', 'S1']
    if left.label() in phrase_labels_list:
        return False
    left_par = left.parent()

    while type(left_par) == ParentedTree:
        if left_par == par:
            return True

        left_par = left_par.parent()
    return True


def remove_redundants(iphrases):
    '''

    :param iphrases: list of the given phrases
    :return:
    '''
    # iphrases = list(set(iphrases))
    iphrases = list(OrderedDict.fromkeys(iphrases))

    return iphrases


def traverse_queue(tree, queue, rel_clauses=[], last_verb=False):
    attachable_left = ['WHADVP', 'WHNP']
    # ITERATE THROUGH THE STACK
    # print(len(queue), type(queue))
    curr = None
    parent = None
    skip_parent = False
    left_only = False
    while queue:

        if curr == None:  # or skip_parent==True :

            curr = queue.pop()
            if not queue:
                clause = " ".join(str(x) for x in curr.leaves())
                rel_clauses.append(clause.strip(" "))
            else:
                parent = queue.pop()
            # skip_parent==False
        else:
            curr = parent
            parent = queue.pop()

        left = curr.left_sibling()

        right = curr.right_sibling()

        in_words = []

        in_words.extend([str(x) for x in curr.leaves()])

        if len(in_words) == 1 and left is not None and (curr.label() == 'CC' or curr.label() in attachable_left):
            cc = in_words[0]
            in_words = [cc]
            left_words = []

            curr_p = curr

            j = 1
            while curr_p.parent() is not parent and curr_p.parent() is not None:

                curr_p = curr_p.parent()
                left_curr_p = curr_p.left_sibling()

                if left_curr_p is not None:
                    # BUILD THE LEFT SIDE (TAKES CARE OF THE VERBS FOR THE CC)
                    while left_curr_p.left_sibling() is not None:
                        left_curr_p = left_curr_p.left_sibling()

                    while left_curr_p is not curr_p:
                        extension = [str(x) for x in left_curr_p.leaves()]
                        left_words = extension + left_words
                        left_curr_p = left_curr_p.right_sibling()
                j += 1

            # Build the left
            while left is not None and left.left_sibling() is not None and sameClause(left.left_sibling(),
                                                                                      curr_p.parent()):
                left = left.left_sibling()

            while left is not None and left.right_sibling() is not curr:
                left_words.extend([str(x) for x in left.leaves()])
                left = left.right_sibling()

            if left is not None:
                left_words.extend([str(x) for x in left.leaves()])

            # Build the right
            while right is not None and right.right_sibling() is not None and right.right_sibling().label() != "VBN":
                in_words.extend([str(x) for x in right.leaves()])
                right = right.right_sibling()
            if right is not None:
                in_words.extend([str(x) for x in right.leaves()])

            in_string, rel_clauses = clean_string(' '.join(lw for lw in in_words), rel_clauses)
            left_string, rel_clauses = clean_string(' '.join(lw for lw in left_words), rel_clauses)

            # ADD THEM TO THE LIST (ALREADY CLEANED )
            rel_clauses.append(left_string.strip())

            if len(in_string) > 0:
                rel_clauses.append(in_string.strip())

        # THIS WAS THE ORIGINAL OF THIS FILE skip_parent=True
        elif len(in_words) > 1 and left is not None:
            if left.label() in attachable_left:
                left_words = [str(x) for x in left.leaves()]
                # if left.label() != 'WHNP':
                #     in_words = left_words+in_words

                if left_only:
                    left_string, rel_clauses = clean_string(' '.join(lw for lw in left_words), rel_clauses)
                    if len(left_string) > 0:
                        rel_clauses.append(left_string)  # Already clean

                elif len(left_words) == 1:
                    words = ' '.join(lw for lw in in_words)
                    words, rel_clauses = clean_string(words, rel_clauses)

                    if len(words) > 1:
                        rel_clauses.append(words.strip())

                elif not left_only and left is not None:
                    # words = ' '.join(str(x) for x in left.leaves())
                    # words = words +' '+' '.join(iw for iw in in_words)

                    words, rel_clauses = clean_string(words, rel_clauses)

                    if len(words) > 1:
                        rel_clauses.append(words)
                else:
                    words, rel_clauses = clean_string(' '.join(lw for lw in in_words), rel_clauses)
                    if len(words) > 1:
                        rel_clauses.append(words.strip())
            else:
                words, rel_clauses = clean_string(' '.join(lw for lw in in_words), rel_clauses)
                if len(words) > 1:
                    rel_clauses.append(words.strip())

        else:
            words, rel_clauses = clean_string(' '.join(lw for lw in in_words), rel_clauses)
            if len(words) > 1:
                rel_clauses.append(words.strip())

    return rel_clauses

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

    print(dep_seq)

    for dep in dep_seq:
        # print(dep)
        if anchor_from and anchor_to and anchor_from[0] == anchor_to[0]:

            pos1 = penn_to_wn(pos_seq[anchor_to[0] - 1])
            pos2 = penn_to_wn(pos_seq[anchor_from[1] - 1])
            org_pos1 = pos1
            org_pos2 = pos2
            if not pos1:
                pos1 = 'n'
            if not pos2:
                pos2 = 'n'

            focus.append(org_pos1 +" "+wlzer.lemmatize(bag_of_words[anchor_to[0] - 1],pos1))
            list.append(org_pos2 +" "+wlzer.lemmatize(bag_of_words[anchor_from[1] - 1],pos2))
            list.append(org_pos1 +" "+wlzer.lemmatize(bag_of_words[anchor_to[1] - 1],pos1))

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
                list.append(pos_seq[preconj[0] - 1]+" "+ wlzer.lemmatize(bag_of_words[preconj[0] - 1],penn_to_wn(pos_seq[preconj[0] - 1])))
                list_idx.append(preconj[0] - 1)

                if (int(dep[1]) - 1) not in list_idx:
                    list.append(pos_seq[dep[1] - 1]+" "+wlzer.lemmatize(bag_of_words[int(dep[1]) - 1].strip(0),penn_to_wn(pos_seq[int(dep[1]) - 1])))
                    list_idx.append(int(dep[1]) - 1)

                if (int(dep[2]) - 1) not in list_idx:

                    if int(dep[2]) - int(dep[1]) > 2 :
                        comp_noun = ""
                        for i in range(int(dep[1])+1,int(dep[2])):

                            comp_noun += " "+pos_seq[i]+" "+wlzer.lemmatize(bag_of_words[i],penn_to_wn(pos_seq[i]))
                            list_idx.append(i)
                        list.append(comp_noun.strip(" "))

                    else:
                        list.append(pos_seq[int(dep[2])-1]+" "+wlzer.lemmatize(bag_of_words[int(dep[2]) - 1],penn_to_wn(pos_seq[int(dep[2]) - 1])))
                        list_idx.append(int(dep[2]) - 1)

            else:
                pos1 = penn_to_wn(pos_seq[int(dep[1]) - 1])
                pos2 = penn_to_wn(pos_seq[int(dep[2]) - 1])

                if begin_list == False:

                    if not pos1:

                        pos = penn_to_wn(pos_seq[int(dep[1])-2])

                    if not pos1:
                        conj_and_or.append(pos_seq[int(dep[1])-1]+" "+wlzer.lemmatize(dep[3]))
                    else:
                        conj_and_or.append(pos_seq[int(dep[1])-1]+" "+wlzer.lemmatize(dep[3],pos1))
                        conj_idx.append(int(dep[1])-1)
                    begin_list = True

                    # if dep[0] == 'conj_or':

                    print(dep)
                    list_head = int(dep[1])-2

                elif int(dep[1])-1 < conj_idx[0]:
                    list_head = int(dep[1])-1

                if not pos2:
                    conj_and_or.append(pos_seq[int(dep[2])-1]+" "+wlzer.lemmatize(dep[4]))
                    conj_idx.append(int(dep[2]) - 1)
                else:
                    conj_and_or.append(pos_seq[int(dep[2])-1]+" "+wlzer.lemmatize(dep[4],penn_to_wn(pos_seq[int(dep[2])-1])))
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


def recurse_terminal(tree, string, clos_brackets):
    pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

    compound_names = ['NP','PP']

    if tree.label() in pos_tags:

        poss = tree.label()
        actual_word = "_".join([w for w in tree.leaves()])

        right_string = "(" + poss + " " + actual_word + ")"+clos_brackets

    elif tree.label():

        if not tree.left_sibling():
            left = tree.parent().left_sibling()
        else:
            left = tree.left_sibling()

        if not left :
            left_string = ""
            pass
        else:

            left_sibling = [t for t in left]

            if isinstance(left_sibling[0], str):
                left_sibling = [tree.left_sibling()]

            poss=""
            # PROCESS THE LEFT SIDE
            l = 0
            for ls in left_sibling:
                if not ls:
                    continue
                if l == 0:
                    # print(ls)
                    poss = ls.label()
                    actual_word = ls.leaves()[0]
                else:
                    poss = poss + "_" + ls.label()
                    actual_word = actual_word + "_" + ls.leaves()[0]
                l += 1

            left_string = ""

            if len(poss)>1 :
                left_string = " (" + poss + " " + actual_word + ")"

        poss = ""
        actual_word = ""

        curr_children = [c for c in tree]

        # label = curr_children[0].leaves()[0]
        right_children = [child for child in curr_children[1:]]

        # print(left_string,tree.label(),right_children)

        ## THIS TAKES CARE OF THE CASE WHEN THE PREPOSITION IS ALONE AND WHAT IT JOINS IS IN THE RIGHT SIBLING

        if not right_children and tree.right_sibling() :

            right_children = [tree.right_sibling()]
            # print(tree.label(), left_string, " \t ",right_children)

        for rc in right_children :

            if rc.label() not in pos_tags and rc.label() not in compound_names:
                string += ' (' + tree.label()
                clos_brackets += ')'

                for child in right_children:
                    recurse_terminal(child, string, clos_brackets)

            elif rc.label() not in pos_tags:
                string += " ("+rc.label()
                clos_brackets += ')'

                poss = ""
                actual_word=""

                r = 0
                for child in rc :
                    if r == 0:
                        poss = child.label()
                        actual_word = "_".join([w for w in child.leaves()])  # rc.leaves())
                    else:
                        poss = poss + "_" + child.label()
                        actual_word = actual_word + "_" + "_".join([w for w in child.leaves()])

                    r = r + 1

            else:
                poss = rc.label()
                actual_word = "_".join([w for w in rc.leaves()])  # rc.leaves())

                # print(" The the end : ",poss," "+actual_word)

            right_string = " (" + poss + " " + actual_word + ")"

            string += left_string + right_string

    else:
        string += ' ('+tree.label()
        clos_brackets += ')'

        for child in tree :
            recurse_terminal(child, string, clos_brackets)
    return string + clos_brackets

def buildQueuePenTree(tree, queue):
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

    # phrase_labels_list = ['ROOT', 'SQ', 'SBARQ', 'SBAR', 'CC', 'SINV', 'S', 'WHNP', 'WHADVP', 'WHPP', 'S1']

    prep_phrases = ['PP']
        # ["prep_with", "prep_through", "prep_from", "prep_to", "prep_as", "prep_into", "prep_by", "prep_in",
        #             "prep_such_as", "prep_because_of", "prep_over", "prep_on", "prep_between", "prepc_as", "prepc_by",
        #             "prepc_of", "prep_of", "prepc_with", "prepc_after", "prepc_for", "prepc_before", "prep_before",
        #             "prep_after", "prep_during", "prepc_during", "prepc_because_of", "prep_without", "prepc_without"]

    list_deps = ['prep_from','prep_to','preconj','conj_and','conj_or','conj_nor','appos']



    if type(tree) == ParentedTree:
        children = [t for t in tree]

        if tree.label() in prep_phrases:

            if not tree.left_sibling():
                left = tree.parent().left_sibling()
            else:
                left = tree.left_sibling()

            # if not left
            # left_sibling = [t for t in left]
            #
            # if isinstance(left_sibling[0],str):
            #     left_sibling = [tree.left_sibling()]
            #
            #
            # #PROCESS THE LEFT SIDE
            # l = 0
            # for ls in left_sibling:
            #
            #     if l == 0:
            #         poss = ls.label()
            #         actual_word = ls.leaves()[0]
            #     else:
            #         poss = poss + "_" + ls.label()
            #         actual_word = actual_word + "_" + ls.leaves()[0]
            #     l += 1
            #
            # left_string = "(" + poss + " " + actual_word + ")"

            #PROCESS THE RIGHT SIDE

            curr_children = [c for c in tree]

            label = curr_children[0].leaves()[0]

            right_children = [child for child in curr_children[1:]]

            poss = ""
            actual_word = ""

            # children = tree.children()

            r = 0
            # for rc in right_children:
            #
            #     if r == 0:
            #         poss = rc.label()
            #         actual_word = "_".join([w for w in rc.leaves()])  # rc.leaves())
            #     else:
            #         poss = poss + "_" + rc.label()
            #         actual_word = actual_word + "_" + "_".join([w for w in rc.leaves()])
            #
            #     r = r + 1
            #
            # right_string = "(" + poss + " " + actual_word + ")"
            #
            # string = "(prep_" + label + " " + left_string + " " + right_string + ")"

            string = "(prep_"+label
            close_brackets = ')'

            string = recurse_terminal(tree, string, close_brackets)

            queue.append(string)

        len_children = len(children)
        i = 1
        while i <= len_children:
            buildQueuePenTree(children[-i], queue)
            i = i + 1


        else:

            ## LIST
            out_pos = ['CC',',']

            children = [w for w in tree]
            if not children or isinstance(children[0],str):
                pass
            else:
                children_labels = [w.label() for w in tree]
                num_commas = children_labels.count(',')
                conjunct = children_labels.count('CC')

                if num_commas>1 :

                    list = [c for c in tree if c.label() not in out_pos]

                    list_parent = tree.label()
                    list_compound = ""

                    i = 0
                    for l in list :

                        # if i==0:
                        pos = l.label()
                        list_compound += "("+l.label()+" " +l.leaves()[0]+")"
                        # else:
                        #     poss += "_"+l.label()
                        #     list_compound += +l.leaves()[0]

                        i += 1

                    if not tree.left_sibling():
                        label_pos = ""
                        label = ""
                    else:
                        label_pos =  tree.left_sibling().label()
                        label = tree.left_sibling().leaves()[0]

                    string_list = "(("+label_pos+" "+label+") "+"("+list_parent+" "+list_compound+"))"

                    queue.append(string_list)

    return queue


def traverse_terminal_for_word(tree,str_out,close_brackets) :
    acc_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NN', 'NNS', 'NNP',
               'NNPS', 'WP', 'PRP', 'PRP$', 'DT']
    try:
        children_labels = [t.label() for t in tree]

        all_close = True

        for c in children_labels:
            if c not in acc_pos:
                all_close = False
                break

        if all_close:
            str_out += str_out + tree.label()
            l = 0
            for child in tree:

                if l == 0:
                    poss = child.label()
                    actual_word = child.leaves()[0]
                else:
                    poss = poss + "_" + child.label()
                    actual_word = actual_word + "_" + child.leaves()[0]
                l += 1

            terminal_str = "(" + poss + " " + actual_word + ")"

            str_out += " "+terminal_str

        else:
            str_out += " (" + tree.label()
            close_brackets += ")"

            if tree.label() in acc_pos:
                poss = tree.label()
                actual_word = tree.leaves()

                str_out += "("+poss+" "+actual_word+")"
            else:
                for child in tree :
                    traverse_terminal_for_word(child, str_out, close_brackets)

    except AttributeError:
        if tree.label() in acc_pos:
            poss = tree.label()
            actual_word = tree.leaves()

            str_out += "(" + poss + " " + actual_word[0] + ")"

        # print(tree)
        # str_out += str_out + tree.label()
        # l = 0
        # for child in tree:
        #
        #     if l == 0:
        #         poss = child.label()
        #         actual_word = child.leaves()[0]
        #     else:
        #         poss = poss + "_" + child.label()
        #         actual_word = actual_word + "_" + child.leaves()[0]
        #     l += 1
        #
        # terminal_str = "(" + poss + " " + actual_word + ")"

        # str_out += " " + terminal_str

    return str_out+close_brackets


def traverse_for_words(tree,word1,word2):

    result = ""
    # within_single_child = False

    try:

        correct_child = None

        for t in tree :
            if word1 in t.leaves() and word2 in t.leaves() : #and t.label() not in acc_pos :

                # within_single_child = True
                correct_child = t

        if correct_child :

            for t in tree :

                # This is the last Umbrella parent
                if word1 in t.leaves() and word2 in t.leaves():
                    return traverse_for_words(t,word1,word2)

        else:
            return traverse_terminal_for_word(tree,result,"")

    except AttributeError:
        return traverse_terminal_for_word(tree, result, "")

    return result



def getPenTreeBankSpecial(pase_str,dep_seq,bog,pos):
    '''
    :param pase_str: A sring of the parsetree from the stanford parser
    :return: A list of the subquestions
    '''

    synt_tree = Tree.fromstring(pase_str)

    parented_ParseTree = ParentedTree.convert(synt_tree)
    return_str = "(" + parented_ParseTree.label()+" "

    queue = DQ([])
    pen_queue = buildQueuePenTree(parented_ParseTree, queue)

    nugget_constructs = ['ccomp', 'xcomp', 'advcl', 'rcmod', 'vmod' 'nn',"prep_to"]#'nsubjpass', 'nsubj', 'dobj']

    extras = ""
    for dep in dep_seq:
        # print(dep)
        if dep[0] in nugget_constructs :
            a = int(dep[1])-1
            b = int(dep[2])-1

            word1 = bog[a]
            word2 = bog[b]

            # res = traverse_for_words(ParentedTree.convert(synt_tree),word1,word2)
            res = ""
            if word1 not in string.punctuation and word2 not in string.punctuation:
                if pos[b].find('V')>-1 or pos[a].find('V')>-1 :
                    res = "(dep " + pos[a] + "_" + pos[b] + " " + word1 + "_" + word2 + ")"
                else:
                    res = "(dep "+pos[b]+"_"+pos[a]+" "+word2+"_"+word1+")"

            if extras.find(res)==-1:
                extras += res+" "



    return_str += "(" + extras.strip() + ")"  # +" "+pen_queue+")"

    while pen_queue:
        return_str += " " + pen_queue.pop()

    # Handle Lists
    (list_head,list_idx,list, focus) = list_extractor(dep_seq, bog, pos)

    print(list_head,"\t",list_idx,"\t",list)

    # print("\n\n",list_head,list_idx,list)

    listed = False
    strlist = ""

    if list:
        if list_head :
            strlist = "(("+pos[list_head]+" "+bog[list_head]+") (NP"
        else:
            strlist = "((conj)"+" (NP"

        for li,lw in zip(list_idx,list) :
            if lw.strip() in return_str: #return_str.find(lw.strip())>-1:
                listed = True
                break
            else:
                strlist +="("+lw+")"

        strlist += "))"


    if not listed and list:
        return_str += " "+strlist+")"

    return return_str.replace(" () ",' ')


if __name__ == "__main__":

    # query = "Which tenant of New Sanno Hotel is the military branch of the Gary Holder-Winfield ?"
    # query = "WHo is the wife of Obama ?"
    # query = "Tennessee is a state located in the United States of America"
    # query = "Which Kingdom provides man,cows,goats, mokeys, and Giraffe"
    # query = "a pebble is a kind of small rock"
    # query = "Students can calculate speed by measuring distance with a meter stick and measuring time with a clock."
    # query = "Both hurricanes and tornadoes always ___."
    # query = "Over time, the ability to ship foods around the world has improved. Which is the most likely effect these shipping improvements have had on people?"
    # query = "Which invention will best help people travel quickly to far away places?"
    query = "a beach ball is flexible"
    (bag_of_words, pos_seq, ner_seq, chunk_seq, _,dep_seq, srls, syntax_tree) = pu.extract_annotations(query)#extract_annotations(annotations)

    synt_tree = Tree.fromstring(syntax_tree)

    synt_tree.pretty_print()

    queue = DQ([])
    res_queue = getPenTreeBankSpecial(ParentedTree.convert(syntax_tree),dep_seq,bag_of_words,pos_seq)

    print("\n\n", res_queue)

    # while res_queue:
    #
    #     print("\n\n",res_queue.pop())

    # print("\n\n\n")
    #
    # clause_queue = traverse_queue(ParentedTree.convert(synt_tree), res_queue)
    # clause_queue = remove_redundants(clause_queue)
    #
    # for c in clause_queue:  # phrases:
    #     if type(c) is not bool:
    #         print(c, "\n")

    # (of(NP(DT the)(NN
    # wife)), (NP(NN obama)))