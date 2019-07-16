import csv
import parse_utils as pu
from nltk.stem import WordNetLemmatizer
import pickle as pkl
import os
from copy import copy

# Load the Concretenes dataset
# All id's stemmed
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

prep_phrases = ["prep_with", "prep_through", "prep_from", "prep_to", "prep_as", "prep_into", "prep_by", "prep_in",
                "prep_such_as", "prep_because_of", "prep_over", "prep_on", "prep_between", "prepc_as", "prepc_by",
                "prepc_of", "prep_of", "prepc_with", "prepc_after", "prepc_for", "prepc_before", "prep_before",
                "prep_after", "prep_during", "prepc_during", "prepc_because_of", "prep_without", "prepc_without"]

acc_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
            'NN', 'NNS', 'NNP', 'NNPS', 'NN', 'NNS', 'NNP', 'NNPS', 'WP', 'PRP', 'PRP$', 'DT', 'NN']


def remove_stops(wl,pos_seq,ner_seq):
    wlzer = WordNetLemmatizer()
    indexes = []
    nList = []
    i = 1
    for w,pos,ner in zip(wl,pos_seq,ner_seq):

        if w not in pu.get_stop_words() and ner=='O':

            if pu.penn_to_wn(pos):
                indexes.append(i)
                nList.append(wlzer.lemmatize(w,pu.penn_to_wn(pos)))
            else:
                indexes.append(i)
                nList.append(wlzer.lemmatize(w, 'n'))
        i = i+1

    out_list=[indexes]

    for nL in nList:
        out_list.append(nL)
    return out_list

def replace_NEs(word,ner):
    ner_map = {'S-PER':'person','B-PER':'person','I-PER':'person','E-PER':'person','S-LOC':'location','B-LOC':'location','I-LOC':'location',
               'E-LOC':'location','S-ORG':'organization','B-ORG':'organization','I-ORG':'organization','E-ORG':'organization'}
    if ner == 'O':
        return word
    else:
        return ner_map[ner]

class Graph(object):
    def __init__(self, size):
        self.adjMatrix = []
        for i in range(size+1):
            self.adjMatrix.append(["" for i in range(size+1)])
        self.size = size

    def addEdge(self, v1, v2, label):
        '''
            Ensure the first variable is the head of the edge for a directed Graphs
        :param v1: the first vertex
        :param v2: the second vertex
        :param label: the edge
        :return: nothing
        '''
        # if v1 == v2:
            # print("Same vertex %d and %d" % (v1, v2))
            # print("Indexes to be assigned : ",v1,v2)
        self.adjMatrix[v1][v2] = label

    def removeEdge(self, v1, v2):
        if self.adjMatrix[v1][v2] == "":
            # print("No edge between %d and %d" % (v1, v2))
            return
        self.adjMatrix[v1][v2] = ""

    def containsEdge(self, v1, v2):
        return True if len(self.adjMatrix[v1][v2]) > 0 else False

    def getEdge(self,v1,v2):
        return self.adjMatrix[v1][v2]

    def __len__(self):
        return self.size

    def toString(self):
        i=0
        print("--------------------------------------------------------\n")

        for row in self.adjMatrix:

            if i==0 :
                build_str = "\t"
            else:
                build_str = "v"+str(i)+"\t"

            j=0
            for val in row:
                if j==0:
                    j += 1
                    continue

                elif i==0:

                    build_str += "v" + str(j) + "\t"
                else:
                    build_str += val+"\t"

                j += 1

            print(build_str)

            i += 1
        print("--------------------------------------------------------\n")

# def main():
#     g = Graph(5)
#     g.addEdge(0, 1);
#     g.addEdge(0, 2);
#     g.addEdge(1, 2);
#     g.addEdge(2, 0);
#     g.addEdge(2, 3);
#
#     g.toString()


# if __name__ == '__main__':
#     main()

class Graphlet:

    def __init__(self, sent_id, raw_sentence,s_type):
        self.raw_sentence = raw_sentence
        self.sentence_id = sent_id
        self.s_type = s_type
        self.nuggets = {}
        self.edges = None
        self.nugget_features = {}
        self.srl = []

    def get_id(self):
        return self.sentence_id

    def get_nuggets(self):
        return self.nuggets

    def get_edges(self):
        return self.edges

    def get_type(self):
        return self.s_type

    def get_nugget_features(self):
        return self.nugget_features

    def incomming_link(self,others,v):
        for o in others:
            if self.edges.containsEdge(int(o[-1]),int(v[-1])):
                return self.edges.getEdge(int(o[-1]),int(v[-1]))
        return ""

    def generate_features(self):
        '''
            numNugF Number of nuggets that are entirely focus words
            numNugFS Number of nuggets that contain only focus words and shared words
            numNugFSO Number of nuggets that contain focus, shared, and other unmatched words
            numNugFO Number of nuggets that contain only focus words and other words
            numNugS Number of nuggets that contain only shared words
            numNugSO Number of nuggets that contain only shared words and other words
            numNugO Number of nuggets that contain only other words
            numDefinedFocus Number of nuggets containing only focus words with outgoing definition edge
            numDefinedShared Number of nuggets containing only shared words with outgoing definition edge
            numQLinksFocus Number of nuggets containing only focus words that have an incoming labeled link
            (e.g., definition, instrument, process, temporal)
            numQLinksShared Number of nuggets containing only shared words that have an incoming labeled link
            numNuggetMultiF Number of nuggets that contain more than one focus word
        :return:
        '''

        # DEFINITION PATTERNS
        # We will asume here that any row sentence that contains these constructs has all their
        # Focus and Shared words with edges
        def_patterns = ["is a type of","is a kind of","means","refers to","also known as", "also means"]


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

        numNugF = 0
        numNugFS = 0
        numNugFSO= 0
        numNugFO = 0
        numNugS = 0
        numNugSO = 0
        numNugO = 0
        numNugDefinedFocus = 0
        numDefinedShared = 0
        numQLinksFocus = 0
        numQLinksShared = 0
        numNuggetMultiF = 0

        all_shared_keys = []
        all_focus_keys = []

        for key,nugget in self.nuggets.items():

            unshared_list = []
            shared_list = []
            focus_list = []

            # Get the rest of the nuggets
            curr_nugget = set(nugget)
            rest_nuggets = []
            rest_keys = []

            for key_comp,nugget_comp in self.nuggets.items():
                if key != key_comp:
                    rest_keys.append(key)
                    rest_nuggets.extend(nugget_comp)

            #Only Shared Words
            if set(curr_nugget).issubset(rest_nuggets):
                numNugS += 1
                all_shared_keys.append(key)

                shared_list = nugget

                # Since all Shared; Now we want to Get if it has an in Edges
                in_edge = self.incomming_link(rest_keys, key)

                if len(in_edge) > 1:
                    numQLinksShared += 1


            else:
                unshared_list = list(set(curr_nugget) - set(rest_nuggets))

                shared_list = list(set(curr_nugget) - set(unshared_list))


                # Check if Shared Words have Definition
                if len(shared_list)>0 :
                    for defin in def_patterns :
                        if defin in self.raw_sentence:
                            numDefinedShared += 1
                            break


            len(nugget)
            allFocus = True
            for item in nugget:
                if len(item.split(" "))>1 or item not in crt_ratings :
                    allFocus = False
                    break
                elif  float(crt_ratings[item][1]) <3.0 or float(crt_ratings[item][1]) >4.2:
                    allFocus = False
                    break
                else:
                    focus_list.append(item)


            # Multiple Focus Words
            if len(focus_list)>1 :
                numNuggetMultiF += 1

            if len(focus_list)>0:
                    for defin in def_patterns :
                        if defin in self.raw_sentence:
                            numNugDefinedFocus += 1
                            break

            shared_plus_focus = shared_list + focus_list

            #Only Focus Words
            if allFocus:
                numNugF += 1
                all_focus_keys.append(key)

                #IF ALL FOCUS Then we check for incoming edges

                in_edge = self.incomming_link(rest_keys,key)

                if len(in_edge)>1:
                    numQLinksFocus += 1


            # Contains only Focus and Shared Words
            if set(shared_plus_focus) == set(nugget) and len(shared_list) > 0 and len(focus_list) > 0:
                numNugFS += 1

            #Contains only other words if no Focus and no Shared
            if (numNugS+numNugF) == 0:
                numNugO += 1

            #Contains Shared, focus and Others
            if len(shared_list)>0 and len(focus_list)>0 and len(unshared_list)>0:
                numNugFSO += 1

            #Contains Only Shared and others NO FOCUS
            if len(set(shared_list))< len(set(nugget)):
                numNugSO += 1

            #Contains
            if set(focus_list + unshared_list) == set(nugget) and len(focus_list) >0 and len(shared_list)==0:
                numNugFO += 1

            if len(set(unshared_list)) == len(set(nugget)) :
                numNugO += 1


        self.nugget_features['numNugF'] = numNugF
        self.nugget_features['numNugFS'] = numNugFS
        self.nugget_features['numNugFSO'] = numNugFSO
        self.nugget_features['numNugFO'] = numNugFO
        self.nugget_features['numNugS'] = numNugS
        self.nugget_features['numNugSO'] = numNugSO
        self.nugget_features['numNugO'] = numNugO
        self.nugget_features['numNugDefinedFocus'] = numNugDefinedFocus
        self.nugget_features['numDefinedShared'] = numDefinedShared
        self.nugget_features['numQLinksFocus'] = numQLinksFocus
        self.nugget_features['numQLinksShared'] = numQLinksShared
        self.nugget_features['numNuggetMultiF'] = numNuggetMultiF


    def __str__(self):
        s = ""
        for key,value in self.nuggets.items():
            if isinstance(value[0],list):
                s += " "+",".join(value[0])+","+",".join(value[1]) +","
            else:
                s += " "+",".join(value)+","

        s = s.strip(",")
        return s

    def generate_edges(self,dep_ref):
        #Generate links and connections
        self.edges = Graph(len(self.nuggets))

        edge_labels_map = {"prep_with": "instrument", "prep_through": "instrument", "prep_from": "process",
                           "prep_to": "process", "prep_as": "example", "prep_into": "process", "prep_by": "instrument",
                           "prep_such_as": "example", "prep_because_of": "process", "prepc_as": "example",
                           "prepc_by": "instrument", "prepc_with": "instrument", "prepc_after": "temporal",
                           "prepc_before": "temporal", "prep_before": "temporal", "prep_after": "temporal",
                           "prep_during": "temporal", "prepc_during": "temporal", "prepc_because_of": "process",
                           "prep_without": "contrast", "prepc_without": "contrast","amod":"example"}

        for i in range(1,self.nuggets.__len__()+1):
            ith = self.nuggets['v'+str(i)]

            self.edges.addEdge(i,i,"1")

            for j in range(i+1,self.nuggets.__len__()+1):
                jth = self.nuggets['v'+str(j)]

                ith_idxs = ith[0] #[str(pos) for pos in ith[0]]
                jth_idxs = jth[0] #[str(pos) for pos in jth[0]]

                key1 = "" + str(ith_idxs[0]) + "," + str(jth_idxs[0])
                key2 = "" + str(ith_idxs[0]) + "," + str(jth_idxs[-1])

                if len(jth_idxs) == 1 and (j-i ==1) :
                    # Assumed as Instrument
                    self.edges.addEdge(j, i, "Instrument")

                elif type(ith[1]) == list:
                    # Ith is a List
                    # Here  the only thing is to check the first index (The head)
                    if key1 in dep_ref and dep_ref[key1] in edge_labels_map:
                        self.edges.addEdge(i,j,edge_labels_map[dep_ref[key1]])

                    elif key1 in dep_ref :
                        self.edges.addEdge(i,j,"dep")
                    elif key2 in dep_ref and dep_ref[key2] in edge_labels_map:
                        self.edges.addEdge(i,j,edge_labels_map[dep_ref[key2]])
                    elif key2 in dep_ref :
                        self.edges.addEdge(i,j,"dep")

                elif type(jth[1]) == list:
                    # jth is a List
                    # Here  the only thing is to check the first index (The head)
                    key1 = ""+str(jth_idxs[0])+","+str(ith_idxs[0])
                    key2 = "" + str(jth_idxs[0]) + "," + str(ith_idxs[-1])

                    if key1 in dep_ref and dep_ref[key1] in edge_labels_map:
                        self.edges.addEdge(j,i,edge_labels_map[dep_ref[key1]])

                    elif key1 in dep_ref :
                        self.edges.addEdge(j,i,"dep")
                    elif key2 in dep_ref and dep_ref[key2] in edge_labels_map:
                        self.edges.addEdge(j,i,edge_labels_map[dep_ref[key2]])
                    elif key2 in dep_ref:
                        self.edges.addEdge(j,i,"dep")

                elif ith_idxs[-1] == jth_idxs[-1]:

                    # More than 2 but
                    if ith_idxs[-1] == jth_idxs[0] :

                        #Assumed as Instrument
                        self.edges.addEdge(j,i,"Instrument")
                    elif ith_idxs[0] == jth_idxs[-1]:
                        self.edges.addEdge(i,j,"Instrument")

                    elif ith_idxs[-1] == jth_idxs[-1] :
                        key_s = ",".join([str(k) for k in jth_idxs])
                        key_r = str(jth_idxs[1])+","+str(jth_idxs[0])

                        if  (key_r in dep_ref and dep_ref[key_r] == 'amod') or (key_s in dep_ref and dep_ref[key_s] == 'amod'):
                            if isinstance(self.nuggets['v'+str(j)][1],list) :
                                long_word = " ".join(self.nuggets['v'+str(j)][1])
                            else:
                                long_word = self.nuggets['v'+str(j)][1]

                            self.nuggets['v'+str(j)] = [jth_idxs,long_word]
                            self.edges.addEdge(j,i,"example")

                        elif key1 in dep_ref and dep_ref[key1] in edge_labels_map:
                            self.edges.addEdge(i,j,edge_labels_map[dep_ref[key1]])
                        elif key2 in dep_ref and dep_ref[key2] in edge_labels_map:
                            self.edges.addEdge(j,i,edge_labels_map[dep_ref[key2]])

                elif ith_idxs[-1] == jth_idxs[-1] and isinstance(ith[1],str) and isinstance(jth[1],str) :

                    key1 = "" + str(jth_idxs[0]) + "," + str(jth_idxs[-1])
                    key2 = "" + str(jth_idxs[-1]) + "," + str(jth_idxs[0])

                    if  len(ith) > 2 and key1 in dep_ref  or key2 in dep_ref :
                        if key1 in dep_ref :
                            self.edges.addEdge(j,i,edge_labels_map[dep_ref[key1]])
                        else:
                            self.edges.addEdge(j,i,edge_labels_map[dep_ref[key2]])

                    elif  key1 in dep_ref  or key2 in dep_ref :
                        if key1 in dep_ref :
                            self.edges.addEdge(j,i,"same")
                        else:
                            self.edges.addEdge(j,i,"same")


                elif ith_idxs[0] == jth_idxs[0] and isinstance(ith[1],str) and isinstance(jth[1],str) :
                    key1 = "" + str(ith_idxs[0]) + "," + str(jth_idxs[-1])
                    key2 = "" + str(jth_idxs[-1]) + "," + str(ith_idxs[0])

                    if  key1 in dep_ref  or key2 in dep_ref :
                        if key1 in dep_ref and dep_ref[key1] in edge_labels_map :
                            self.edges.addEdge(j,i,edge_labels_map[dep_ref[key1]])
                        elif key2 in dep_ref and dep_ref[key2] in edge_labels_map:
                            self.edges.addEdge(j,i,edge_labels_map[dep_ref[key2]])

                    elif len(ith_idxs)>1 and ith_idxs[1] and jth_idxs[1]:
                        self.edges.addEdge(j, i, "extra")

                else:
                    if ith_idxs[-1] == jth_idxs[0]:
                        # Assumed as Instrument
                        self.edges.addEdge(j, i, "Instrument")

                    elif jth_idxs[-1] == ith_idxs[0]:
                        # Assumed as Instrument
                        self.edges.addEdge(i, j, "Instrument")

                    # Just check the regular
                    if key1 in dep_ref and dep_ref[key1] in edge_labels_map:
                        self.edges.addEdge(j,i,edge_labels_map[dep_ref[key1]])

                    elif key1 in dep_ref :
                        self.edges.addEdge(j,i, "dep")
                    elif key2 in dep_ref and dep_ref[key2] in edge_labels_map:
                        self.edges.addEdge(j,i,edge_labels_map[dep_ref[key2]])
                    elif key2 in dep_ref :
                        self.edges.addEdge(j,i,"dep")

        # Remove the indexes from the nuggets
        nuggets_list = {}
        for key,value in self.nuggets.items():
            nuggets_list[key] = value[1:]

        self.nuggets = nuggets_list

        # print(self.nuggets)

    def process(self):
        inter_nuggets = {}
        interPenTrees = []

        # Init Lemmatizer
        wlzer = WordNetLemmatizer()

        nugget_constructs = ['ccomp', 'xcomp', 'advcl', 'rcmod', 'vmod', 'amod', 'nn','nsubjpass','nsubj','dobj']
        (bag_of_words, pos_seq, ner_seq, chunk_seq,dep_ref, dep_seq, srls, syntax_tree) = pu.extract_annotations(
            self.raw_sentence)

        print("SRLs : ", srls)

        (list_head, list_idx, list, focus) = pu.list_extractor(dep_seq, bag_of_words,pos_seq)

        i = 0

        if len(list)>1 and list_head:
            list_head_word = pu.penn_to_wn(pos_seq[list_head])

            if not list_head_word:
                list_head_word = bag_of_words[list_head-1]
                list_head_pos = pu.penn_to_wn(pos_seq[list_head-1])
                if not list_head_pos:
                    list_head_pos = 'n'

                inter_nuggets['v1']= [[list_head+1]+[i+1 for i in list_idx],[wlzer.lemmatize(list_head_word,list_head_pos)],list]

            i = i+1

        for dep in dep_seq:
            if dep[0] in nugget_constructs or dep[0] in prep_phrases:
                a = int(dep[1])
                b = int(dep[2])

                if dep[0] == 'nn' :

                    pos1 = pu.penn_to_wn(pos_seq[a-1])
                    pos2 = pu.penn_to_wn(pos_seq[b-1])

                    if not pos1:
                        pos1 = 'n'
                    if not pos2:
                        pos2 = 'n'

                    if a<b :
                        inter_nuggets['v' + str(i + 1)] = [[a-1,b-1], "" +wlzer.lemmatize(dep[3],pos1) + " " + wlzer.lemmatize(dep[4],pos2)]
                    else:
                        inter_nuggets['v' + str(i + 1)] = [[b-1,a-1], "" +wlzer.lemmatize(dep[4],pos2) + " " + wlzer.lemmatize(dep[3],pos1)]

                elif dep[0] == 'nsubj' or dep[0] == 'dobj':
                    pos1 = pu.penn_to_wn(pos_seq[int(dep[1]) - 1])
                    pos2 = pu.penn_to_wn(pos_seq[int(dep[2]) - 1])
                    if not pos1:
                        pos1 = 'n'
                    if not pos2:
                        pos2 = 'n'
                    if a<b :
                        inter_nuggets['v' + str(i + 1)] = [[a-1,b-1],wlzer.lemmatize(replace_NEs(dep[3], ner_seq[int(dep[1])-1]),pos1)
                                                                ,wlzer.lemmatize(replace_NEs(dep[4], ner_seq[int(dep[2])-1]),pos2)]
                    else:
                        inter_nuggets['v' + str(i + 1)] = [[b-1,a-1],wlzer.lemmatize(replace_NEs(dep[4], ner_seq[int(dep[2])-1]),pos2)
                                                              ,wlzer.lemmatize(replace_NEs(dep[3], ner_seq[int(dep[1])-1]),pos1)]
                else :

                    pos1 = pu.penn_to_wn(pos_seq[int(dep[1]) - 1])
                    pos2 = pu.penn_to_wn(pos_seq[int(dep[2]) - 1])
                    if not pos1:
                        pos1 = 'n'
                    if not pos2:
                        pos2 = 'n'
                    if a<b :
                        inter_nuggets['v' + str(i + 1)] = [[a-1, b-1], wlzer.lemmatize(dep[3],pos1), wlzer.lemmatize(dep[4],pos2)]
                    else:
                        inter_nuggets['v' + str(i + 1)] = [[b-1,a-1], wlzer.lemmatize(dep[4]) , wlzer.lemmatize(dep[3],pos1)]

                i = i + 1

        removed = []

        for i in range(1,len(inter_nuggets)):
            try:
                curr_nugget = inter_nuggets['v'+str(i)]
            except KeyError:
                continue

            if i in removed:
                continue

            added = False

            for j in range(i+1,len(inter_nuggets)+1) :
                next_nugget = inter_nuggets['v' + str(j)]

                if j in removed:
                    continue

                if len(next_nugget[0])==1 and next_nugget[1] in pu.get_stop_words():
                    removed.append(j)
                    continue

                elif next_nugget[0][0] == curr_nugget[0][0] and next_nugget[0][1] == curr_nugget[0][1] and len(next_nugget[0]) == len(curr_nugget[0]):
                    removed.append(j)
                    continue

                # Duplicates
                if curr_nugget[0][0] in next_nugget[0] and curr_nugget[0][1] in next_nugget[0]:
                    lenth = len(self.nuggets)
                    added = True
                    if len(curr_nugget[0]) > len(next_nugget[0]):
                        self.nuggets["v" + str(lenth + 1)] = curr_nugget
                    elif next_nugget:
                        self.nuggets["v" + str(lenth + 1)] = next_nugget

                    removed.append(i)
                    removed.append(j)

                    break

                #Merge Nuggets Pivoting on one Item and remove both
                elif curr_nugget[0][1] == next_nugget[0][0]:
                    added = True
                    if 'VB' in pos_seq[int(curr_nugget[0][1]) - 1] :
                        lenth = len(self.nuggets)  # Len of the saved nuggets

                        if len(curr_nugget)==3:

                            self.nuggets['v' + str(lenth + 1)] = [[curr_nugget[0][0], curr_nugget[0][1],next_nugget[0][-1]],curr_nugget[1],
                                                        curr_nugget[2],next_nugget[-1]]

                        elif len(curr_nugget)==2 :
                            self.nuggets['v' + str(lenth + 1)] = [
                                [curr_nugget[0][0], curr_nugget[0][1], next_nugget[0][-1]], curr_nugget[1],next_nugget[-1]]

                    elif curr_nugget:

                        lenth = len(self.nuggets)  # Len of the saved nuggets
                        self.nuggets['v' + str(lenth + 1)] = curr_nugget

                        self.nuggets['v' + str(lenth + 2)] = [next_nugget[0][1]].extend(next_nugget[1])

                    removed.append(i)
                    removed.append(j)

                    break

            if not added :
                lenth = len(self.nuggets)  # Len of the saved nuggets
                self.nuggets['v' + str(lenth + 1)] = curr_nugget

        if len(inter_nuggets) not in removed:
            lenth = len(self.nuggets)  # Len of the saved nuggets
            if len(inter_nuggets) == 0:
                self.nuggets['v1'] = remove_stops(bag_of_words,pos_seq,ner_seq)
            else:
                keys = [*inter_nuggets]
                self.nuggets['v' + str(lenth + 1)] = inter_nuggets[keys[-1]]

        extra_cleaner = self.nuggets
        self.nuggets = {}
        k = 1
        # p=1
        for key,value in extra_cleaner.items():
            if value:
                if isinstance(value[1],str):
                    self.nuggets['v' + str(k)] = value

                    print(value)
                else:

                    nValue = [w for w in value[1]]
                    nValue = nValue + value[2]

                    print(nValue)
                    self.nuggets['v' + str(k)] = [value[0]]
                    self.nuggets['v' + str(k)].extend(nValue)
                k += 1

        self.generate_edges(dep_ref)
        self.generate_features()
        print("\n",self.__str__(),"\n")
        print("The Features",self.get_nugget_features())
        self.edges.toString()

class TAG:
    def __init__(self,graphlets_list):
        self.graphlet_ids = []
        self.all_graphlet_features = []
        self.all_nugget_words = []

        for graphlet in graphlets_list:

            print("Graphlet :", graphlet)

            self.graphlet_ids.append(graphlet.get_id())
            self.all_graphlet_features.append(graphlet.get_nugget_features())
            self.all_nugget_words.append(graphlet.__str__())

        # self.all_nugget_words = self.all_nugget_words.strip(",")

    def get_graphlet_ids(self):
        return self.graphlet_ids

    def get_all_graphlet_features(self):
        return  self.all_graphlet_features

    def get_nugget_words(self):
        return  self.all_nugget_words


def generate_graphlets():
    #load dataset

    # Load KB
    KB_file = "data/expl-tablestore-cleaned.csv"

    with open(KB_file,encoding='utf-8') as f:
        KB_sentences = list(csv.reader(f, delimiter='\t'))

    i = 1
    for row in KB_sentences:

        if i<360:
             i+=1
             continue
        # if i>1005:
        #      break
        graphlet = Graphlet(row[0],row[1],row[2])

        graphlet.process()

        print(graphlet.generate_features())
        print(graphlet.get_edges())
        print(graphlet.get_nuggets())
        #
        # with open('data/graphlets/'+row[0],'wb') as gf:
        #     pkl.dump(graphlet, gf)
        # #
        # print(i)
        # #
        i = i+1


def generate_TAGs():
    # LOAD ALL THE Pickles

    file_names = []
    for file in os.listdir("/home/mulang/Desktop/Learning/EMNLP-19/ExCon-TG19/data/graphlets"):
        # if file.endswith(".pkl"):
        file_names.append(os.path.join("/home/mulang/Desktop/Learning/EMNLP-19/ExCon-TG19/data/graphlets", file))

    graphlets = []
    for name in file_names:

        with open(name, "rb") as f:
            graphlet =pkl.load(f)
            graphlets.append(graphlet)


            f.close()

    print(len(graphlets))

    tags = []

    i=0
    while i < len(graphlets):

        curr_glet = graphlets[i]

        if not tags:
            tags.append([curr_glet])
            continue

        else:
            j=0

            new_tag = []
            temp_tags = copy(tags)

            print("Len : ",len(temp_tags))

            while j < len(temp_tags) :
                if i == j:
                    j += 1
                    continue
                curr_tag = temp_tags[j]

                tag_flag=True
                new_tag_in_loop = []
                curr_tag_in_loop = copy(curr_tag)

                for f in range(len(curr_tag)) :

                    outer_glet = curr_tag[f]

                    curr_words = set(curr_glet.__str__().split(","))
                    outer_words = set(outer_glet.__str__().split(","))

                    nug_intersection = [w.strip(" ") for w in list(curr_words.intersection(outer_words)) ]

                    if len(new_tag_in_loop) == 0 : #or len(nug_intersection)==0:
                        new_tag_in_loop.append(curr_glet)

                    if len(nug_intersection) > 0 and tag_flag:
                        new_tag_in_loop.append(outer_glet)
                        curr_tag_in_loop.append(curr_glet)

                        tag_flag = False

                    elif len(nug_intersection) > 0:
                        new_tag_in_loop.append(outer_glet)

                if len(new_tag_in_loop) == len(curr_tag_in_loop) and len(new_tag_in_loop) >1:
                    new_tag_in_loop = []
                elif not new_tag and len(new_tag_in_loop) >1 :
                    new_tag = new_tag_in_loop
                elif not new_tag and len(new_tag_in_loop) == 1 and tag_flag:
                    new_tag = new_tag_in_loop
                elif len(new_tag_in_loop) == 1 and tag_flag:
                    new_tag.extend(new_tag_in_loop)
                else:
                    new_tag.extend(new_tag_in_loop[1:])

                tags[j] = copy(curr_tag_in_loop)

                j += 1

            if len(new_tag)>0:
                tags.append(new_tag)

            print("Len Tags : ",len(tags))

        print(" Current I : ", i)
        i = i+1

        # if i > 1:
        #     break

    print("Total Number of TAGs ",len(tags))

    # h=1
    # for tag in tags:
    #     print("Type : ",type(tag))
    #     fin_TAG = TAG(tag)
    #
    #     with open('data/TAGs/' + "tag_"+str(h), 'wb') as gf:
    #         pkl.dump(fin_TAG, gf)
    #
    #     h += 1
    #

    print("Finnish")

def load_Graphlets(graphlets_folder):

    file_names = []
    for file in os.listdir(graphlets_folder):
        file_names.append(os.path.join(graphlets_folder, file))

    graphlets = {}
    for name in file_names:
        with open(name, "rb") as f:
            graphlet = pkl.load(f)
            graphlets[graphlet.get_id()] = graphlet

            f.close()
    return  graphlets

def load_Single_Graphlet(s_id,folder):
    file_names = []
    for file in os.listdir(folder):
        file_names.append(os.path.join(folder, file))
    graphlets = []
    for name in file_names:
        with open(name, "rb") as f:
            graphlet = pkl.load(f)

            if graphlet.get_id() == s_id :
                return graphlet

            f.close()
    return None

def get_all_Graphlets(folder):
    graphlets = load_Graphlets(folder)

    # for curr_g in graphlets:

def display_graphlets(folder,sent_id=None):

    if not sent_id:
        graphlets = load_Graphlets(folder)

        for curr_g in graphlets:

            # curr_g.generate_nugget_edges()
            print("Graphlet ID             :  ", curr_g.get_id())
            print("Graphlet All Words      :  ", curr_g.__str__())
            print("Graphlet Nuggets        :  ", curr_g.get_nuggets())
            print("Graphlet Features       :  ", curr_g.get_nugget_features())
            # print("Graphlet Edges      :  ", curr_g.get_edges().toString())

            print("\n\n")
    else:
        graphlet = load_Single_Graphlet(sent_id,folder)

        if graphlet:
            print("Graphlet ID             :  ", graphlet.get_id())
            print("Graphlet All Words      :  ", graphlet.__str__())
            print("Graphlet Nuggets        :  ", graphlet.get_nuggets())
            print("Graphlet Features       :  ", graphlet.get_nugget_features())
            # print("Graphlet Edges      :  ", graphlet.get_edges().toString())


            print("\n\n")

        else:
            print("Looks like the ID you gave is incorect")
            print()

if __name__ == "__main__":
    # tag = TAG(KB_sentences[5][1])
    # tag = Graphlet("A meter stick is a common instrument for measuring length and distance")
    # tag = Graphlet("32dsacsa","a stop watch can be used to measure time","wdq22dq")
    # tag = Graphlet("Give a general name for man,cows,goats, mokeys, and Giraffe")
    # tag = Graphlet("Speed is represented as distance divided by time")
    # tag = Graphlet("if two things are made from the same objects or material then those two things are similar")
    # tag = Graphlet("Students can calculate speed by measuring distance with a meter stick and measuring time with a clock")
    # tag = Graphlet("speed is measured over a long distance using a clock")
    # tag = Graphlet("67328jhjfsf989","soft is a kind of touch sensation","DEF")
    # tag = Graphlet("67328jhjfsf989","a pebble is a kind of small rock","RAT")
    # tag = Graphlet("67328jhjfsf989","Tennessee is a state located in the United States of America","RAT")
    # tag = Graphlet("786gdagjw87ghwsd","when both a dominant and recessive gene are present , the dominant trait will be visible or expressed","GHJ")
    # tag = Graphlet("786gdagjw87ghwsd","a pebble is a kind of small rock","GHJ")
    # tag = Graphlet("teruyw34867985h","a string is usually short in length with values between 2 and 1000 cm","YTHGFHA")
    # tag = Graphlet("32uhsjc889","friction causes the speed of an object to decrease","djew")
    # tag = Graphlet("32uhsjc889","gravity causes orbits","THD")
    # tag = Graphlet("32uhsjc889","shivering is a kind of shaking","kheudw")
    # tag = Graphlet("32uhsjc889","shivering is a kind of shaking","kheudw")
    # tag = Graphlet("32uhsjc889","Mamals iclude dogs, humans, elephants, and squirrels. They have great features ","kheudw")
    # tag = Graphlet("29euhjsdh","when both a dominant and recessive gene are present , the dominant trait will be visible or expressed","3298dj")
    tag = Graphlet("23eyfgsiu8434","milliliters ( mL ) is a unit used for measuring volume generally used for values between 1 and 1000","HGHS")
    tag.process()

    # print("The Question : ",q.__str__())
    # print("List : ",q.list)

    # for q_feat in q.all_word_features():
# 	# all_word_featuresfor f in q_feat:
# 	print("Featues : ", [f for f in q_feat])
#     generate_graphlets()

    # generate_TAGs()
