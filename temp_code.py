if type(value[1]) == list:
    nuggets_list[key] = value[1].extend(value[2])

else:
    nuggets_list[key] = value[1:]






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
                            long_word = " ".join(self.nuggets['v'+str(j)][1])
                            self.nuggets['v'+str(j)] = [jth_idxs,long_word]

                            reduced_ith = self.nuggets['v'+str(i)][1]
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



####################################
def recurse_terminal(tree, string):
    pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                'POS', 'PRP', \
                'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
                'WP', 'WP$', \
                'WRB']

    poss = ""
    actual_word = ""

    if tree.label() in pos_tags:

        if r == 0:
            poss = rc.label()
            actual_word = "_".join([w for w in rc.leaves()])  # rc.leaves())
        else:
            poss = poss + "_" + rc.label()
            actual_word = actual_word + "_" + "_".join([w for w in rc.leaves()])

        r = r + 1

    right_string = "(" + poss + " " + actual_word + ")"

    string = "(prep_" + label + " " + left_string + " " + right_string + ")"

    return string



#########################################################33

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

            left_sibling = [t for t in left]

            if isinstance(left_sibling[0],str):
                left_sibling = [tree.left_sibling()]


            #PROCESS THE LEFT SIDE
            l = 0
            for ls in left_sibling:

                if l == 0:
                    print(ls)
                    poss = ls.label()
                    actual_word = ls.leaves()[0]
                else:
                    poss = poss + "_" + ls.label()
                    actual_word = actual_word + "_" + ls.leaves()[0]
                l += 1

            left_string = "(" + poss + " " + actual_word + ")"

            #PROCESS THE RIGHT SIDE

            curr_children = [c for c in tree]

            label = curr_children[0].leaves()[0]

            right_children = [child for child in curr_children[1:]]

            poss = ""
            actual_word = ""

            # children = tree.children()

            r = 0
            for rc in right_children:

                if r == 0:
                    poss = rc.label()
                    actual_word = "_".join([w for w in rc.leaves()])  # rc.leaves())
                else:
                    poss = poss + "_" + rc.label()
                    actual_word = actual_word + "_" + "_".join([w for w in rc.leaves()])

                r = r + 1

            right_string = "(" + poss + " " + actual_word + ")"

            string = "(prep_" + label + " " + left_string + " " + right_string + ")"

            queue.append(string)

        len_children = len(children)
        i = 1
        while i <= len_children:
            buildQueuePenTree(children[-i], queue)
            i = i + 1


        else:
            out_pos = ['CC',',']

            children = [w for w in tree]
            if isinstance(children[0],str):
                pass
            else:
                children_labels = [w.label() for w in tree]
                num_commas = children_labels.count(',')
                conjunct = children_labels.count('CC')

                if num_commas>1 :
                    #definately a list

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

                    label_pos =  tree.left_sibling().label()
                    label = tree.left_sibling().leaves()[0]

                    string_list = "(("+label_pos+" "+label+") "+"("+list_parent+" "+list_compound+"))"

                    queue.append(string_list)

    return queue



def getPenTreeBankSpecial(pase_str,dep_seq,bog,pos):
    '''
    :param pase_str: A sring of the parsetree from the stanford parser
    :return: A list of the subquestions
    '''

    synt_tree = Tree.fromstring(pase_str)

    queue = DQ([])
    pen_queue = buildQueuePenTree(ParentedTree.convert(synt_tree), queue)

    print("Results : ",pen_queue)

    nugget_constructs = ['ccomp', 'xcomp', 'advcl', 'rcmod', 'vmod', 'amod', 'nn', 'nsubjpass', 'nsubj', 'dobj']


    # the_list = list_extractor(dep_seq, bog,pos)
    #
    # print(the_list)


    return pen_queue