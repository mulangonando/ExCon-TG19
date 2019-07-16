     
     The class **"Graphlet"** encapsulates the nuggets within a sentence
     This class has a special feld called **"edges"** that is bascally an object of a Grpah of the 
     **"Adjacency Matric"** of the graphlet.
     
     Takes three parameters during instanciation:
        sentence Id (os contained in the dataset files)
        raw sentence (AFter basic cleaning)
        Sentence type : the type of the sentence given in the training file [KINDOF, PROTO-ACTION,CAUSE etc]
             
     
     **Graph.Process()**
     The member fucntion Graph.Process() carries out all processing transfroming the raw sentence into a graphlet ( a
     set of Nuggets (Vertices) and their edges stores in the Adjacency Matrix ---> Designed in a directed graph manner
     such that each cell in the matrix contains either an empty string or the label of the edge. also note that v1,v2 is 
     not same as v2,v1
     
     
     **get_nugget_features(self)**
     
     after processing completion we run this function to generate the features detailed in the paper on page 420
     aLl these are basically count based syntactic features
     
     get_nugget_features(self):
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
        
     These features are stored in the field **"nugget_features"** --> A dictionary that provides a way of accessing each
     by the feature name.
     
     This dictionary can be retrieved through an object of the class usung the member funtion  ** "get_nugget_features()" **
     
     All fields are accessible through getter methods in the class.
     
     **self.__str__()**
     
     returns a string with the words from the nuggets comma separated 
     
     DOCS