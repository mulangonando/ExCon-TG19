#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:25:34 2017

@author: mulang
"""

from SPARQLWrapper import SPARQLWrapper, JSON
import time
import pickle as pic
import re
import os, sys
import csv
from elasticsearch import Elasticsearch
import gc
import time

proj_dir = os.path.abspath(os.path.join('.'))
dbo_file = os.path.abspath(os.path.join('.', 'dbpedia', 'dbo_final'))
sys.path.append(proj_dir)
dbp_file = os.path.abspath(os.path.join('.', 'dbpedia', 'dbp_final'))
dbo_binaries = os.path.abspath(os.path.join('.', 'dbpedia', 'onto_binaries_original'))
dbp_binaries = os.path.abspath(os.path.join('.', 'dbpedia', 'dbp_binaries'))
dbo_text_file = os.path.abspath(os.path.join('.', 'props_lists', 'dbo.csv'))
dbp_text_file = os.path.abspath(os.path.join('.', 'props_lists', 'dbp.csv'))

triples_folder = os.path.abspath(os.path.join('.', 'triples'))
abstarcts_folder = os.path.abspath(os.path.join('.','abstracts'))
# dbp_text_file =

# Elasticsearch configs
dbpedia_props_index = "dbpedia-props"
es_host = "localhost"
es_port = "9200"

# context = create_default_context(cafile="path/to/cert.pem")
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

class KBproperty(object):
    def __init__(self):
        self.prop_uri = ""
        self.domain = ""
        self.range = ""
        self.label = ""
        self.comment = ""
        self.instances_count = 0
        self.uniq_subjs = 0
        self.uniq_objs = 0
        self.annotations = None
        self.phrases = []
        self.syn_hypo = ""
        self.multi_word_syns = []

def search_es(nindex,tdoc,sstr):
    # query = {"query": {"match_phrase":{"uri": sstr}}}

    filed=''
    if tdoc == 'triple':
        field = 'subject'
    else:
        field = 'uri'

    query = {"query": {"wildcard": {"subject" : "*Tonight_Show_Starring_Johnny*"}}}
    # props_query = {"query": {"wildcard": {""+filed: "*"+sstr+"*"}}}
    # triples_query = {"query": {"wildcard": {""+filed: "*" + sstr + "*"}}}

    res = es.search(index="dbpedia_triples", doc_type="triple",body=query) #Check subject or Object
    # res = "multi_match": {"query": "guide","fields": ["_all"]}
    # res = es.search(index=nindex, doc_type=tdoc,body=query) #Return the property form ES
    # res = es.search(index=nindex, doc_type=tdoc, body=query)  # Return the property form ES

    print("%d documents found" % res['hits']['total'])
    for doc in res['hits']['hits']:
        #doc keys --> ['_index', '_type', '_id', '_score', '_source']
        # print("The Source thing looks like this",doc['_source'].keys())
        #The Source thing looks like this dict_keys(['uri', 'domain', 'range', 'label', 'comment', 'synonyms', 'phrases', 'annotations', 'num_subjects', 'num_objects'])

        if tdoc == 'triple':
            print("%s %s %s" % (doc['_id'],doc['_source']['subject'],doc['_source']['object']))
        else:
            print("%s %s %s %s %s" % (doc['_id'],doc['_source']['uri'],doc['_source']['domain'],doc['_source']['range'],doc['_source']['synonyms']))


def search_prop(sstr,sstr_full):

    query1 = {"size": 10,"query": {
                    "bool": {"must": {"bool":{"should":[
                        {"multi_match": {"query": sstr_full, "fields": ["label","synonyms"]}},
                        {"wildcard": {"uri": "*" + sstr_full.replace(" ", "")+"*"}}]}}}
        }}

    query2 = {"size": 10,"query": {"wildcard": {"label": "*" + sstr + "*"}}}

    res1 = es.search(index="properties", doc_type="dbo-relation", body=query1,request_timeout=20)
    res2 = es.search(index="properties", doc_type="dbo-relation", body=query2,request_timeout=20)

    dbo_prefix = 'http://dbpedia.org/ontology/'
    dbp_prefix = 'http://dbpedia.org/property'

    props = {}
    uniq_labels = []
    dbo_found=False
    dbp_found=False

    for doc in res1['hits']['hits']:
        uri = doc['_source']['uri']
        if uri not in props.keys() :
            props[uri] = {'label':doc['_source']['label'],'domain':doc['_source']['domain'],'range':doc['_source']['range'],'synonyms':doc['_source']['synonyms']}
            uniq_labels.append(uri.split("/")[-1])

        prop_type=uri.split("/")[-2]

        if prop_type=="ontology":
            dbo_found = True
        else:
            dbp_found = True

    for doc in res2['hits']['hits']:
        uri = doc['_source']['uri']
        if uri not in props.keys() :
            props[uri] = {'label':doc['_source']['label'],'domain':doc['_source']['domain'],'range':doc['_source']['range'],'synonyms':doc['_source']['synonyms']}
            uniq_labels.append(uri.split("/")[-1])

        prop_type=uri.split("/")[-2]

        if prop_type=="ontology":
            dbo_found = True
        else:
            dbp_found = True


    if dbp_found and not dbo_found:
        # print("Here Here")
        for label in uniq_labels:
            # print("Searched : ",dbo_prefix + label)
            query ={"size": 10,"query": {"wildcard": {"uri": "*"+label}}}
            res = es.search(index="properties", doc_type="dbo-relation", body=query, request_timeout=20)

            for doc in res['hits']['hits']:
                uri = doc['_source']['uri']
                if uri not in props.keys():
                    props[uri] = {'label': doc['_source']['label'], 'domain': doc['_source']['domain'],
                                  'range': doc['_source']['range'], 'synonyms': doc['_source']['synonyms']}

    elif dbo_found and not dbp_found:
        # print("Here Here")
        for label in uniq_labels:
            # print("Searched : ",dbo_prefix + label)
            query ={"query": {"wildcard": {"uri": "*"+label}}}
            res = es.search(index="properties", doc_type="dbo-relation", body=query, request_timeout=20)

            for doc in res['hits']['hits']:
                uri = doc['_source']['uri']
                if uri not in props.keys():
                    props[uri] = {'label': doc['_source']['label'], 'domain': doc['_source']['domain'],
                                  'range': doc['_source']['range'], 'synonyms': doc['_source']['synonyms']}
    return props


def search_triple(tstrs):
    triples = []

    '''
        so we search by all combined by "_" if miss, search by each 
        but results fileter on existance of full word on splits by _ or space [From the seacrh module]
    '''

    for tstr in tstrs.split(" "):
        query1 = {"size": 10,"query": {"bool":{"must": [ {"wildcard": {"doc.sphrase.keyword" : "*"+tstr+"*"}}]}}}

        # , "should": [
        query2 = {"size": 10,"query": {"bool":{"must": [ {"wildcard": {"doc.ophrase.keyword" : "*"+tstr+"*"}}]}}}

        res1 = es.search(index="dbpedia_triples", doc_type="triple",body=query1,request_timeout=20)
        res2 = es.search(index="dbpedia_triples", doc_type="triple",body=query2,request_timeout=20)

        for doc in res1['hits']['hits']:
            e_label_list = doc['_source']['doc']['subject'].split("/")[-1].split("_")
            if doc['_source']['doc']['predicate'].strip().find('wikiPage')==-1 and tstr in e_label_list :
                # print("  %s       %s       %s      %s  " % (doc['_id'],doc['_source']['doc']['subject'],doc['_source']['doc']['predicate'],doc['_source']['doc']['object'])) #doc['_source']['subject'],
                triples.append([doc['_id'],doc['_source']['doc']['subject'],doc['_source']['doc']['predicate'],doc['_source']['doc']['object']])

        for doc in res2['hits']['hits']:
            e_label_list = doc['_source']['doc']['subject'].split("/")[-1].split("_")
            if doc['_source']['doc']['predicate'].strip().find('wikiPage')==-1 and tstr in e_label_list:
                # print("  %s       %s       %s      %s  " % (doc['_id'],doc['_source']['doc']['subject'],doc['_source']['doc']['predicate'],doc['_source']['doc']['object'])) #doc['_source']['subject'],
                triples.append([doc['_id'],doc['_source']['doc']['subject'],doc['_source']['doc']['predicate'],doc['_source']['doc']['object']])


    return triples

def abstracts_to_es():

    abs_files = os.listdir(abstarcts_folder)
    i=0
    for abs_f in abs_files[0:2]  :
        abs_triples = []
        with open(abstarcts_folder+"/"+abs_f, 'r') as f: #Add os.pathseperator
            rows = csv.reader(f)
            k=1
            for row in rows:
                if row[0].strip()[0:4].strip() == 'http':
                    abs_triples.append(row)

                    doc = {}

                    if abs_triples[-1][-1].find('*')>-1:
                        # print("\n line " + abs_triples[-1][-1])

                        doc['uri'] = row[1]
                        doc['abstract'] = row[2]

                        res = es.index(index="dbpedia_abstracts", doc_type='abstract', id=k, body=doc)
                else:
                    abs_triples[-1] = abs_triples[-1]+" "+' '.join(part.stip() for part in row)

                    print("\n The K : "+str(k)+" line "+abs_triples[-1])
                    print("\n")

                k = k+1

            f.close()
        del abs_triples
        gc.collect()

        i = i+1

        if i>2:
            break


def get_AllKBproperties():
    # fileObject = open(dbpedia_binaries,'r')
    valid_poss = ['NN', 'NNS', 'NNP', 'NNPS','VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ','RB', 'RBR', 'RBS','JJ', 'JJR', 'JJS']

    dbpedia_props = []

    file_names = []
    for file in os.listdir(dbp_binaries):
        if file.endswith(".pkl"):
            file_names.append(os.path.join(dbp_binaries, file))

    for name in file_names:
        with open(name, "rb") as f:
            kb_prop = KBproperty()
            kb_prop = pic.load(f)
            dbpedia_props.append(kb_prop)
            synos = kb_prop.syn_hypo
            kb_prop.syn_hypo = synos.strip()[0:-1]

            print((kb_prop.prop_uri, "Synonyms : ",kb_prop.syn_hypo))
            print("The SYynoonyms : ", kb_prop.syn_hypo)
            f.close()

    return dbpedia_props

def index_bulk_triples_es():
    # Toal Rriples frm dbp : 40962844
    triples_files = os.listdir(triples_folder)
    print("Number of Files : ", len(triples_files))

    k = 1

    for tf in triples_files:

        # print("cleared the count")
        with open(triples_folder+"/"+tf, 'r') as f: #Add os.pathseperator
            rows = csv.reader(f)
            for row in rows :
                k=k+1
                # if row[0].split("/")[-1]=='spouse':
                #     print(row[1])
                yield {
                    "_index": "dbpedia_triples",
                    "_type": "tripl",
                    "doc": {"subject": row[1],
                            "sphrase" : ' '.join(row[1].split("/")[-1].split("_")),
                            'predicate' : row[0],
                            'object' : row[2],
                            'ophrase':' '.join(row[2].split("/")[-1].split("_"))
                            },
                }
            del rows
            gc.collect()

        f.close()

    print("Total count : ",k)

def index_triples_on_es():
    dbp_files_folder = ""
    triples_files = os.listdir(dbp_files_folder)

    docs = {}
    k = 1

    try:
        for tf in triples_files:
            if k<1298317 :
                k = k+1
                continue
            with open(dbp_files_folder+"/"+tf, 'r') as f: #Add os.pathseperator
                rows = csv.reader(f)
                doc = {}
                for row in rows :
                    doc['subject'] = row[1]
                    doc['predicate'] = row[0]
                    doc['object'] = row[2]

                    k = k + 1

                    if len(doc) % 500 == 0:
                        res = es.index(index="dbpedia_triples", body=docs)
                    docs['triple'] = doc

    except Exception as ex:
        with open("dbpedia/missed.txt", 'a') as log:
            log.write("Error on prop "+k+" : "+ex)
        print('Exception : ', ex)

def index_on_es():
    # get the key as the uri of the property
    # fetch all the subjects and objects of this property
    # create the document
    # update the document with subsequent objects and subjects

    all_props = get_AllKBproperties()

    # sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    # sparql.setUseKeepAlive()

    with open(dbp_text_file, 'r') as f:
        prop_entries = csv.reader(f)

        doc = {}
        for row in prop_entries:

            doc['uri'] = row[0]
            doc['domain'] = row[1]
            doc['range'] = row[2]
            doc['label'] = row[3]
            doc['comment'] = row[4]
            doc['synonyms'] = ""

            for prop in all_props:

                if prop.prop_uri.strip() == row[0].strip():
                    doc['synonyms'] = " ".join(prop.syn_hypo.split(","))
                    doc['phrases'] = prop.phrases
                    doc['annotations'] = prop.annotations

            # print("synonyms : ", doc['synonyms'])
            '''Fetch the new value from dbpedia tog with all triples'''
            try:

                doc['num_subjects'] = int(row[5])
                doc['num_objects'] = int(row[6])

                res = es.index(index="properties", doc_type='dbo-relation', body=doc)

            except Exception as ex:
                with open("dbpedia/missed.txt", 'a') as log:
                    log.write("Error on prop "+k+" : "+ex)
                print('Exception : ', ex)

def save_to_files():

    all_props = get_AllKBproperties()

    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setUseKeepAlive()

    with open(dbp_file, 'r') as f:
        prop_entries = csv.reader(f)

        doc = {}
        k = 1
        for row in prop_entries:
            doc = []
            doc.append(row[0])
            doc.append(row[1])
            doc.append(row[2])
            doc.append(row[3])
            doc.append(row[4])

            if int(row[-2]) == 0 :
                continue
            elif k<26 :
                k=k+1
                continue

            else:

                '''Fetch the new value from dbpedia tog with all triples'''
                try:
                    count_query = "SELECT COUNT(DISTINCT ?s) COUNT(DISTINCT ?o)  WHERE { ?s <" + row[0].strip(
                        "'").strip(
                        '"') + "> ?o . }"
                    sparql.setQuery(count_query)
                    sparql.setReturnFormat(JSON)
                    results = sparql.query().convert()

                    # count = 0;
                    for result in results["results"]["bindings"]:
                        doc.append(int(result["callret-0"]["value"]))
                        doc.append(int(result["callret-1"]["value"]))

                    # Save the prop
                    # Send to Elasticsearch
                    # res = es.index(index="dbpedia-props", doc_type='dbp-relation', id=k, body=doc)

                    with open(dbp_text_file, 'a') as fd:
                        writer = csv.writer(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                            lineterminator='\n')
                        writer.writerow(doc)

                    # Create the tripples
                    triples_query = "SELECT ?s ?o WHERE { ?s <" + row[0].strip("'").strip(
                        '"') + "> ?o . } OFFSET 0 LIMIT 9998"

                    sparql.setQuery(triples_query)
                    sparql.setReturnFormat(JSON)
                    res = sparql.query().convert()

                    i = 0
                    prev_i = 0
                    count_on = True
                    while count_on:
                        triples = []
                        for r in res["results"]["bindings"]:
                            triple_doc = []
                            triple_doc.append(row[0].strip("'").strip('"'))
                            triple_doc.append(r["s"]["value"])
                            triple_doc.append(r["o"]["value"])

                            i = i + 1

                            triples.append(triple_doc)

                        print("Curr prop "+str(k)+" Triple number : ", i + 1)

                        with open('dbp_refetched/dbp_' + row[0].strip("'").strip('"').split('/')[-1] + "_" + str(
                                k) + ".csv", 'a') as fd:
                            writer = csv.writer(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                                lineterminator='\n')
                            writer.writerows(triples)

                        del triples
                        gc.collect()

                        if i%969806==0:
                            time.sleep(.1500)


                        if i - prev_i < 9997:
                            count_on = False
                            break
                        else:
                            prev_i = i


                        triples_query = "SELECT ?s ?o WHERE { ?s <" + row[0].strip("'").strip(
                            '"') + "> ?o . } OFFSET " + str(i) + " LIMIT " + str(9998) + ""

                        sparql.setQuery(triples_query)
                        sparql.setReturnFormat(JSON)
                        res = sparql.query().convert()

                except Exception as ex:
                    with open("dbpedia/missed.txt", 'a') as log:
                        log.write("prop : " + str(k) + " : ")
                    print('Exception : ', ex)
                    pass

            time.sleep(.1500)
            k = k + 1


def entitySearch(query):
    indexName = "dbpedia_triples"
    docType = "triple"
    triples = []

    elasticResults = es.search(index=indexName, doc_type=docType, body={"query": {
        "bool": {
            "must": {
                "bool": {"should": [
                    {"multi_match": {"query": query, "fields": ["doc.sphrase"], "fuzziness": 6}},
                    {"multi_match": {"query": "http://dbpedia.org/resource/"+ query.replace(" ", "_"),
                                     "fields": ["doc.object"], "fuzziness": 6}}]}
            }
        }
    }, "size": 20})

    for doc in elasticResults['hits']['hits']:
        e_label_list = doc['_source']['doc']['subject'].split("/")[-1].split("_")
        if doc['_source']['doc']['predicate'].strip().find('wikiPage') == -1 :#and query in e_label_list:
            triples.append([doc['_source']['doc']['subject'], doc['_source']['doc']['predicate'],
                            doc['_source']['doc']['object']])

    return triples

def ontologySearch(query):
    indexName = "dbontologyindex"
    docType = " " # Change this to use it
    results = []
    elasticResults = es.search(index=indexName, doc_type=docType, body={
        "query": {
            "bool": {
                "must": {
                    "bool": {"should": [
                        {"multi_match": {"query": "http://dbpedia.org/ontology/" + query.replace(" ", ""),
                                         "fields": ["uri"], "fuzziness": "AUTO"}},
                        {"multi_match": {"query": query, "fields": ["label"]}},
                    ]}
                }
            }
        }
        , "size": 15
    })
    # print(elasticResults)
    for result in elasticResults['hits']['hits']:
        if not result["_source"]["uri"][result["_source"]["uri"].rfind('/') + 1:].istitle():
            results.append((result["_source"]["label"], result["_source"]["uri"]))
    return results
    # for result in results['hits']['hits']:
    # print (result["_score"])
    # print (result["_source"])
    # print("-----------")


def propertySearch(query):
    indexName = "properties"
    results = []
    elasticResults = es.search(index=indexName, doc_type='dbo-relation', body={
        "query": {
            "bool": {
                "must": {
                    "bool": {"must": [
                        {"multi_match": {"query": query, "fields": ["label","synonyms"]}},
                        {"multi_match": {"query": "http://dbpedia.org/property/" + query.replace(" ", ""),
                                         "fields": ["uri"], "fuzziness": "AUTO"}}]}
                }
            }
        }
        , "size": 10})
    for result in elasticResults['hits']['hits']:
        results.append((result["_source"]["label"], result["_source"]["uri"]))
    return results

if __name__ == '__main__':
    print(search_prop('wife','wife'))


