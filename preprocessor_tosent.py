#!/usr/bin/env python
# coding: utf-8

# In[9]:


from os import listdir 
import os
import sys
import re
import json
import csv
from tqdm import  tqdm
import jsonlines


# In[10]:


# !python -m spacy download en_core_web_md
import spacy
nlp = spacy.load('en_core_web_md')


# In[11]:


def removeSentenceWithoutVerb(fileContent):
    ''' Removes sentences from the file which do not have any verb forms '''
    
    fileData = []
    for eachSentence in fileContent:
        sent = nlp(eachSentence)
        isVBPresent = False
        for j in sent:
            if j.tag_.startswith("VB"): isVBPresent=True
        if isVBPresent:
            fileData.append(eachSentence)
    
    return fileData


# In[14]:


def dataloader(path, outpath):
    ''' Parses the data 
        1. Assumed sentences end with '. '
        2. Cleans sentences - strips spaces
        3. Remove sentences if no verb form
    '''
    
    allFileContents = {}
    files = [ f for f in listdir(path) if f]
    for f in tqdm(files):
        base = os.path.basename(f).split('.')[0]
        ff = os.path.join(path,f)
        fileHandle = open(ff,'r',encoding="utf-8")
        fw =  open(os.path.join(outpath,base)+'.csv','w')
        writer = csv.writer(fw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        data = fileHandle.read().split('\n')
        text = [i.replace("\t"," ").replace('“',"'").replace('”',"'").replace("…","...").replace("’","'").strip("\n").strip(" ").split(". ") for i in data]
        
        fileContent = []
        for each in text:
            for eacheach in each:
                if eacheach and len(eacheach)>2:
                    fileContent.append(eacheach.strip(" "))
        
        allFileContents[base] = {"content":removeSentenceWithoutVerb(fileContent)}
        
        with open('ctfwriteup.json','w') as fw:
            json.dump(allFileContents, fw)

        for lineno,eachline in enumerate(allFileContents[base]['content'],1):
            writer.writerow([base,lineno,eachline])
    return allFileContents
        


# In[15]:


fileData = dataloader(<input_folder>,<output_folder>)

