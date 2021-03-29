import re
from datetime import datetime

def _split_doc(doc):
    """
    Split a document into a list of words.
    
    Arguments: 
    doc: String which represents a document.
    
    Returns:
    words: List of words that apear in the document.
    """
    doc = re.sub(r'\n', ' ', doc)
    doc = re.sub(r'\{\{.*?\}\}', r'', doc)
    doc = re.sub(r'Category:.*', '', doc)
    doc = re.sub(r'Source\s*.*', '', doc)
    doc = re.sub(r'References\s*.*', '', doc)
    doc = re.sub(r'External [Ll]inks\s*.*', '', doc)
    doc = re.sub(r'External [Ll]inks and [Rr]eferences\s*.*', '', doc)
    doc = re.sub(r'See [Aa]lso\s*.*', '', doc)
    
    doc = doc.lower() 
    
    doc = re.sub(r'http://[^\s]*', '', doc)        
    doc = re.sub(r'-', ' ', doc)
    doc = re.sub(r'[^a-z ]', '', doc)
    doc = re.sub(r' +', ' ', doc)
    words = doc.split()
    return words

def _count_words(doc, vocab):
    """
    Counts how many times each word of dictionary appears in a document.
    
    Arguments:
    doc: String which represents a document.
    vocab: Dictionary mapping from words to integer ids.
    
    Returns:
    ids: List of tokens which correspondes to the words appearing in the document.
    cts: List of the number that each token appear the document. 
    """
    ddict = dict()
    words = _split_doc(doc)
    for word in words:
        if (word in vocab):
            wordtoken = vocab[word]
            if (not wordtoken in ddict):
                ddict[wordtoken] = 0
            ddict[wordtoken] += 1
    ids, cts = list(ddict.keys()), list(ddict.values())
    return ids, cts


def parse_docs(docs, vocab, parallel=False):
    """
    Parse a document into a list of word ids and a list of counts,
    or parse a set of documents into two lists of lists of word ids
    and counts.

    Arguments: 
    docs:  List of D documents. Each document must be represented as
           a single string. (Word order is unimportant.) Any
           words not in the vocabulary will be ignored.
    vocab: Dictionary mapping from words to integer ids.

    Returns: 
    wordids: List of lists. wordids, says what vocabulary tokens are present in
    each document. wordids[i][j] gives the jth unique token present in
    document i. (Don't count on these tokens being in any particular
    order.)
    wordcts: List of lists. wordcts, says how many times each vocabulary token is
    present. wordcts[i][j] is the number of times that the token given
    by wordids[i][j] appears in document i.
    """
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp
    
    wordids = list()
    wordcts = list()
    D = len(docs)    
    for d in range(0, D):
        doc = docs[d].decode('utf-8')
        ids, cts = _count_words(doc, vocab)
        
        # if failed to parse, return nothing
        if len(ids)>0:
            wordids.append(ids)
            wordcts.append(cts)
    
    return((wordids, wordcts))

def timestamp():
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S")  

def main():
    pass

if __name__ == '__main__':
    main()