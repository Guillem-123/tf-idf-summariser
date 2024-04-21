import string
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


text='''
Call me Ishmael.
 Some years ago—never mind how long precisely—having little or no money in my purse,and nothing particular to interest me on shore,
I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen, and regulating the circulation.
Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses,
and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately 
stepping into the street, and methodically knocking people’s hats off—then, I account it high time to get to sea as soon as I can. This is my substitute for pistol and ball. 
With a philosophical flourish Cato throws himself upon his sword; I quietly take to the ship. ere is nothing surprising in this. If they but knew it, almost all men in their degree,
some time or other, cherish very nearly the same feelings towards the ocean with me. ere now is your insular city of the Manhattoes, belted round by wharves as Indian isles by
coral reefs—commerce surrounds it with her surf. Right and left, the streets take you waterward. Its extreme down-town is the battery, where that noble mole is washed by waves,
and cooled by breezes, which a few hours previous were out of sight of land. Look at the crowds of water-gazers there. Circumambulate the city of a dreamy Sabbath afternoon.
Go from Corlears Hook to Coenties Slip, and from thence, by Whitehall northward. What do you see?—Posted like silent sentinels all around the town, stand thousands upon thousands of mortal men fixed in ocean reveries.
Some leaning against the spiles; some seated upon the pier-heads; some looking over the bulwarks of ships from China; some high aloft in the rigging, as if striving to get a still better seaward peep.
But these are all landsmen; of week days pent up in lath and plaster—tied to counters, nailed to benches, clinched to desks. How then is this?
'''

#create a matrix containing the sentences as key 
#and a dict with the respective words/frequencies as value 
def frequency_matrix(sentences):
    frequency={}
    
    lemmatizer=WordNetLemmatizer() 
    stop_words=set(stopwords.words('english'))

    for sent in sentences:
        frequency_table={}
        words=word_tokenize(sent)
   
        for word in words:
            #normalise and lemmatize words
            word=word.lower()
            word=lemmatizer.lemmatize(word)
            #filter out stopwords and add them into the table + frequency
            if word not in stop_words:
                if word in frequency_table:
                    frequency_table[word]+=1
                else:
                    frequency_table[word]=1  


        
        frequency[sent[:15]]=frequency_table

    return frequency

#calculate the tf value for every word
#tf value is occurrences of word in sentence/nr of words in sentences
def tf_matrix(frequency):
    tf_matrix={}

    for k,v in frequency.items():
        tf_dict={}

        word_in_sentence= len(v)

        for word, count in v.items():
            tf_dict[word]=count/word_in_sentence

    
        tf_matrix[k]=tf_dict

    return tf_matrix

#create a matrix containing the nr of documents containing a word 
#necessary for calculating the idf value
def docs_containing_word(frequency):
    word_per_doc={}

    for sentence, frequency_table in frequency.items():
        for word, count in frequency_table.items():
            if word in word_per_doc:
                word_per_doc[word]+=1
            else:
                word_per_doc[word]=1
    return word_per_doc



#calculate the idf value for all the words
#idf= log(number of documents/number of documents that contain the word)
def gen_idf_matrix(freq_matrix, docs_containing_word, total_n_docs):
    idf_matrix={}
    for sent,frequency_table in freq_matrix.items():
        idf_table={}
        for word in frequency_table.keys():
            idf_table[word]=math.log10(total_n_docs/float(docs_containing_word[word]))
        
        idf_matrix[sent]=idf_table
    
    return idf_matrix


#multiply the two tables to get the tf-idf score 

def gen_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    #use zip to iterate over both dicts at the same time
    for (sentence1, freq_table1), (sentence2, freq_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value_a), (word2, value_b) in zip(freq_table1.items(),
                                                    freq_table2.items()):  
            tf_idf_table[word1] = float(value_a * value_b)

        tf_idf_matrix[sentence1] = tf_idf_table

    return tf_idf_matrix

#give weight to paragraph by adding tf-idf value of the sentence and dividing by the nr of words in that sentence
def get_score(tf_idf_matrix):

    sentenceValue = {}
#iterate over sentences and add the value
#divide by nr of words in sentence
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


#find the average score from the sentence value dictionary
def get_average_score(sentenceValue) :
    sum = 0
    for entry in sentenceValue:
        sum += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    avg = (sum / len(sentenceValue))

    return avg

#create a summary of the text
#by taking the sentences and checking if their value is higher that a threshold
#this means that we're taking the most characteristic sentences of the text
def summarise(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sent in sentences:
        if sent[:15] in sentenceValue and sentenceValue[sent[:15]] >= (threshold):
            summary += " " + sent
            sentence_count += 1

    return summary



def run_summarization(text):
    
    sentences = sent_tokenize(text)
    total_documents = len(sentences)

    freq_matrix = frequency_matrix(sentences)


    
    tf = tf_matrix(freq_matrix)

    count_doc_per_words = docs_containing_word(freq_matrix)

    
    idf = gen_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
 
    tf_idf_matrix = gen_tf_idf_matrix(tf, idf)

    sentence_scores = get_score(tf_idf_matrix)

    threshold = get_average_score(sentence_scores)

    #variable that can be used to modulate the brevity/score chosen for each word
    #to generate the summary
    percentage_of_average_score=0.7

    summary = summarise(sentences, sentence_scores, percentage_of_average_score * threshold)
    return summary


if __name__ == '__main__':
    result = run_summarization(text)
    print(result)
