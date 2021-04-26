import numpy as np
import csv
from tqdm import tqdm
import codecs
from nltk.stem import WordNetLemmatizer
#from nltk.stem.porter import PorterStemmer
import gensim # contains Doc2Vec and related functions
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as hcluster
import matplotlib.pyplot as plt

PATH = "C:\\Users\\lprescott\\Documents\\Religious texts\\"

bible_file_names = ["Bible.txt"]
# apologies, but due to not wanting to distribute any files without permission, please see https://tanzil.net/trans/ for all translations of the Quran used below
# files were reformatted for this program such that "1|1|In the name of God, the Gracious, the Merciful." became "Al-Fatiha	1	1	In the name of God, the Gracious, the Merciful."
quran_file_names = ['en_ahmedali.txt','en_ahmedraza.txt','en_arberry.txt','en_daryabadi.txt','en_hilali.txt','en_itani.txt','en_maududi.txt','en_mubarakpuri.txt','en_pickthall.txt','en_qarai.txt','en_qaribullah.txt','en_sahih.txt','en_sarwar.txt','en_shakir.txt','en_wahiduddin.txt','en_yusufali.txt']

bible_book_names = ["Genesis","Exodus","Leviticus","Numbers","Deuteronomy","Joshua","Judges","Ruth","1 Samuel","2 Samuel","1 Kings","2 Kings","1 Chronicles","2 Chronicles","Ezra","Nehemiah","Esther","Job","Psalms","Proverbs","Ecclesiastes","Song of Solomon","Isaiah","Jeremiah","Lamentations","Ezekiel","Daniel","Hosea","Joel","Amos","Obadiah","Jonah","Micah","Nahum","Habakkuk","Zephaniah","Haggai","Zechariah","Malachi","Matthew","Mark","Luke","John","Acts","Romans","1 Corinthians","2 Corinthians","Galatians","Ephesians","Philippians","Colossians","1 Thessalonians","2 Thessalonians","1 Timothy","2 Timothy","Titus","Philemon","Hebrews","James","1 Peter","2 Peter","1 John","2 John","3 John","Jude","Revelation"]
quran_book_names = ["Al-Fatiha","Al-Baqarah","Al Imran","An-Nisa","Al-Maidah","Al-Anam","Al-Araf","Al-Anfal","At-Tawbah","Yunus","Hud","Yusuf","Ar-Rad","Ibrahim","Al-Hijr","An-Nahl","Al-Isra","Al-Kahf","Maryam","Ta-Ha","Al-Anbiya","Al-Hajj","Al-Muminun","An-Nur","Al-Furqan","Ash-Shuara","An-Naml","Al-Qasas","Al-Ankabut","Ar-Rum","Luqmaan","As-Sajda","Al-Ahzaab","Saba (surah)","Faatir","Yaseen","As-Saaffaat","Saad","Az-Zumar","Ghafir","Fussilat","Ash_Shooraa","Az-Zukhruf","Ad-Dukhaan","Al-Jaathiyah","Al-Ahqaaf","Muhammad","Al-Fath","Al-Hujuraat","Qaaf","Adh-Dhaariyaat","At-Toor","An-Najm","Al-Qamar","Ar-Rahman","Al-Waqia","Al-Hadeed","Al-Mujadila","Al-Hashr","Al-Mumtahanah","As-Saff","Al-Jumuah","Al-Munafiqoon","At-Taghabun","At-Talaq","At-Tahreem","Al-Mulk","Al-Qalam","Al-Haaqqa","Al-Maaarij","Nooh","Al-Jinn","Al-Muzzammil","Al-Muddaththir","Al-Qiyamah","Al-Insaan","Al-Mursalaat","An-Naba","An-Naaziaat","Abasa","At-Takweer","Al-Infitar","Al-Mutaffifeen","Al-Inshiqaaq","Al-Burooj","At-Taariq","Al-Alaa","Al-Ghaashiyah","Al-Fajr","Al-Balad","Ash-Shams","Al-Layl","Ad-Dhuha","Ash-Sharh","At-Teen","Al-Alaq","Al-Qadr","Al-Bayyinahh","Az-Zalzalah","Al-Aadiyaat","Al-Qaariah","At-Takaathur","Al-Asr","Al-Humazah","Al-Feel","Quraysh","Al-Maaoon","Al-Kawthar","Al-Kaafiroon","An-Nasr","Al-Masad","Al-Ikhlaas","Al-Falaq","Al-Naas"]



def read_file(file_name, books):
    # some Arabic symbols in a couple of the Quranic files mess up the utf-8 decoding, but the default charmaps decoding works instead
    text = [[] for i in range(len(books))]
    with codecs.open(PATH + file_name, 'r', encoding='utf-8') as file:
        try:
            for line in file.read().split("\n"):
                temp_book_num = int(line.split("\t")[1]) # book number
                for i, book in enumerate(books):
                    if temp_book_num == i+1:
                        text[i].append(gensim.utils.simple_preprocess(line.split("\t")[3].replace("\n",""))) # line or verse number
                        break
            return text
        except:
            with codecs.open(PATH + file_name, 'r', encoding='charmap') as file:
                try:
                    for line in file.read().split("\n"):
                        temp_book_num = int(line.split("\t")[1]) # book number
                        for i, book in enumerate(books):
                            if temp_book_num == i+1:
                                text[i].append(gensim.utils.simple_preprocess(line.split("\t")[3].replace("\n",""))) # line or verse number
                                break
                    return text
                except:
                    print("ERROR")

def prep(text_file_name, books):
    # read sentences
    print("Reading and lemmatizing")
    temp_text = read_file(text_file_name, books)
    
    print("# books =", len(books))
    print("Example 1st 5 sentences of 1st book", books[0:5], temp_text[0][0:5])
    
    # lemmatize sentences
    text = [[] for i in range(len(books))]
    for i in range(len(books)):
        for j, temp_sent in enumerate(temp_text[i]):
            sent = []
            for k, word in enumerate(temp_sent):
                sent.append(WordNetLemmatizer().lemmatize(word))
            text[i].append(sent)
    print("# of sentences in each book =", [len(book_texts) for book_texts in text])
    print("# of words in each book =", [sum([len(sent) for sent in book_texts]) for book_texts in text])            
    
    # put together TaggedDocument object for doc2vec training
    # doc2vec requires 1d array of words for build_vocab & train, not 2d words in sentences in doc
    text_book_list = []
    for i in range(len(text)):
        text_book_list.append(gensim.models.doc2vec.TaggedDocument(words=[x for y in text[i] for x in y], tags=[i]))
    print("Made TaggedDocument")
    
    # doc2vec settings
    doc2vec_dim = 100
    doc2vec_window = 10
    doc2vec_min_count = 1
    doc2vec_epochs = 20
    
    # build vocab and train doc2vec model
    model = gensim.models.Doc2Vec(vector_size=doc2vec_dim, window=doc2vec_window, min_count=doc2vec_min_count, epochs=doc2vec_epochs, workers=8)
    model.build_vocab(text_book_list)
    model.train(text_book_list, total_examples=model.corpus_count, epochs=model.iter)
    print("Done training")
    
    return books, text, text_book_list, model

def similarity(books, text_book_list, model, destination_file_name):
    # calculate len(books)^2 cosine similarity matrix
    similarity_matrix = np.zeros((len(model.docvecs), len(model.docvecs)))
    for i in range(len(books)):
        for j in range(len(books)):
            if i == j:
                similarity_matrix[i][j] = 1
            else:
                similarity_matrix[i][j] = cosine_similarity([model.docvecs[i]], [model.docvecs[j]])[0][0]
    
    # plot dendrogram from similarity matrix
    plt.figure(figsize=(10,5), dpi=200)
    for i in range(len(similarity_matrix)): # must convert diagonal to 0 for below function to work
        similarity_matrix[i][i] = 0
    condensed_similarity_matrix = ssd.squareform(similarity_matrix)
    #print("condensed_similarity_matrix =", condensed_similarity_matrix)
    linkage = hcluster.linkage(1 - condensed_similarity_matrix / 2)
    #print("linkage =", linkage)
    dendro = hcluster.dendrogram(linkage, labels=books)
    #print("dendro =", dendro)
    #plt.show()
    plt.savefig(PATH + destination_file_name + '.png')
    
    return similarity_matrix

def semantic(books, text, model):
    # the length of whole book document vectors was intended to represent some form of semantic density
    # but it also positively, nonlinearly correlated with book length, so the more frequently used average word vector length was used for semantic density instead
    book_doc_vector_legths = [np.linalg.norm(model.docvecs[i]) for i in range(len(books))]
    print("Book document vector lengths =", book_doc_vector_legths)
    
    # calculate sum of word vector length for semantic density
    book_word_sum_array = []
    for i, book in enumerate(text):
        temp_book_sum = 0
        for j, sentence in enumerate(book):
            for k, word in enumerate(sentence):
                temp_book_sum = temp_book_sum + np.linalg.norm(model.wv[word])
        book_word_sum_array.append(temp_book_sum)
    
    print("Book avg word vector lengths =", book_word_sum_array)
    return book_doc_vector_legths, book_word_sum_array

def sentiment(text, model):
    # 4x different values representing sentiment of each sentence
    compounds = []
    negatives = []
    neutrals = []
    positives = []
    sid = SentimentIntensityAnalyzer()
    
    for i, book in enumerate(tqdm(text)):
        temp_compounds = []
        temp_negatives = []
        temp_neutrals = []
        temp_positives = []
        for j, sentence in enumerate(book):
            ss = sid.polarity_scores(" ".join(word for word in sentence)) # convert list of words to sentence string
            temp_compounds.append(ss['compound'])
            temp_negatives.append(ss['neg'])
            temp_neutrals.append(ss['neu'])
            temp_positives.append(ss['pos'])
        compounds.append(temp_compounds)
        negatives.append(temp_negatives)
        neutrals.append(temp_neutrals)
        positives.append(temp_positives)
    
    print("Sentiments done")
    return compounds, negatives, neutrals, positives

def main(file_names, book_names, destination_file_name):
    for text_file_name in file_names:
        print("Current file =", text_file_name)
        books, text, text_book_list, model = prep("Text files\\" + text_file_name, book_names)
        
        similarity_matrix = similarity(books, text_book_list, model, text_file_name.replace(".txt","")+"_dendrogram")
        book_doc_vec_lens, book_sum_word_vec_lens = semantic(books, text, model)
        compounds, negatives, neutrals, positives = sentiment(text, model)
        
        # write semantic densities, similarity matrix, and sentiments to file
        with open(PATH + destination_file_name + "_" + text_file_name.replace(".txt","") + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            writer.writerow(["Books"] + books)
            writer.writerow(["Book document vector lengths"] + book_doc_vec_lens)
            writer.writerow(["Book sum of word vector lengths"] + book_sum_word_vec_lens)
            writer.writerow([])
            
            writer.writerow(["Similarity matrix"] + books)
            for i, temp in enumerate(similarity_matrix):
                writer.writerow([books[i]] + [str(temp_2) for temp_2 in temp])
            
            writer.writerow([])
            writer.writerow(["Book","Sentence Length","Compound","Negative","Neutral","Positive","Sentence Text"])
            for i, book in enumerate(tqdm(text)):
                for j, sentence in enumerate(book):
                    writer.writerow([books[i], len(sentence), compounds[i][j], negatives[i][j], neutrals[i][j], positives[i][j]," ".join(word for word in sentence)])



# run the main method analyzin sentiment, semantic, and similarity for both lists of text documents
main(bible_file_names, bible_book_names, "Bible_analysis")
main(quran_file_names, quran_book_names, "Quran_analysis")











