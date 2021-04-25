# Religious-text-NLP
Just some fun sentiment and semantic analyses using doc2vec

Data includes:
1.	Number of verses per book
2.	Number of words per verse
3.	Sentiment averages – (un)weighted based on number of words per book
4.	Semantic density – average of word vector lengths
5.	Cosine similarity matrix of all books

Novel differences and conclusions:
1.	Using doc2vec instead of summing word2vec outputs allows for better comparison of whole books.
2.	Bible
    1.	Book-averaged sentiments quantified many expected results
        1.	Most negative books in the Old Testament are Lamentations, Obadiah, and Nahum
        2.	Most positive books in the Old Testament are Song of Solomon and Daniel
        3.	The New Testament being more positive, especially after the 4 gospels
        4.	Most positive books in the New Testament are 2 John and 3 John
    2.	Semantic density showed no obvious trends
    3.	Similarity matrix showed many expected groupings
        1.	The Torah and earlier Protocanonical books did not group as well, except for Ezra and Nehemiah, which are often grouped into one book
        2.	Job through Malachi grouped very well with the exception of Daniel
        3.	Matthew-John and somewhat Acts grouped and were distinct from Old and other New Testament books
        4.	The rest of the New testament grouped very well except for Revelations
3.	Quran
    1.	Words per verse was sporadic in the traditional order (which is in approximately descending order by book length) and Egyptian standard order (which is an adaption of Noldeke’s work) but was nearly constant and then nearly linearly increasing in Noldeke’s order.
    2.	Book-averaged sentiments showed Noldeke’s ordering to be the smoothest and therefore likely more accurate than other orders.
    3.	Semantic density (average word vector length i.e. plain word2vec without document vectors) showed Noldeke’s ordering again to be the smoothest and with increasing density until the middle of the third Meccan surahs and then slightly decreasing during the remaining Medinan surahs.
    4.	Similarity matrix (averaged from 16x different translations) showed that Noldeke’s ordering better clusters than the Egyptian standard again indicating its better accuracy. Meccan and Medinan surahs are visually cluster and are distinct between each other. The first, second, and third Meccan surahs also appear somewhat distinguishable.

Caveats:
1.	Only English translations were used, so some context may have been lost. Combining results from multiple translations hopefully fixed some of this.
2.	The doc2vec model did not ignore any words based on their frequency, so hapax legomena were included.
3.	Stemming/lemmatizing functions used here before doc2vec or sentiment analysis do not know old second- and third-person singular conjugations often used in these texts (e.g., thou thinkest and he thinketh).
4.	Sentiment analysis functions use dictionaries of known words, so it may have missed many old words used in these texts. Averaging whole books, however, showed significant differences in sentiment.
5.	Whole-book document vector lengths are intended to represent a version of semantic density, but they positively, nonlinearly correlate with number of words and verses per book, indicating that traditional word2vec word semantic density averaging is better.

Other fun things to try:
1.	In what contexts are different Biblical names of God used?
2.	Plot dimensionality reduced embeddings for the 99 Names of God in the Quran
3.	Initial results didn’t turn out well, but improve a hierarchical clustering algorithm of the Quran’s similarity matrix to see if a better ordering can be found
4.	Find verse-level shared narratives between the Bible and Quran by thresholding cosine similarities
5.	Other biblical authorship and timeline questions
6.	Attempt to correlate surah sentiment and context with other life events (persecution, battles, etc.)
7.	Attempt to quantify legitimacy of hadiths by similarity to the Quran
8.	TBD!

References
1.	Word2vec
    1.	Mikolov T, Chem K, Corrado G, Dean J. Efficient estimation of word representations in vector spaces. arXiv. 2013 Sep. https://arxiv.org/abs/1301.3781
    2.	Mikolov T, Sutskever I, Chen K, Corrado G, Dean J. Distributed representations of words and phrases and the compositionality. arXiv. 2013 Oct. https://arxiv.org/abs/1310.4546
2.	Doc2vec
    1.	Le Q, Mikolov T. Distributed representations of sentences and documents. arXiv. 2014 May. https://arxiv.org/abs/1405.4053
    2.	Rehurek R, Sojka P. Software framework for topic management with large corpora. Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks. 2010 May. https://is.muni.cz/publication/884893/en
3.	16x Quran translations – https://tanzil.net/trans/
4.	Noldeke and Egyptian standard orderings
    1.	Reynolds G. The Qur’an in its historical context. Routledge. 2007 Sep. https://doi.org/10.4324/9780203939604
