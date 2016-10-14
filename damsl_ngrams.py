from swda_time import CorpusReader
from collections import deque, Counter

def main():
    
    corpus = CorpusReader('swda_time', 'swda_time/swda-metadata-ext.csv')
    counts, counts_m1 = count_n_grams(corpus,5)
    probs = ngram_model(counts,counts_m1)

    print probs
def ngram_model(counts, counts_m1):
    model = {}
    
    for ngram_str in counts.keys():
        ngram = ngram_str.split()
        model[ngram_str] = float(counts[ngram_str])/counts_m1[' '.join(ngram[:-1])]
    return model

def count_n_grams(corpus,n):
    ngram_counts = Counter()
    ngram_min1_counts = Counter()
    for trans in corpus.iter_transcripts(display_progress=False):
        ngram = deque([],n)
        for utt in trans.utterances:
            ngram.append(utt.damsl_act_tag())
            if len(ngram) == n:
                ngram_l = list(ngram)
                ngram_counts[' '.join(ngram_l)] += 1
                ngram_min1_counts[' '.join(ngram_l[:-1])] += 1

    return ngram_counts,ngram_min1_counts
        
if __name__ == '__main__':
    main()
