import warnings
from asl_data import SinglesData
import logging

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    exceptions = []
    
    for idx, word in enumerate(test_set.wordlist):
        word_logL_dict = {}
        best_logL = float('-inf')
        X, lengths = test_set.get_item_Xlengths(idx)

        for guess_word, model in models.items():
            try:
                logL = model.score(X, lengths)
                word_logL_dict[guess_word] = logL
                if logL > best_logL:
                    best_logL = logL
                    best_word = guess_word
            except:
                exceptions.append(str(idx)+ ' ' +guess_word)
        probabilities.append(word_logL_dict)
        guesses.append(best_word)
        logging.debug(exceptions)
    return probabilities,guesses