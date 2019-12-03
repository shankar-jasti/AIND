import math
import statistics
import warnings
import logging

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            logging.debug("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except ValueError:
            logging.warning("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_bic_score = float('inf')
        best_states = 0
        hyper_alpha = 1.482
        for nStates in list(range(self.min_n_components,self.max_n_components+1)):
            try:
                model = self.base_model(nStates)
                if model is None:
                    return None
                logL = model.score(self.X, self.lengths)
                p = (model.startprob_.size-1) + (model.transmat_.size-1) + (model.means_.size) + (model.covars_.diagonal().size)
                bic_score = -2*logL + hyper_alpha * p * np.log(len(self.X))

                if bic_score < best_bic_score:
                    best_bic_score = bic_score
                    best_states = nStates
                    logging.debug("BIC better score for {} with {} states".format(self.this_word, nStates))
            except ValueError:
                logging.debug("BIC ValueError on {} with {} states".format(self.this_word, nStates))
                
        if best_states == 0:
            return None
        return self.base_model(best_states)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('-inf')
        best_states = 0
        hyper_alpha = 1.4
        for nStates in list(range(self.min_n_components,self.max_n_components+1)):
            try:
                model = self.base_model(nStates)
                if model is None:
                    return None
                others_mean_score = np.mean([model.score(*self.hwords[word]) for word in self.words if word != self.this_word])
                word_score = model.score(self.X, self.lengths)
                dic_score = word_score - hyper_alpha * others_mean_score
                if dic_score > best_score:
                    best_score = dic_score
                    best_states = nStates
                    logging.debug("DIC better score for {} with {} states".format(self.this_word, nStates))
            except ValueError:
                logging.debug("DIC ValueError on {} with {} states".format(self.this_word, nStates))
        if best_states == 0:
            return None
        return self.base_model(best_states)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('-inf')
        if len(self.sequences) == 1:
            isSplit = False
        else:
            isSplit = True
            nFolds = min(3,len(self.sequences))
            split_method = KFold(n_splits=nFolds)
        for nStates in list(range(self.max_n_components,self.max_n_components+1)):
            try:
                cv_score = 0
                if isSplit:
                    for train_idx, test_idx in split_method.split(self.sequences):
                        train_X, train_lengths = combine_sequences(train_idx, self.sequences)
                        test_X, test_lengths = combine_sequences(test_idx, self.sequences)
                        cv_model = GaussianHMM(n_components=nStates, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                        cv_score += cv_model.score(test_X, test_lengths)
                    avg_score = cv_score/nFolds 
                else:
                    cv_model = GaussianHMM(n_components=nStates, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    avg_score = cv_model.score(self.X, self.lengths)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_nStates = nStates
                    logging.debug("CV better score for {} with {} states".format(self.this_word, nStates))
            except ValueError:
                logging.debug("CV ValueError on {} with {} states".format(self.this_word, nStates))
                return self.base_model(nStates)
        return self.base_model(best_nStates)
                
