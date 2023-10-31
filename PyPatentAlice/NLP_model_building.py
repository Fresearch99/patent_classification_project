# -*- coding: utf-8 -*-
"""
Author: Dominik Jurek

DATE: 10/27/2023
METHOD: Imports data set and main USPC classes from 'training_data_build.py'
        and fit NLP classification for patent claims.
"""

#######################################################
#   Load Environment
#######################################################

import pandas as pd
import numpy as np
import os

import pickle


import multiprocessing as mp

import wordcloud
import matplotlib.pyplot as plt

# Set seed
RANDOM_SEED = 42

# Number or cores
CORES = mp.cpu_count()

######################################
#   Sklearn Model Building
######################################

#=====================================
# Package Loading
from sklearn.model_selection import train_test_split    

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import preprocess_string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

#------------------------------------------
# source: https://scikit-learn.org/stable/modules/svm.html#svm-classification
from sklearn.svm import SVC
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    
from sklearn.preprocessing import Normalizer
# from sklearn.preprocessing import FunctionTransformer

from sklearn.ensemble import RandomForestClassifier
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    
from sklearn.pipeline import Pipeline

from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve

from sklearn.model_selection import RandomizedSearchCV

# Model saving
import joblib

# Output of report
import json

#------------------------------------------
# Various Model types
from sklearn.linear_model import LogisticRegression
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from sklearn.naive_bayes import MultinomialNB
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
        
from sklearn.tree import DecisionTreeClassifier
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    
from sklearn.neighbors import KNeighborsClassifier
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

from sklearn.linear_model import SGDClassifier
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

from sklearn.ensemble import AdaBoostClassifier
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
        
from sklearn.ensemble import GradientBoostingClassifier
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    
#------------------------------------------
# Source: https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
#         https://medium.com/wisio/a-gentle-introduction-to-doc2vec-db3e8c0cce5e
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#-----------------------------------------
# Doc2Vec Support Class
# source: https://medium.com/swlh/a-text-classification-approach-using-vector-space-modelling-doc2vec-pca-74fb6fd73760
#         https://stackoverflow.com/questions/50278744/pipeline-and-gridsearch-for-doc2vec
from sklearn.base import BaseEstimator


#====================================
# Tokenizer and Stopword definition
def tokenizer(text):
    tokens = word_tokenize(text)
    stems = []
    for item in tokens:
        if (len(item) > 2):
            stems.append(PorterStemmer().stem(item))
    return stems


# Create Stopwords list
stop_words = set(stopwords.words("english"))

# Add preprocessed stopwords to fit tokenization
add_stopword = []
for item in stop_words:
    add_stopword.extend(tokenizer(item))
add_stopword = set(add_stopword)

stop_words.update(add_stopword)

# Custom tokenizer
def custom_tokenizer(text):
    return [w for w in tokenizer(text) if ~bool(w in stop_words)]


#=================================================
# Helper function for output validation
#=================================================
def _validation_output(model, 
                       model_name, 
                       output_version, 
                       output_directory, 
                       X_test, 
                       y_test):
    '''Internal function generating validation outputs for model, return dict with scores'''

    print('Model Validation for: ' + str(model_name))
    print('Confusion Matrix :')
    cm = confusion_matrix(y_test, model.predict(X_test), labels=[0,1], normalize='true')

    print(cm)
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=['Alice valid', 'Alice invalid'],
        cmap=plt.cm.Blues,
        normalize='true')
    disp.ax_.set_title("Normalized confusion matrix")
    disp.ax_.grid(False)

    plt.savefig(output_directory + '//' + model_name 
                + '_confusion_matrix_'+str(output_version)+'.jpg',
                bbox_inches='tight')

    #************************************************
    mcc = matthews_corrcoef(y_test, model.predict(X_test))
    print('MCC : %.2f' % mcc)
    f1_s = f1_score(y_test, model.predict(X_test), average='micro') 
    # -> if use 'binary', will calculate average only for positive labels
    print('F1 Score : %.2f' % f1_s)
    precision_s = precision_score(y_test, model.predict(X_test), average='micro')
    print('Precision Score : %.2f' % precision_s)
    recall_s = recall_score(y_test, model.predict(X_test), average='micro')
    print('Recall Score : %.2f' % precision_s)
    accuracy_s = accuracy_score(y_test, model.predict(X_test))
    print('Accuracy Score : %.2f' % accuracy_s)

    #------------------------------
    # Score dictionary for output
    score_dict = {'Precision Score': precision_s,
                  'Recall Score': recall_s,
                  'Accuracy Score': accuracy_s,
                  'F1 Score': f1_s,
                  'MCC': mcc}
    
    #------------------------------
    print('Report: ')
    print(classification_report(y_test, model.predict(X_test)))

    # Update report with MCC and F1 Score:
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    report.update(score_dict)

    #------------------------------------
    # Save updated reports
    with open(output_directory + '//' + model_name + '_performance_report_' + 
              str(output_version) + '.json', 'w') as fp:
        json.dump(report, fp)

    pd.DataFrame(report).transpose().to_csv(
        output_directory + '//' + model_name + '_performance_report_' 
        + str(output_version) + '.csv',
        float_format='%.3f', encoding='utf-8')
    pd.DataFrame(report).transpose().to_latex(
        output_directory + '//' + model_name + '_performance_report_' 
        + str(output_version) + '.tex',
        float_format='%.3f', encoding='utf-8')

    #---------------------------------------------
    print('Precision Recall Curver: ')
    y_true = y_test.to_list()
    # Control for cases without predict_proba functions,
    #       Precision functions take alternativel also decision functions
    try:
        y_pred = [l[1] for l in model.predict_proba(X_test)]
    except AttributeError:
        y_pred = list(model.decision_function(X_test))

    # Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)

    plt.figure()
    plt.step(recall, precision, where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
    'Average precision score, Alice Classification: AP={0:0.2f}'
    .format(average_precision))
    #plt.show()
    plt.savefig(output_directory + '//' + model_name 
                + '_precision_recall_curve_'+str(output_version)+'.jpg')

    #************************************************
    print('ROC Curver: ')
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    # Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Alice Classification Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(output_directory + '//' + model_name + '_ROC_curve_' 
                + str(output_version)+'.jpg')

    return(score_dict)

#################################################
# Various models with TF-IDF Matrix             #
#################################################
# Use different model for performance evalutation
def tfidf_various_build(X_train: pd.DataFrame, 
                        X_test: pd.DataFrame, 
                        y_train: pd.DataFrame, 
                        y_test: pd.DataFrame,
                        output_version: str, 
                        output_directory: str,
                        classification_model, 
                        cls_name: str):
    r'''
    METHOD: Builds Pipeline for text classification of claims using TF-IDF text
            vectors and respective classification model. 
    INPUT:  X_train / X_test (pd.DataFrame): column of claim text for training / testing.
            y_train / y_test (pd.DataFrame): column with binary treatment
                indication whether the text is ineligible (=1) or eligible (=0).
            output_directory (str): path of directory to save outputs, including performance
                reports.
            classification_model (str): sklearn-based classification model that
                can fit into a pipeline.
            cls_name (str): name of classification.
            output_version (str): atted to output name.
    OUTPUT: TFIDF_{cls_name}.joblib/TFIDF_{cls_name}.pkl: saved pipelines-objects.
            Confusion matrix, ROC curve, precision-recall curve, and model performance
                reports for each model are saved in the output_directory for each
                model.
    RETURN: tfidf_cls: pipelines-objects for classificaiton.
            score_dict (dict):  performance scores for classification model.
    r'''

    #-----------------------------------------
    # Build and fit pipeline
    tfidf_cls = Pipeline([
        ('vect', CountVectorizer(
            tokenizer = preprocess_string,
            lowercase = True,
            ngram_range = (1, 2),
            max_df = 0.5, #=> in balanced set, exclude terms frequent in both classes
            min_df = 10) # => correct for misspelling
            #max_features=1000) # -> can limit number of features if needed
                                                         ),
        ('tfidf', TfidfTransformer()),
        ('clf', classification_model),
        ])

    tfidf_cls.fit(X_train, y_train)

    #-------------------------------------
    # Validation
    score_dict = _validation_output(model=tfidf_cls,
                                    model_name='tfidf_' + str(cls_name),
                                    output_version=output_version,
                                    output_directory=output_directory,
                                    X_test=X_test,
                                    y_test=y_test)

    #-----------------------------------
    # Save Model
    joblib.dump(tfidf_cls, output_directory + '//' + 'tfidf_' + str(cls_name) + '_' + \
                str(output_version) + '.joblib')

    with open(output_directory + '//' + 'tfidf_' + str(cls_name) + '_' + \
              str(output_version) + '.pkl', 'wb') as model_file:
        pickle.dump(tfidf_cls, model_file)

    return(tfidf_cls, score_dict)


#################################################
# Support Vector Machine with TF-IDF Matrix     #
#################################################

def tfidf_svc_build(X_train: pd.DataFrame, 
                    X_test: pd.DataFrame, 
                    y_train: pd.DataFrame, 
                    y_test: pd.DataFrame, 
                    output_version: str, 
                    optimize: str = False):
    r'''
    METHOD: Builds pipeline for text classification of claims using TF-IDF text
            vectors and SVC classifier. 
    INPUT:  X_train / X_test (pd.DataFrame): column of claim text for training / testing.
            y_train / y_test (pd.DataFrame): column with binary treatment
                indication whether the text is ineligible (=1) or eligible (=0).
            output_version (str): atted to output name.
            optimize (bool): should Random Search hyperparameter tuning 
                be performed. True create also TFIDF_SVC_optimization_results output as csv.
                Generally not recommended since extends runtime significantly
    OUTPUT: Outputs are saved in new subfolder TFIDF_SVC_+'output_version
            TFIDF_SVC_Poly2_Model.joblib/TFIDF_SVC_Poly2_Model.pkl: saved pipeline objects
                for polynomial kernel of degree 2 in SVC.
            TFIDF_SVC_RBF_Model.joblib/TFIDF_SVC_RBF_Model.pkl: aved pipeline objects
                for RBF kernel in SVC.
            TFIDF_SVC_Linear_Model.joblib/TFIDF_SVC_Linear_Model.pkl: aved pipeline objects
                for linear kernel in SVC.
            Confusion matrix, ROC curve, precision-recall curve, and model performance
                reports for the model are saved in the output_directory for each
                model.
    RETURN: pipeline obkects for poly2, rbf, linear kernel
    r'''

    #--------------------------------------------
    # Create Output Directory
    output_directory = r'TFIDF_SVC_'+str(output_version)
    #====================================
    # Create Output Path if not already exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    #============================================
    def _pipeline_build_and_validation_tfidf_svc(kernel='rbf',
                                                 #C=10,
                                                 #coef0=0.1,
                                                 degree=2,
                                                 svc_name='rbf'):
        '''Wrapper for pipeline building and validation'''
        #--------------------------------------------
        # Define Pipeline for base rbf kernel
        tfidf_svc = Pipeline([
                            ('vect', CountVectorizer(
                                tokenizer = preprocess_string,
                                lowercase = True,
                                ngram_range = (1, 2),
                                max_df = 0.5,
                                min_df = 10) # => filter out misspellings
                                #max_features=1000,
                                #min_df = 10)
                                                     ),
                            ('tfidf', TfidfTransformer()),
                            ('clf', SVC(
                                verbose=0,
                                #class_weight='balanced', => already balanced training set
                                random_state=RANDOM_SEED,
                                probability=True,
                                kernel=kernel,
                                #C=C,
                                #coef0=coef0,
                                degree=degree,
                                cache_size=500)),
                            ])

        tfidf_svc.fit(X_train, y_train)

        #-------------------------------------
        # Validation
        _validation_output(tfidf_svc, 'tfidf_svc_' + str(svc_name),
                           output_version,
                           output_directory,
                           X_test,
                           y_test)

        #-----------------------------------
        # Save Model

        joblib.dump(tfidf_svc, output_directory + '//' + 'tfidf_svc_'
                    + str(svc_name) + '_' + str(output_version) + '.joblib')

        with open(output_directory + '//' + 'tfidf_svc_' + str(svc_name) + '_'
                  + str(output_version) + '.pkl', 'wb') as model_file:
          pickle.dump(tfidf_svc, model_file)

        return(tfidf_svc)

    #============================================
    # Build and definition
    
    #--------------------------------------------
    # Define Pipeline for rbf kernel
    text_rbf_svc = _pipeline_build_and_validation_tfidf_svc(
        kernel='rbf',
        #C=10,
        #coef0=0.1,
        degree=2,
        svc_name='rbf'
        )

    #--------------------------------------------
    # Define Pipeline for quadratic kernel
    text_poly2_svc = _pipeline_build_and_validation_tfidf_svc(
        kernel='poly',
        #C=10,
        #coef0=0.1,
        degree=2,
        svc_name='poly2'
        )

    #--------------------------------------------
    # Define Pipeline for linear kernel
    text_linear_svc = _pipeline_build_and_validation_tfidf_svc(
        kernel='linear',
        #C=10,
        #coef0=0.1,
        degree=2,
        svc_name='linear'
        )

    #==========================================
    # Optimization
    def _optimization():
        ''' Uses Random Grid Search to find optimal parameter settings '''

        param_grid = {
             'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
             'vect__max_df': [0.25, 0.5, 0.75],
             'vect__min_df': [0, 10, 20, 30],
             'vect__tokenizer': [preprocess_string, custom_tokenizer],

             'tfidf__use_idf': [True, False],

             'clf__C': [1e-1, 1, 1e+1, 1e+2, 1e+3],
             'clf__kernel': ['poly', 'rbf', 'linear'],
             'clf__degree': [2, 3, 4],
             'clf__coef0': np.arange(1e-2,2e-1,0.015),
             }


        gs_clf = RandomizedSearchCV(text_poly2_svc,
                              param_grid,
                              verbose=1,
                              cv=5,
                              n_iter=1000,
                              random_state=RANDOM_SEED,
                              scoring='accuracy',
                              n_jobs = -1)

        gs_clf.fit(X_train, y_train)

        #------------------------------------
        # Evaluation Output

        scores_df_clf = pd.DataFrame(gs_clf.cv_results_).\
            sort_values(by='rank_test_score')
        scores_df_clf.to_csv(path_or_buf = output_directory 
                             + '//TFIDF_SVC_optimization_results_' 
                             +  str(output_version) + '.csv',
                            encoding = 'utf-8',
                            index=False)

    #--------------------------------------
    # Optimization Execution:
    if (optimize==True):
        _optimization()

    return(text_poly2_svc, text_rbf_svc, text_linear_svc)


###########################################
# Doc2Vec + Support Vector Machine
###########################################

#----------------------------------------------
# Create Doc2Vec Class for Pipeline
#----------------------------------------------
class Doc2VecTransformer(BaseEstimator):
    '''Class for Doc2Vec text vector basis'''
    def __init__(self,  dm=0,
                        vector_size=300,
                        negative=5,
                        min_count=1,
                        alpha=0.065,
                        min_alpha=0.065,
                        epochs=5,
                        seed=RANDOM_SEED,
                        dbow_words=0,
                        workers=CORES,
                        epochs_infer=20):
         self.dm = dm
         self.vector_size = vector_size
         self.negative = negative
         self.min_count = min_count
         self.alpha = alpha
         self.min_alpha = min_alpha
         self.epochs = epochs
         self.seed = seed
         self.dbow_words = dbow_words
         self.workers = workers

         self.epochs_infer = epochs_infer

    def fit(self, x, y=None):
         # Create tagged documents
         tagged_x = [TaggedDocument(preprocess_string(v), [i]) for i, v in enumerate(x)]

         # initiate model
         self.model_dbow = Doc2Vec(tagged_x,
                                   dm=self.dm,
                                   vector_size=self.vector_size,
                                   #negative=self.negative,
                                   min_count=self.min_count,
                                   #alpha=self.alpha,
                                   #min_alpha=self.min_alpha,
                                   epochs=self.epochs,
                                   seed=self.seed,
                                   dbow_words=self.dbow_words,
                                   workers=self.workers)
         return self

    def transform(self, x):
        return np.asarray(np.array(
            [self.model_dbow.infer_vector(
                preprocess_string(v), \
                    epochs=self.epochs_infer) for i, v in enumerate(x)]))

    def fit_transform(self, x, y=None):
        self.fit(x)
        return self.transform(x)

#----------------------------------------------
# Function for model building
#----------------------------------------------
def doc2vec_svc_build(X_train: pd.DataFrame, 
                      X_test: pd.DataFrame, 
                      y_train: pd.DataFrame, 
                      y_test: pd.DataFrame, 
                      output_version: str, 
                      optimize: str = False):
    r'''
    METHOD: Builds pipeline for text classification of claims using Doc2Vec text
            vectors and SVC classifier. 
    INPUT:  X_train / X_test (pd.DataFrame): column of claim text for training / testing.
            y_train / y_test (pd.DataFrame): column with binary treatment
                indication whether the text is ineligible (=1) or eligible (=0).
            output_version (str): atted to output name.
            optimize (bool): should Random Search hyperparameter tuning 
                be performed. True create also Doc2Vec_SVC_optimization_results output as csv.
                Generally not recommended since extends runtime significantly
    OUTPUT: Outputs are saved in new subfolder Doc2Vec_SVC_+'output_version
            Doc2Vec_SVC_Poly2_Model.joblib/Doc2Vec_SVC_Poly2_Model.pkl: saved pipeline objects
                for polynomial kernel of degree 2 in SVC.
            Doc2Vec_SVC_RBF_Model.joblib/Doc2Vec_SVC_RBF_Model.pkl: aved pipeline objects
                for RBF kernel in SVC.
            Doc2Vec_SVC_Linear_Model.joblib/Doc2Vec_SVC_Linear_Model.pkl: aved pipeline objects
                for linear kernel in SVC.
            Confusion matrix, ROC curve, precision-recall curve, and model performance
                reports for the model are saved in the output_directory for each
                model.
    RETURN: pipeline obkects for poly2, rbf, linear kernel
    r'''

    #--------------------------------------------
    # Create Output Directory
    output_directory = r'Doc2Vec_SVC_'+str(output_version)

    #--------------------------------------------
    # Create Output Path if not already exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    #============================================
    def _pipeline_build_and_validation_doc2vec_svc(kernel='rbf',
                                                   #C=10,
                                                   #coef0=0.1,
                                                   degree=2,
                                                   svc_name='rbf'):
        '''Wrapper for pipeline building and validation'''
        #--------------------------------------------
        # Pipeline Construction
        doc2vec_clf = Pipeline([
                                ('doc2vec', Doc2VecTransformer(
                                    dm=0,
                                    vector_size=300,
                                    #negative=5,
                                    min_count=10, #=> correct for misspellings
                                    #alpha=0.065,
                                    #min_alpha=0.065,
                                    epochs=5,
                                    seed = RANDOM_SEED,
                                    dbow_words=0,
                                    workers=CORES,
                                    epochs_infer=20)),
                                ('normalizer',Normalizer()),
                                ('clf', SVC(verbose=0,
                                            #class_weight='balanced',
                                            random_state=RANDOM_SEED,
                                            probability=True,
                                            kernel=kernel,
                                            #C=C,
                                            #coef0=coef0,
                                            degree=degree,
                                            cache_size=500)),
                                ])

        doc2vec_clf.fit(X_train, y_train)

        #-------------------------------------
        # Validation
        _validation_output(doc2vec_clf,
                           'doc2vec_svc_' + str(svc_name),
                           output_version,
                           output_directory,
                           X_test,
                           y_test)

        #-----------------------------------
        # Save Model
        joblib.dump(doc2vec_clf, output_directory + '//' + 'doc2vec_svc_' 
                    + str(svc_name) + '_' + str(output_version) + '.joblib')

        with open(output_directory + '//' + 'doc2vec_svc_' + str(svc_name) + '_' 
                  + str(output_version)+'.pkl', 'wb') as model_file:
           pickle.dump(doc2vec_clf, model_file)

        return(doc2vec_clf)

    #============================================
    # Build and definition
    
    #--------------------------------------------
    # Define Pipeline for rbf kernel
    text_doc2vec_rbf_svc = _pipeline_build_and_validation_doc2vec_svc(
        kernel='rbf',
        #C=10,
        #coef0=0.1,
        degree=2,
        svc_name='rbf'
        )

    #--------------------------------------------
    # Define Pipeline for quadratic kernel
    text_doc2vec_poly2_svc = _pipeline_build_and_validation_doc2vec_svc(
        kernel='poly',
        #C=10,
        #coef0=0.1,
        degree=2,
        svc_name='poly2'
        )

    #--------------------------------------------
    # Define Pipeline for linear kernel
    text_doc2vec_linear_svc = _pipeline_build_and_validation_doc2vec_svc(
        kernel='linear',
        #C=10,
        #coef0=0.1,
        degree=2,
        svc_name='linear'
        )


    #----------------------------------
    # Optimization
    def _optimization_doc2vec():
        ''' Random Grid Search hyperparameter tuning for Doc2Vec Pipeline'''
        param_doc2vec_clf = {
             'doc2vec__vector_size': [100, 200, 300],
             'doc2vec__negative': [0, 5, 10, 20],
             'doc2vec__min_count': [1, 5, 10, 20],
             'doc2vec__alpha': np.arange(1e-2,1e-1,0.015),
             'doc2vec__min_alpha': np.arange(1e-2,1e-1,0.015),
             'doc2vec__epochs': [5, 10, 20],
             'doc2vec__epochs_infer': [5, 10, 20, 30],

             'clf__C': [1e-1, 1, 1e+1, 1e+2, 1e+3],
             'clf__kernel': ['poly', 'rbf', 'linear'],
             'clf__degree': [2, 3, 4],
             'clf__coef0': np.arange(-1e-2,1e-1,0.015)
             }

        gs_doc2vec_clf = RandomizedSearchCV(
            text_doc2vec_linear_svc,
            param_doc2vec_clf,
            verbose=1,
            cv=5,
            n_iter=100,
            random_state=RANDOM_SEED,
            scoring='accuracy',
            n_jobs=-1
            )

        gs_doc2vec_clf.fit(X_train, y_train)

        #------------------------------------
        # Evaluation Output
        scores_df_doc2ve_clf = pd.DataFrame(gs_doc2vec_clf.cv_results_).\
            sort_values(by='rank_test_score')
        scores_df_doc2ve_clf.to_csv(path_or_buf = output_directory 
                                    + '//Doc2Vec_SCV_optimization_results_' 
                                    + str(output_version)+'.csv',
                                    encoding = 'utf-8',
                                    index=False)

    #--------------------------------------
    # Optimization Execution:
    if (optimize==True):
        _optimization_doc2vec()

    return(text_doc2vec_rbf_svc, text_doc2vec_poly2_svc, text_doc2vec_linear_svc)


##################################################
# Helper function for word cloud visualization   #
##################################################
def wordcloud_creation(model_data, output_version, output_directory):
    '''create word clouds for training data from 
        'model_data' (with columns 'claim_text' and 'treated') in output_directory'''
    #-------------------------------------------------
    # Create word cloud of word
    # Documentation: https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#
    wc = wordcloud.WordCloud(stopwords=wordcloud.STOPWORDS,
                             background_color='white',
                             max_font_size=40,
                             color_func=lambda *args, **kwargs: "black",
                             random_state=RANDOM_SEED)

    #-----------------------------------------
    # Define treated and uncreated corpa
    treated_texts = [str(t) for t in  model_data.loc[model_data.treated==1, 'claim_text']]
    treated_text = ' '.join(treated_texts)

    untreated_texts = [str(t) for t in  model_data.loc[model_data.treated==0, 'claim_text']]
    untreated_text = ' '.join(untreated_texts)

    #-----------------------------------------
    # Generate word clouds and save
    treated_wc = wc.generate(treated_text)
    treated_wc.to_file(output_directory + '//wc_unweighted_treated_trainingData_' 
                       + str(output_version) + '.jpg')

    untreated_wc = wc.generate(untreated_text)
    untreated_wc.to_file(output_directory + '//wc_unweighted_untreated_trainingData_' 
                         + str(output_version) + '.jpg')


    #==============================================
    # Create word cloud weighted for differential frequency of words 
    # in treated and non-treated corpa

    #-----------------------------------------
    # Weight by differential frequency of terms in trated and non-treated corpa
    vectorized_model = TfidfVectorizer(use_idf=False,
                                       smooth_idf=False,
                                       stop_words=list(wordcloud.STOPWORDS))
    vectorized_corpa = vectorized_model.fit_transform([treated_text, untreated_text])

    #-----------------------------------------
    # Find relative difference between vectors (already normalized to unit length)
    differential_frequency = vectorized_corpa.toarray()[0] - vectorized_corpa.toarray()[1]

    #-----------------------------------------
    # create arrays for vector differences
    untreated_feature_frequency = []
    for f in differential_frequency:
        if f < 0:
            untreated_feature_frequency.append(-f)
        else:
            untreated_feature_frequency.append(0)
    # normalize to unit norm
    norm = np.linalg.norm(untreated_feature_frequency)
    untreated_feature_frequency = untreated_feature_frequency/norm

    # Get normalized vector for term difference of treated features
    treated_feature_frequency = []
    for f in differential_frequency:
        if f > 0:
            treated_feature_frequency.append(f)
        else:
            treated_feature_frequency.append(0)
    norm = np.linalg.norm(treated_feature_frequency)
    treated_feature_frequency = treated_feature_frequency/norm

    #-----------------------------------------
    # get terms from tfidf vector model and pair with weights
    wc_untreated_weights = {}
    for item in list(zip(vectorized_model.get_feature_names_out(), untreated_feature_frequency)):
        wc_untreated_weights[item[0]] = item[1]

    wc_treated_weights = {}
    for item in list(zip(vectorized_model.get_feature_names_out(), treated_feature_frequency)):
        wc_treated_weights[item[0]] = item[1]

    #-----------------------------------------
    # Create word cloud for both
    diff_treated_wc = wc.generate_from_frequencies(wc_treated_weights)
    diff_treated_wc.to_file(
        output_directory + '//wc_differential_frequency_weighting_treated_trainingData_' 
        + str(output_version) + '.jpg')

    diff_untreated_wc = wc.generate_from_frequencies(wc_untreated_weights)
    diff_untreated_wc.to_file(
        output_directory + '//wc_differential_frequency_weighting_untreated_trainingData_'
        + str(output_version) + '.jpg')

    return

#############################
# Main Routine              #
#############################
if __name__ == '__main__':
    # The main differences is what control set should be used.
    # For testing purposes, I restrict to using published patent claims
    # as controls.  As comments, I also include additional variations of the
    # main routine with different control set variations.

    # Change the working directory
    home_directory = os.getcwd()
    os.chdir(home_directory)
    
    output_version='issued_patents_control'
    
    model_data = pd.read_pickle('model_data_' + str(output_version) + '.pkl')

    with open('main_classes_' + str(output_version) + '.pkl', 'rb') as fp:
        main_classes = pickle.load(fp)

    r'''
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    output_version='ApplicationControls'

    model_data = pd.read_pickle('model_data_' + str(output_version) + '.pkl')

    with open('main_classes_Alice_' + str(output_version) + '.pkl', 'rb') as fp:
        main_classes = pickle.load(fp)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    output_version='issued_patents_control_only705',

    model_data = pd.read_pickle('model_data_' + str(output_version) + '.pkl')

    with open('main_classes_Alice_' + str(output_version) + '.pkl', 'rb') as fp:
        main_classes = pickle.load(fp)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    output_version='ApplicationControls_only705',

    model_data = pd.read_pickle('model_data_' + str(output_version) + '.pkl')

    with open('main_classes_Alice_' + str(output_version) + '.pkl', 'rb') as fp:
        main_classes = pickle.load(fp)
    r'''

    #-------------------------------------------
    print('Start Main Routine', flush=True)

    #-------------------------------------------
    import time
    from datetime import timedelta

    #==========================================
    # Word cloud visualization
    print('\t Data visualiziation with word cloud', flush=True)
    start_time = time.time()

    #-----------------------------------
    # Output director for word cloud
    wc_output_directory = 'Wordcloud_'+str(output_version)

    # Create WC Output Path if not already exist
    if not os.path.exists(wc_output_directory):
        os.makedirs(wc_output_directory)

    wordcloud_creation(model_data,
                       output_version=output_version,
                       output_directory=wc_output_directory)
    print("\t\tElapsed Execution time: "
          + str(timedelta(seconds=(time.time() - start_time))), flush=True)

    #=========================================
    # Split sample
    X_train, X_test, y_train, y_test = train_test_split(model_data['claim_text'],
                                                        model_data['treated'],
                                                        test_size=0.15,
                                                        random_state=RANDOM_SEED,
                                                        stratify=model_data['treated'])

    #=============================================
    # Model Validation with different types of models
    print('\t Train and Evaluate Different Types of Models with TFIDF', flush=True)
    start_time = time.time()

    #--------------------------------------------
    # Create Output Directory
    output_different_models_directory = r'TFIDF_several_models_output_'+str(output_version)
    #--------------------------------------------
    # Create Output Path if not already exist
    if not os.path.exists(output_different_models_directory):
        os.makedirs(output_different_models_directory)

    #------------------------------------------
    # Create DF from score dictionaries
    performance_df = pd.DataFrame()

    #-------------------------------------------
    # Model build and execution
    print('\t\t Logistic Regression', flush=True)
    _, score_dict = tfidf_various_build(X_train, X_test, y_train, y_test,
                                        output_version=output_version,
                                        output_directory=output_different_models_directory,
                                        classification_model=LogisticRegression(random_state=RANDOM_SEED),
                                        cls_name='logistic_regression')
    score_dict.update({'Model Name': 'Logistic Regression'})
    score = pd.DataFrame([score_dict])
    performance_df = pd.concat([performance_df, score], ignore_index=True)
    # -> replace due to capsule build from:
    # performance_df = performance_df.append(score_dict, ignore_index=True)
    

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print('\t\t Naive Bayesian', flush=True)
    _, score_dict = tfidf_various_build(X_train, X_test, y_train, y_test,
                                        output_version=output_version,
                                        output_directory=output_different_models_directory,
                                        classification_model=MultinomialNB(),
                                        cls_name='multinomial_NB')
    score_dict.update({'Model Name': 'Naive Bayes'})
    score = pd.DataFrame([score_dict])
    performance_df = pd.concat([performance_df, score], ignore_index=True)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print('\t\t SVC', flush=True)
    _, score_dict = tfidf_various_build(X_train, X_test, y_train, y_test,
                                        output_version=output_version,
                                        output_directory=output_different_models_directory,
                                        classification_model=SVC(random_state=RANDOM_SEED,
                                                                 probability=True),
                                        cls_name='svc_classifier')
    score_dict.update({'Model Name': 'Support Vector Machine'})
    score = pd.DataFrame([score_dict])
    performance_df = pd.concat([performance_df, score], ignore_index=True)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print('\t\t Decision Tree', flush=True)
    _, score_dict = tfidf_various_build(X_train, X_test, y_train, y_test,
                                        output_version=output_version,
                                        output_directory=output_different_models_directory,
                                        classification_model=DecisionTreeClassifier(random_state=RANDOM_SEED),
                                        cls_name='decision_tree_classifier')
    score_dict.update({'Model Name': 'Decision Tree'})
    score = pd.DataFrame([score_dict])
    performance_df = pd.concat([performance_df, score], ignore_index=True)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print('\t\t Random Forest', flush=True)
    _, score_dict = tfidf_various_build(X_train, X_test, y_train, y_test,
                                        output_version=output_version,
                                        output_directory=output_different_models_directory,
                                        classification_model=RandomForestClassifier(random_state=RANDOM_SEED),
                                        cls_name='random_forest_classifier')
    score_dict.update({'Model Name': 'Random Forest'})
    score = pd.DataFrame([score_dict])
    performance_df = pd.concat([performance_df, score], ignore_index=True)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print('\t\t K Neighbors', flush=True)
    _, score_dict = tfidf_various_build(X_train, X_test, y_train, y_test,
                                        output_version=output_version,
                                        output_directory=output_different_models_directory,
                                        classification_model=KNeighborsClassifier(),
                                        cls_name='k_neighbors')
    score_dict.update({'Model Name': 'K-nearest Neighbors'})
    score = pd.DataFrame([score_dict])
    performance_df = pd.concat([performance_df, score], ignore_index=True)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print('\t\t Stochastic Gradient Descent', flush=True)
    # Note: no predict_proba, only with modified huber
    _, score_dict = tfidf_various_build(X_train, X_test, y_train, y_test,
                                        output_version=output_version,
                                        output_directory=output_different_models_directory,
                                        classification_model=SGDClassifier(random_state=RANDOM_SEED),
                                        cls_name='sgd')
    score_dict.update({'Model Name': 'Stochastic Gradient Descent'})
    score = pd.DataFrame([score_dict])
    performance_df = pd.concat([performance_df, score], ignore_index=True)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print('\t\t Stochastic Gradient Descent with modified huber loss', flush=True)
    # Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
    _, score_dict = tfidf_various_build(X_train, X_test, y_train, y_test,
                                        output_version=output_version,
                                        output_directory=output_different_models_directory,
                                        classification_model=SGDClassifier(random_state=RANDOM_SEED,
                                                                           loss='modified_huber'),
                                        cls_name='sgd_modHuber')
    score_dict.update({'Model Name': 'Stochastic Gradient Descent - Modified Huber'})
    score = pd.DataFrame([score_dict])
    performance_df = pd.concat([performance_df, score], ignore_index=True)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print('\t\t AdaBoost', flush=True)
    _, score_dict = tfidf_various_build(X_train, X_test, y_train, y_test,
                                        output_version=output_version,
                                        output_directory=output_different_models_directory,
                                        classification_model=AdaBoostClassifier(random_state=RANDOM_SEED),
                                        cls_name='adaboost')
    score_dict.update({'Model Name': 'AdaBoost'})
    score = pd.DataFrame([score_dict])
    performance_df = pd.concat([performance_df, score], ignore_index=True)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print('\t\t Gradient Boosting', flush=True)
    _, score_dict = tfidf_various_build(X_train, X_test, y_train, y_test,
                                        output_version=output_version,
                                        output_directory=output_different_models_directory,
                                        classification_model=GradientBoostingClassifier(random_state=RANDOM_SEED),
                                        cls_name='gradientboosting')
    score_dict.update({'Model Name': 'Gradient Boosting'})
    score = pd.DataFrame([score_dict])
    performance_df = pd.concat([performance_df, score], ignore_index=True)

    #-----------------------------------------
    # Save performance results
    performance_df = performance_df[['Model Name', 'Precision Score', 
                                     'Recall Score', 'F1 Score', 
                                     'Accuracy Score', 'MCC']]
    performance_df.to_csv(output_different_models_directory
                          + '//performance_summary_various_models_' + str(output_version) + '.csv',
                          float_format='%.3f', encoding='utf-8', index=True)
    performance_df.to_latex(output_different_models_directory
                          + '//performance_summary_various_models_' + str(output_version) + '.tex',
                            float_format='%.3f', encoding='utf-8', index=False)

    print("\t\tElapsed Execution time: " + str(timedelta(seconds=(time.time() - start_time))), flush=True)

    #=============================================
    # SVC Model Training
    print('\t Train TF-IDF SVC Model', flush=True)
    start_time = time.time()
    tfidf_svc_build(X_train, X_test, y_train, y_test, 
                    output_version=output_version, 
                    optimize=False)
    print("\t\tElapsed Execution time: " + str(timedelta(seconds=(time.time() - start_time))), flush=True)

    print('\t Train Doc2Vec SVC Model', flush=True)
    start_time = time.time()
    doc2vec_svc_build(X_train, X_test, y_train, y_test, 
                      output_version=output_version, 
                      optimize=False)
    print("\t\tElapsed Execution time: " + str(timedelta(seconds=(time.time() - start_time))), flush=True)

    print('End Main Routine, output verison: ' + str(output_version), flush=True)

