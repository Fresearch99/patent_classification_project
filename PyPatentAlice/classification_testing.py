#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Dominik Jurek

DATE: 10/22/2023
METHOD: Use classification outputs from 'patent_claim_classification.py' and
        test if we identify the most affected patents.
"""
#######################################################
#   Load Environment
#######################################################

import pandas as pd
import numpy as np
import re
import os
import shutil

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from statsmodels.formula.api import poisson
from statsmodels.formula.api import negativebinomial
from linearmodels import PanelOLS    


import requests
from io import BytesIO
import zipfile
import csv


home_directory = os.getcwd()
os.chdir(home_directory)

# Set seed
RANDOM_SEED = 42

#---------------------------------------------------
# Define PatentsView directory
PatentsView_directory = 'PatentsView_raw_data'

##############################################################
# Build Analysis Dataset
##############################################################
def analysis_df_build():
    '''Load and merge data on patent issuance, CPC groups, and predicted treatment status'''
    print('Load PatentsView patent and current CPC data', flush=True)
    #-------------------------------
    # patent data
    #-------------------------------
    if ('patent.tsv' in os.listdir(PatentsView_directory)):
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Load from directory
        patent_data = pd.read_csv(PatentsView_directory + '/patent.tsv', delimiter="\t", 
                             quoting=csv.QUOTE_NONNUMERIC, low_memory=False)
    else:
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Load patent data from Patent View
    
        # Wrap around limited amount of retrys
        for request_attempt in range(5):
            r = requests.get(r"https://s3.amazonaws.com/data.patentsview.org/download/g_patent.tsv.zip")
            if (r.ok == True) & \
               (len(r.content) == int(r.headers['Content-Length'])):
               break
        z = zipfile.ZipFile(BytesIO(r.content))
        z.infolist()[0].filename = 'patent.tsv'
        z.extract(z.infolist()[0])
    
        patent_data = pd.read_csv(z.open(z.infolist()[0]), delimiter="\t", 
                                  quoting=csv.QUOTE_NONNUMERIC, low_memory=False)
    
        shutil.move('patent.tsv', PatentsView_directory + '/patent.tsv')
    
    
    # Restrict to information that need
    #patent_data = patent_data[['number', 'date', 'type', 'title', 'num_claims']].drop_duplicates()
    
    # Typecasting, since utility patents have numeric identifiers we also control
    # for reissues, etc.
    patent_data['patent_id'] = pd.to_numeric(patent_data.patent_id,
                                             downcast = 'integer', errors = 'coerce')
    
    patent_data = patent_data[(~patent_data.patent_id.isnull()) & (patent_data.withdrawn==0)]
    
    patent_data['issue_date_dt'] = pd.to_datetime(patent_data['patent_date'])
    
    # Add issue quarter and year
    patent_data['issue_year'] = patent_data.issue_date_dt.dt.year
    patent_data['issue_quarter'] = patent_data.issue_date_dt.dt.quarter
    
    # Define also beginning-of-period dates, since this will make plotting easier later
    patent_data['quarter_issue_date_dt'] = patent_data['issue_date_dt']+pd.offsets.QuarterEnd(1)
    
    
    patent_df = patent_data[
        ['patent_id', 'issue_date_dt', 'issue_year', 'issue_quarter', 'quarter_issue_date_dt']
        ].drop_duplicates()
    
    #-------------------------------
    # cpc classifications
    #-------------------------------
    if ('cpc_current_PatentsView.tsv' in os.listdir(home_directory)):
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Local
        cpc_current = pd.read_csv('cpc_current_PatentsView.tsv', delimiter="\t", 
                                  quoting=csv.QUOTE_NONNUMERIC, low_memory=False)
    else:
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Load application data from Patent View
    
        # Wrap around limited amount of retrys
        for request_attempt in range(5):
            r = requests.get(r"https://s3.amazonaws.com/data.patentsview.org/download/g_cpc_current.tsv.zip")
            if (r.ok == True) & \
               (len(r.content) == int(r.headers['Content-Length'])):
               break
    
        z = zipfile.ZipFile(BytesIO(r.content))
        z.infolist()[0].filename = 'cpc_current_PatentsView.tsv'
        z.extract(z.infolist()[0])
    
        cpc_current = pd.read_csv(z.open(z.infolist()[0]), delimiter="\t", 
                                  quoting=csv.QUOTE_NONNUMERIC, low_memory=False)
    
    
    #-------------------------------------
    # Focus on primary categories
    cpc_current = cpc_current[cpc_current.cpc_type=='inventional']
    
    cpc_current = cpc_current[cpc_current.cpc_sequence==0]
    
    # Drop unneeded columns and make cpc groups unique
    cpc_current = cpc_current.drop(['cpc_section',
                                    'cpc_type',
                                    'cpc_class', 
                                    'cpc_sequence'], axis=1).drop_duplicates().\
        rename(columns={'cpc_subclass':'group_id',
                        'cpc_group':'subgroup_id'})
    
    # Cast id to int
    cpc_current['patent_id'] = pd.to_numeric(cpc_current.patent_id,
                                             downcast='integer', errors='coerce')
    
    
    # =============================================================================
    # Load classification outputs and defined patent-level treatment
    # =============================================================================
    # Focus here patents in the affected CPC groups since the USPC-based 
    # classification is more limited and USPC was discontinued after 2013
    
    print('Load Alice classification outcomes and define treated patents', flush=True)
    output_directory = r'patent_classification'
    
    # Load Alice patent classification
    patent_classification_df = pd.read_csv(
        output_directory 
        + '/FullText__patents_cpcAffected__predicted__TFIDF_poly2_issued_patents_control.csv', 
        usecols=['patent_id', 'claim_sequence', '0', '1', 'predicted_label'],
        low_memory=False)
    
    # Remove duplicates that may have occured during classification (e.g., reissues)
    patent_classification_df=patent_classification_df.drop_duplicates()
    
    #---------------------------------
    # Define affected patents as thos where the first claim is treated.
    # The first claim is the most important claim since it is usually the broadest claim in the patent
    first_claim = patent_classification_df[patent_classification_df.claim_sequence==0][
        ['patent_id', 'predicted_label']].drop_duplicates().\
        rename(columns={'predicted_label':'Treated'})
    
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print('Merge and output patent and classification data', flush=True)
    # Define the main dataframe for analysis
    df = patent_df.merge(
        cpc_current, on='patent_id', how='inner').merge(
            first_claim, on='patent_id', how='inner').reset_index(drop=True)

    return(df)
        
##############################################################
# Analysis
##############################################################
if __name__ == '__main__':
    # Load data for analysis
    df = analysis_df_build()

    # Aggregate to CPC group - quarter - treatment counts
    agg_counts = df.groupby(
        ['group_id', 'quarter_issue_date_dt', 'Treated'])['patent_id'].\
        count().reset_index().rename(columns={'patent_id':'Count'})

    alice_decision_date = np.datetime64('2014-06-19')
    agg_counts['Post'] = (agg_counts['quarter_issue_date_dt'] > alice_decision_date).astype(int)


    # Normalize the count 1 at the beginning of 2012
    norm_counts = agg_counts[
        ['group_id', 'quarter_issue_date_dt', 'Treated', 'Count']].\
        sort_values('quarter_issue_date_dt')

    first = (norm_counts.drop('quarter_issue_date_dt', axis=1).\
            groupby(['group_id', 'Treated']).transform('first'))
    norm_counts['normalized_Count'] = norm_counts['Count'] / first['Count']


    # Create plot
    sns.set_style("whitegrid")
    sns.set_palette("dark")

    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(data=norm_counts,
                        y='normalized_Count',
                        x='quarter_issue_date_dt',
                        hue='Treated',
                        markers=True,
                        err_style='bars',
                        errorbar='ci')

    ax.axvline(x = alice_decision_date.__hash__(),
            color='r', linestyle='--') 
    ax.set(xlabel='Issue Date', 
        ylabel='Normalized Count',
        title='Count Issued Patents in Affected CPC Groups')

    plt.legend(title = 'Predicted Status', loc='upper left', labels=['Control', 'Treated'])

    # Plot if needed
    # plt.savefig('time_series_treated_control_issuances.png', dpi=300)
    
    plt.show()

    

    #---------------------------------
    # Regression models

    # Log is useful to estimate percentage changes
    agg_counts['log_Count'] = np.log(agg_counts['Count'])

    reg_did = sm.ols(formula='log_Count ~ Treated * Post',
                    data=agg_counts).fit(
                        cov_type='cluster',
                        cov_kwds={'groups': agg_counts['group_id']})
    print(reg_did.summary())    


    panel_df = agg_counts.copy()
    panel_df = panel_df.set_index(['group_id', 'quarter_issue_date_dt'])
    reg_fe = PanelOLS.from_formula(
        formula='log_Count ~ Treated * Post + EntityEffects + TimeEffects',
        data=panel_df, drop_absorbed=True).fit(
                cov_type='clustered', cluster_entity=True, cluster_time=False)
    print(reg_fe)    



    reg_poisson = poisson(formula='Count ~ Treated * Post',
                        data=agg_counts).fit()
    reg_negbin = negativebinomial(formula='Count ~ Treated * Post',
                                data=agg_counts).fit()


    print(reg_poisson.summary())
    print(reg_negbin.summary())


    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #Save regression output
    
    # Replicate the Panel OLS with statsmodel to make saving as regression
    # output easier
    reg_fe_ols = sm.ols(
        formula='log_Count ~ Treated * Post + C(group_id) + C(quarter_issue_date_dt)',
        data=agg_counts).fit(
            cov_type='cluster',
            cov_kwds={'groups': agg_counts['group_id']})
    print(reg_fe_ols.summary())    # -> same coefficient estimates as with Panel model.


    from statsmodels.iolib.summary2 import summary_col
    reg_table = summary_col(
        [reg_did, reg_fe_ols, reg_poisson, reg_negbin], 
                stars=True, 
                float_format="%.3f",
                info_dict={
                    'N': lambda x: "%#6d" % x.nobs,
                    'R-squared:': lambda x: "%#8.3f" % x.rsquared,
                    'Adj. R-squared:': lambda x: "%#8.3f" % x.rsquared_adj,
                    'Pseudo R-squared:': lambda x: "%#8.3f" % x.prsquared},
                regressor_order=['Intercept', 'alpha', 'Post', 'Treated', 'Treated:Post'])
    # Note, this table contains all the coefficients for the fixed effects form 
    # the statsmodel version of the Panel model.  Remove those by hand from 
    # the table when transfering into LaTex.
    
    with open('regression_models_DiD.tex', 'w') as f:
        f.write(reg_table.as_latex())
        f.close()

    #----------------------------------
    # Save sample data
    agg_counts.to_csv('example_data_for_analysis.csv', index=False)
    df.to_csv('example_data_classified_patents.csv', index=False)
    
    # Load Alice patent classification
    patent_classification_df = pd.read_csv(
        'patent_classification' 
        + '/FullText__patents_cpcAffected__predicted__TFIDF_poly2_issued_patents_control.csv', 
        usecols=['patent_id', 'claim_sequence', '0', '1', 'predicted_label'],
        low_memory=False)

    
    patent_classification_df.to_csv('example_data_claim_classification.csv', index=False)


    