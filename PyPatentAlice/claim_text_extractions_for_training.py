# -*- coding: utf-8 -*-
"""
    DATE: 10/27/2023
    AUTHOR: Dominik Jurek
    METHOD: Collect from USPTO websites the relevant patent information to 
            build the classification model
"""


#################################################################
# Load Packages
#################################################################

import pandas as pd
import numpy as np
import re
import os
import shutil

import requests
from io import BytesIO
#from lxml import html

import zipfile
import csv

import multiprocessing as mp


#----------------------------------------
# Expand field limit to iterate through all claims
import ctypes
csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))
csv.field_size_limit()


#=========================================================================
# Yearly pre-grant application claim extraction
#=========================================================================
def application_claim_PatentView(year: int, 
                                 app_df: pd.DataFrame, 
                                 output_path: str, 
                                 output_name: str = ''):
    '''
    METHOD: Load application claim texts from PatentsView pre-grant application data
            for given prublication year.
    INPUT:  year (int): publication year for claim texts.
            app_df (pd.DataFrame): relevant applications based on .
                PatentsView's Pre-grant Applications with 'document_number' and 'app_id'.
            output_path (str): path to output directory.
            output_name (str): string added to the end of the outpuf file name.
    OUTPUT: PatentsView_PregrantApp_claim_extraction (pd.DataFrame): extracted claims for publication year
                saved in output_path; keys should be the 'appl_id' and 'document_number'.
    RETURN: None
    '''
    
    #------------------------------
    print('\t Search application claims for year ' + str(year), flush=True)
    try:
        
        patent_claim_master_url = r'https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/pg_claims_{0}.tsv.zip'

        url_link_list = patent_claim_master_url.format(year)
        
        
        # Wrap around limited amount of retrys
        for request_attempt in range(5):
            r = requests.get(url_link_list)
            # Check if no error and length is correct
            if (r.ok == True) & \
               (len(r.content) == int(r.headers['Content-Length'])):
               break

        z = zipfile.ZipFile(BytesIO(r.content))

        app_claims = pd.read_csv(z.open(z.infolist()[0].filename), delimiter="\t", 
                                 quoting=csv.QUOTE_NONNUMERIC, low_memory=False)

        # rename some columns to be consistent with other dataframes
        app_claims = app_claims.rename(columns={'pgpub_id':'document_number'})
        
        #------------------------------------------
        # Limit observations to independent claims that are in the app DF (wihtout dependency)
        indep = (app_claims.dependent.isnull())
        indep_app_claims = app_claims[indep]

        #------------------------------------------
        # Cleaning the entries and remove digits from the beginning
        indep_app_claims.loc[:, 'claim_text'] = indep_app_claims.claim_text.\
            astype(str).apply(lambda x: \
                              re.sub(r'^\d{1,3}\s{0,}\.{0,1}\s', '', \
                                     x).strip())

        #------------------------------------------
        # Further control for independent claims following https://www.uspto.gov/sites/default/files/documents/patent_claims_methodology.pdf
        # And check via reg expression if there is a reference to a different claim
        indep_app_claims.loc[:, 'dep_reference'] = indep_app_claims.claim_text.\
            apply(lambda x: bool(re.search(r'\bclaim\s+\d+\b|\bclaims\s+\d+\b', str(x))))
        indep_app_claims = indep_app_claims[~indep_app_claims.dep_reference]

        #------------------------------------------
        # Select applications which are in the search application dataframe
        indep_app_claims = indep_app_claims[~indep_app_claims.document_number.isnull()]

        searched_indep_app_claims = indep_app_claims.merge(
            app_df, on='document_number', how='inner')
        searched_indep_app_claims.reset_index(inplace=True, drop=True)

        #--------------------------------------------------
        # Rename and typcasting        
        searched_indep_app_claims = searched_indep_app_claims.rename(
            columns={'claim_sequence':'claim_num',
                     'document_number':'pgpub_number',
                     'filing_date_dt':'filing_date_PregrantApp_dt', 
                     'filing_year':'filing_year_PregrantApp'})
        
        searched_indep_app_claims.loc[:, 'claim_int'] = pd.to_numeric(
            searched_indep_app_claims.claim_num, errors='coerce')
        
        #----------------------------------------
        # Output Result
        searched_indep_app_claims.to_csv(
            path_or_buf = output_path + '/PatentsView_PregrantApp_claim_extraction_'\
                + str(year) + '_' + output_name + '.csv',
            index=False, encoding = 'utf-8')

        print('\t Lenght of output DF of independent application claims for type '\
              + str(output_name) + ', year '+ str(year) + ': '\
                  + str(len(searched_indep_app_claims)), flush=True)

    except Exception as exc:
        print('\t Error in claim search for year: ' + str(year) + ' => ' + str(exc))

    return

#================================================================
# PatentsView Claim Extraction of Full Text Claims
#================================================================

def patent_claim_PatentView(year: int,
                            patent_list: list,
                            output_path: str, 
                            output_name: str):
    '''
    METHOD: Load from USPTO PatentsView claim texta for issued patents.
    INPUT:  year (int): patent issue year for claim texts.
            patent_list (list): relevant 'patent_id' as int.
            output_path (str): path to output directory.
            output_name (str): string of added to the end of the outpuf file name.
    OUTPUT: PatentsView_claim_extraction (pd.DataFrame):  extracted independent 
        claims text for 'patent_list' saved in output_path.
    RETURN: None
    '''

    #------------------------------
    # Turn patent id list into in
    patent_list = [int(i) for i in patent_list if not(np.isnan(i))]

    #------------------------------
    print('\t Search patent claims for type ' + str(output_name) + ', year ' + str(year), flush=True)
    try:
        patent_claim_master_url = 'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_{0}.tsv.zip'
        
        url_link_list = patent_claim_master_url.format(year)
        
        # Wrap around limited amount of retrys
        for request_attempt in range(5):
            r = requests.get(url_link_list)
            # Check if no error and length is correct
            if (r.ok == True) & \
               (len(r.content) == int(r.headers['Content-Length'])):
               break

        z = zipfile.ZipFile(BytesIO(r.content))

        patent_claims = pd.read_csv(z.open(z.infolist()[0].filename), delimiter="\t", 
                                    quoting=csv.QUOTE_NONNUMERIC, low_memory=False)

        # rename some columns to be consistent with other dataframes
        patent_claims = patent_claims.rename(columns={'pgpub_id':'document_number'})

        #------------------------------------------
        # Limit observations to independent claims that are in the patent DF
        indep = (patent_claims.dependent.isnull())|(patent_claims.dependent==-1)|(patent_claims.dependent=='-1')
        indep_patent_claims = patent_claims[indep]

        #------------------------------------------
        # Cleaning the entries and remove digits from the beginning
        indep_patent_claims.loc[:, 'claim_text'] = indep_patent_claims.claim_text.astype(str).\
            apply(lambda x: \
                  re.sub(r'^\d{1,3}\.{0,1}\s', '', \
                         x).strip())

        #------------------------------------------
        # Further control for independent claims following https://www.uspto.gov/sites/default/files/documents/patent_claims_methodology.pdf
        # And check via reg expression if there is a reference to a different claim
        indep_patent_claims.loc[:, 'dep_reference'] = indep_patent_claims.claim_text.\
            apply(lambda x: bool(re.search(r'\bclaim\s+\d+\b|\bclaims\s+\d+\b', str(x))))
        indep_patent_claims = indep_patent_claims[~indep_patent_claims.dep_reference]

        #------------------------------------------
        # Select patents which are in the searched classes
        # Note that I focus on utility patents, which have as identifier an integer
        # See: https://www.uspto.gov/patents-application-process/applying-online/patent-number#:~:text=A%20Patent%20Number%20is%20assigned,six%2C%20seven%20or%20eight%20digits.
        indep_patent_claims['patent_id'] = pd.to_numeric(indep_patent_claims.patent_id,
                                                         downcast = 'integer', errors = 'coerce')
        indep_patent_claims = indep_patent_claims[~indep_patent_claims.patent_id.isnull()]

        searched_indep_patents = indep_patent_claims[indep_patent_claims.patent_id.isin(patent_list)]
        searched_indep_patents.reset_index(inplace=True, drop=True)
      
        #----------------------------------------
        # Output Result
        searched_indep_patents.to_csv(
            path_or_buf = output_path + '/PatentsView_claim_extraction_'\
                + str(year) + '_' + output_name + '.csv',
                index=False, encoding = 'utf-8')
        print('\t Lenght of output DF of independent claims for type ' + str(output_name) +
              ', year '+ str(year) + ': ' + str(len(searched_indep_patents)), flush=True)

    except Exception as exc:
        print('\t Error in claim search for year: ' + str(year) + ' => ' + str(exc))

    return


########################################################################
# Extraction of control claims that are patented
########################################################################

def control_patent_claim_fulltext_extraction(rejections_data: pd.DataFrame, 
                                             application: pd.DataFrame, 
                                             nclasses: int, 
                                             output_path: str, 
                                             min_year: int = 2010, 
                                             max_year: int = 2010, 
                                             output_name: str = ''):
    '''
    METHOD: Extracts from USPTO PatentsView claim texts from successful applications in same cohort
            as rejection application, thus identifying eligble claims in same USPC classes
            and without office actions.
    INPUT:  rejections_data ((pd.DataFrame)): identified rejections from 
                'Office Action Research Dataset for Patents'.
            application (pd.DataFrame): based on PatentsView's Pre-grant Applications.
            nclasses (int): number of USPC classes to be considered based on how rejection frequency.
            output_path (str): path to output directory.
            min_year (int): first year of application pre-grant publication to be extracted.
            max_year (int): last year of application pre-grant publication to be extracted.
            output_name (str): string of added to the end of the outpuf file name.
    OUTPUT: Control_Patent_claim_extraction (pd.DataFrame): extracted claims for 
        pre-grant publication year saved in output path.
    RETURN: None
    '''
    #================================================
    # Step 1: Gather data on patent that can serve as controls
    print('\t Load relevant data', flush=True)
    #----------------------------------
    # Application Data
    #----------------------------------
    # Note: only the 2017 vintage has the application status codes that are 
    # easy to use in this verison (office action dataset also ends in 2017)
    if ('application_data_2017.csv' in os.listdir(USPTO_data_directory)):
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Local
        application_data = pd.read_csv(USPTO_data_directory
                                       + '/application_data_2017.csv', low_memory=False)
    else:
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Load application data from economic research dataset

        # Wrap around limited amount of retrys
        for request_attempt in range(5):
            r = requests.get(r"https://bulkdata.uspto.gov/data/patent/pair/economics/2017/application_data.csv.zip")
            if (r.ok == True) & \
               (len(r.content) == int(r.headers['Content-Length'])):
               break

        z = zipfile.ZipFile(BytesIO(r.content))
        z.infolist()[0].filename = 'application_data_2017.csv'
        z.extract(z.infolist()[0])

        application_data = pd.read_csv(z.open(z.infolist()[0]), low_memory=False)

        shutil.move('application_data_2017.csv', 
                    USPTO_data_directory + '/application_data_2017.csv')


    #==================================================
    # Step 2: Get office actions to find patent that are non-affected
 
    #----------------------------------
    # Office Actions Data
    #----------------------------------
    if ('office_actions_uspto_research_2017.csv' in os.listdir(USPTO_data_directory)):
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Local
        office_actions = pd.read_csv(USPTO_data_directory 
                                     + '/office_actions_uspto_research_2017.csv', low_memory=False)
    else:
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Load rejection and office data

        for request_attempt in range(5):
            r = requests.get(r"https://bulkdata.uspto.gov/data/patent/office/actions/bigdata/2017/office_actions.csv.zip")
            if (r.ok == True) & \
               (len(r.content) == int(r.headers['Content-Length'])):
               break

        z = zipfile.ZipFile(BytesIO(r.content))
        z.infolist()[0].filename = 'office_actions_uspto_research_2017.csv'
        z.extract(z.infolist()[0])

        office_actions = pd.read_csv(z.open(z.infolist()[0]), low_memory=False)

        shutil.move('office_actions_uspto_research_2017.csv', 
                    USPTO_data_directory + '/office_actions_uspto_research_2017.csv')

    
    #==============================================================
    # Step 3: Merge and find main USPC classes for the respective years and non-affected applications

    rejections_data['app_id'] = pd.to_numeric(rejections_data.app_id, 
                                              downcast = 'integer', errors = 'coerce')
    office_actions['app_id'] = pd.to_numeric(office_actions.app_id, 
                                             downcast = 'integer', errors = 'coerce')
    application_data['app_id'] = pd.to_numeric(application_data.application_number, 
                                               downcast = 'integer', errors = 'coerce')


    rejections_application_date = pd.merge(application_data,
                                            rejections_data,
                                            how = 'inner',
                                            on = ['app_id'])

    # Convert date to dateobjects
    rejections_application_date.filing_date = pd.to_datetime(
        rejections_application_date.filing_date, errors = 'coerce')


    application_data.filing_date = pd.to_datetime(application_data.filing_date, 
                                                  errors = 'coerce')

    #=================================================================
    # Step 4: Create control data which have no recorded office action and are within the same cohorts
    print('\t Control patent contruction', flush=True)
    application_control_data = application_data[
        ~application_data.app_id.isin(list(office_actions.app_id))\
            & (application_data.filing_date.dt.year.isin(
                list(rejections_application_date.filing_date.dt.year)))].\
        copy()

    #------------------------------------------
    # Define main categories
    rejections_application_date['uspc_class_str'] = rejections_application_date.\
        uspc_class.astype(str).apply(lambda s: re.sub(r'^0*', '', str(s).split('.')[0]))

    print('\tUSPC main classes in rejections:')
    print(rejections_application_date['uspc_class_str'].value_counts(ascending=False).\
          nlargest(10)) #-> restrict to the 10 top classes, this can be set to nclasses to view all 
                        # selected patent classes. 

    uspc_main_category = list(set(rejections_application_date['uspc_class_str'].value_counts(ascending=False). \
                                  nlargest(nclasses).reset_index()['index']))

    #------------------------------------------
    # Filter control data to classes most affected

    application_control_data['uspc_class_str'] = application_control_data.uspc_class.astype(str).\
        apply(lambda s: re.sub(r'^0*', '', str(s).split('.')[0]))
    application_control_data = application_control_data[
        application_control_data.uspc_class_str.isin(uspc_main_category)]

    # Restrict to patented cases
    # Appl_status_codes is integer, see https://www.uspto.gov/sites/default/files/documents/Appendix%20A.pdf
    application_control_data['appl_status_code'] = pd.to_numeric(
        application_control_data.appl_status_code, errors='coerce', downcast='integer')
    control_patents = application_control_data[
        application_control_data.appl_status_code == 150]
    # application status code 150 is for patented cases

    print('\t Number of extractable control applications from PatentsView Pre-grant applications: '\
          + str(len(control_patents)), flush=True)
    
    # Find document numbers from PatentsView Pre-grant application data
    application_with_doc_num = application[
        application.app_id.isin(control_patents.app_id)][
            ['app_id', 'document_number', 'filing_date_dt', 'filing_year']].drop_duplicates()

    #===============================================
    # Step 5: Download claims for granted control patents

    # Use the same method as the main claim extraction to get the control claims 
    # note, above the app_id's are coerced to int

    r'''
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Parallel Execution -> not recommended for desktop machines
    cores = mp.cpu_count()
    print('\t\t Number of Cores: ' + str(cores))

    pool = mp.Pool(cores)
    # Run the scraping method for the contents required
    for year in range(min_year, max_year+1):
        print('\t Start control claim extraction from PatentsView Pre-grant application for '\
              + str(output_name) + ' and year ' + str(year) + '\n', flush=True)
        pool.apply_async(
                        application_claim_PatentView,
                        args=(
                              year,
                              application_with_doc_num,
                              output_path,
                              'PatentsView_PregrantApp_ControlPatents_' + output_name
                              )
                        )
    pool.close()
    pool.join()
    r'''
    
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Linear Execution
    # Run the scraping method for the contents required
    for year in range(min_year, max_year+1):
        print('\t Start control claim extraction from PatentsView Pre-grant application for '\
              + str(output_name) + ' and year ' + str(year) + '\n', flush=True)
        application_claim_PatentView(year,
                                     application_with_doc_num,
                                     output_path,
                                     'PatentsView_PregrantApp_ControlPatents_' + output_name)
                        
    
    
    #====================================
    # Step 6: PatentsView Claims extraction if application with publication number and date after
    #           Alice and no 101 office action
    # Limit to control claim applications without 101 rejections Alice identification
    #====================================
    # Source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3024621
    #         https://www.uspto.gov/sites/default/files/documents/Variable%20tables_v20171120.pdf

    # Application data preparations
    # Focusing on  utility patents, which have as identifier an integer
    # See: https://www.uspto.gov/patents-application-process/applying-online/patent-number#:~:text=A%20Patent%20Number%20is%20assigned,six%2C%20seven%20or%20eight%20digits.
    application_data['patent_num'] = pd.to_numeric(application_data.patent_number,
                                                   downcast = 'integer', errors = 'coerce')
    application_data['patent_issue_date_dt'] = pd.to_datetime(
        application_data.patent_issue_date, errors = 'coerce')

    application_data['uspc_class_str'] = application_data.uspc_class.astype(str).\
        apply(lambda s: re.sub(r'^0*', '', str(s).split('.')[0]))

    #----------------------------------------
    # Define conditions for patents
    # Restrict control patents to those granted after the Alice decision
    alice_decision_date = np.datetime64('2014-06-19')

    # Restriction to patents without 101 office actions
    alice_101_condition = (office_actions.rejection_101 == 1)
    office_actions_101_app_id_list = list(office_actions[alice_101_condition]['app_id'])

    # Find patents with:
    # - No 101 office action
    # - same filing cohort as rejected claims
    # - patent issue date after alice decision
    # - uspc classes in main classes
    application_patentView_control_data = application_data[
        ~(application_data.app_id.isin(office_actions_101_app_id_list)) & \
            (application_data.filing_date.dt.year.isin(list(rejections_application_date.filing_date.dt.year))) & \
                (application_data.patent_issue_date_dt > alice_decision_date) & \
                    (application_data.uspc_class_str.isin(uspc_main_category))
                    ].copy()

    #---------------------------------------------------------
    # Restrict to patented cases with patent number
    application_patentView_control_data['appl_status_code'] = pd.to_numeric(
        application_patentView_control_data.appl_status_code,
        errors='coerce', downcast='integer')

    control_PatentsView_patents = application_patentView_control_data[
        application_patentView_control_data.appl_status_code == 150] # code 150: issued patents

    control_PatentsView_patents = control_PatentsView_patents[
        ~control_PatentsView_patents.patent_num.isnull()]

    print('\t Number of extractable control patent from PatentsView: '\
          + str(len(control_PatentsView_patents)), flush=True)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # PatentsView Extraction Routine
    cores = mp.cpu_count()
    print('\t\t Number of Cores: ' + str(cores))

    #-------------------------------------------------------
    # restrict scraping to minimum and maximum year issue years of control patents
    patentsView_first_year = min(control_PatentsView_patents.patent_issue_date_dt.dt.year)
    patentsView_last_year = max(control_PatentsView_patents.patent_issue_date_dt.dt.year)

    r'''
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Parallel execution -> not recommended for desktop machines
    pool = mp.Pool(cores)
    # Run the scraping method for the contents required
    for year in range(patentsView_first_year, patentsView_last_year+1):
        print('\t Start control claim extraction from PatentsView for ' + str(output_name) +
              ' and year ' + str(year) + '\n', flush=True)
        pool.apply_async(
                        patent_claim_PatentView,
                        args=(
                              year,
                              list(control_PatentsView_patents.patent_num),
                              output_path,
                              'PatentsView_ControlPatents_' + output_name
                              )
                        )
    pool.close()
    pool.join()
    r'''
    
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Linear executions
    for year in range(patentsView_first_year, patentsView_last_year+1):
        print('\t Start control claim extraction from PatentsView for ' + str(output_name) +
              ' and year ' + str(year) + '\n', flush=True)
        patent_claim_PatentView(year,
                                list(control_PatentsView_patents.patent_num),
                                output_path,
                                'PatentsView_ControlPatents_' + output_name)
                        

    return

########################################################################
# Main Executions
########################################################################
if __name__ == '__main__':

    print('Start main routine')
    ####################################
    # Parameter and Input Definition
    ####################################
    # Application publication years to be checked 
    # Recommended is 2005 to 2023, 
    # for testing purposes restrict to 2013 and 2014, around the Alice decision
    min_year_global = 2013
    max_year_global = 2014

    #----------------------------------------
    # Define the home directory.  Subfolders will be created here and data are
    # directly saved in here
    home_directory = os.getcwd()
    os.chdir(home_directory)


    # Define name of the directoy which stores raw data from PatentsView
    PatentsView_directory = 'PatentsView_raw_data'

    if not os.path.exists(PatentsView_directory):
        os.makedirs(PatentsView_directory)


    # Define directory for USPTO research data and public PAIR data
    # includes raw data that are not PatentsView
    USPTO_data_directory = 'USPTO_raw_data'
    if not os.path.exists(USPTO_data_directory):
        os.makedirs(USPTO_data_directory)

    #----------------------------------------
    # Define output path
    output_path = 'claim_extraction'

    #====================================
    # Create Output Path if not already exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #====================================
    # Input Data

    # Use office action research dataset from Chief Economist of USPTO
    # source: https://www.uspto.gov/learning-and-resources/electronic-data-products/office-action-research-dataset-patents
    #----------------------------------
    # Rejection Data
    #----------------------------------
    if ('rejections_uspto_research_2017.csv' in os.listdir(USPTO_data_directory)):
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Local
        rejections = pd.read_csv(USPTO_data_directory
                                 +'/rejections_uspto_research_2017.csv', low_memory=False)
    else:
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Load rejection and office data

        for request_attempt in range(5):
            r = requests.get(r"https://bulkdata.uspto.gov/data/patent/office/actions/bigdata/2017/rejections.csv.zip")
            if (r.ok == True) & \
               (len(r.content) == int(r.headers['Content-Length'])):
               break

        z = zipfile.ZipFile(BytesIO(r.content))
        z.infolist()[0].filename = 'rejections_uspto_research_2017.csv'
        z.extract(z.infolist()[0])

        rejections = pd.read_csv(z.open(z.infolist()[0]), low_memory=False)

        shutil.move('rejections_uspto_research_2017.csv', 
                    USPTO_data_directory + '/rejections_uspto_research_2017.csv')


    #====================================
    # Define Extraction Claims in reject data for any of the critical court decision
    condition_rejections = (rejections.alice_in == 1)|(rejections.bilski_in == 1)|\
        (rejections.mayo_in == 1)|(rejections.myriad_in==1)
    bilski_to_alice_rejections = rejections[condition_rejections].copy()

    #----------------------------------------
    #   Application data from Patent View
    #----------------------------------------
    if ('application_pregrant_publication.tsv' in os.listdir(PatentsView_directory)):
            #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Local
            application = pd.read_csv(
                PatentsView_directory + '/application_pregrant_publication.tsv', 
                delimiter="\t", quoting=csv.QUOTE_NONNUMERIC, low_memory=False)
    else:
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Load application data from economic research dataset

        # Wrap around limited amount of retrys
        for request_attempt in range(5):
            r = requests.get(r'https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/pg_published_application.tsv.zip' , stream=True)
            if (r.ok == True) & \
                   (len(r.content) == int(r.headers['Content-Length'])):
                   break

        z = zipfile.ZipFile(BytesIO(r.content))
        z.infolist()[0].filename = 'application_pregrant_publication.tsv'
        z.extract(z.infolist()[0])
        
        application = pd.read_csv(z.open(z.infolist()[0]), delimiter="\t", \
                                  quoting=csv.QUOTE_NONNUMERIC, low_memory=False)
        shutil.move('application_pregrant_publication.tsv', 
                    PatentsView_directory + '/application_pregrant_publication.tsv')

    application['document_number'] = application.pgpub_id
    application['app_id'] = pd.to_numeric(application.application_id,
                                            downcast = 'integer', errors = 'coerce')
    application['filing_date_dt'] = pd.to_datetime(application['filing_date'], 
                                                   errors='coerce')
    application['filing_year'] = application.filing_date_dt.dt.year
    
    r'''
    ####################################
    # Function Execution
    ####################################
    
    #---------------------------------------------
    # Bilski to Alice
    #---------------------------------------------
    # Note: we can chose to focus only on Alice or also extract data for 
    # additional cases prior to Alice.  For testing, I comment out here 
    # the routines that include other cases

    print('Start control Claim Full Text extraction for Bilski, Mayo, Myriad, Alice')
    #===================================
    # Full text control patent claim extraction (internally parallel)

    control_patent_claim_fulltext_extraction(
                                             rejections_data = bilski_to_alice_rejections,
                                             application = application,
                                             nclasses = 100,
                                             output_path = output_path,
                                             min_year = min_year_global,
                                             max_year = max_year_global,
                                             output_name = 'Bilski_to_Alice'
                                             )

    print('End control Claim Full Text extraction for Bilski, Mayo, Myriad, Alice')
    #===================================
    # Full text claim extraction


    #---------------------------------------------
    # Bilski to Alice
    #---------------------------------------------
    print('Start rejected Claim Full Text extraction for Bilski, Mayo, Myriad, Alice')
    # Coerce to app_id to integer => all relevant app_ids are digits
    bilski_to_alice_rejections['app_id_int'] = pd.to_numeric(
        bilski_to_alice_rejections.app_id,
        downcast = 'integer', errors = 'coerce')

    # Find document numbers from PatentsView Pre-grant application data
    bilski_to_alice_rejections_application_with_doc_num = application[
        application.app_id.isin(list(bilski_to_alice_rejections.app_id_int))][
            ['app_id', 'document_number', 'filing_date_dt', 'filing_year']].drop_duplicates()
    
    print('\t Number of extractable application with rejections by Bilski, Mayo, Myriad, Alice: '\
          + str(len(bilski_to_alice_rejections_application_with_doc_num)), flush=True)

    
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Parallel Execution
    cores = mp.cpu_count()
    print('\t\t Number of Cores: ' + str(cores))

    pool = mp.Pool(cores)
    # Run the scraping method for the contents required
    for year in range(min_year_global, max_year_global+1):
        print('\t Start rejected claim extraction for Bilski to Alice for year ' + str(year) + '\n')
        pool.apply_async(
                        application_claim_PatentView,
                        args=(
                              year,
                              bilski_to_alice_rejections_application_with_doc_num,
                              output_path,
                              'Bilski_to_Alice'
                              )
                        )
    pool.close()
    pool.join()


    print('End rejected Claim Full Text extraction for Bilski, Mayo, Myriad, Alice')

    r'''
    ####################################
    # Function Execution Alice only
    ####################################

    alice_condition = (rejections.alice_in == 1)
    alice_rejections = rejections[alice_condition].copy()


    print('Start control Claim Full Text extraction for Alice only')
    #===================================
    # Full text control patent claim extraction 

    #---------------------------------------------
    # Alice only
    #---------------------------------------------
    control_patent_claim_fulltext_extraction(
                                             rejections_data = alice_rejections,
                                             application = application,
                                             nclasses = 50,
                                             output_path = output_path,
                                             min_year = min_year_global,
                                             max_year = max_year_global,
                                             output_name = 'Alice'
                                             )

    print('End control Claim Full Text extraction for Alice only')

    #===================================
    # Full text claim extraction
    print('Start rejected Claim Full Text extraction for Alice only')

    #---------------------------------------------
    # Alice only
    #---------------------------------------------
    # Coerce to app_id to integer => all relevant app_ids are digits

    alice_rejections['app_id_int'] = pd.to_numeric(alice_rejections.app_id, 
                                                   downcast = 'integer', errors = 'coerce')


    # Find document numbers from PatentsView Pre-grant application data
    alice_rejections_application_with_doc_num = application[
        application.app_id.isin(list(alice_rejections.app_id_int))][
            ['app_id', 'document_number', 'filing_date_dt', 'filing_year']].drop_duplicates()
    
    print('\t Number of extractable application with rejections by Alice only: '\
          + str(len(alice_rejections_application_with_doc_num)), flush=True)

    r'''
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Parallel Execution -> not recommended for desktop machines
    cores = mp.cpu_count()
    print('\t\t Number of Cores: ' + str(cores))

    pool = mp.Pool(cores)
    # Run the scraping method for the contents required
    for year in range(min_year_global, max_year_global+1):
        print('\t Start rejected claim extraction for Alice only for year '\
              + str(year) + '\n')
        pool.apply_async(
                        application_claim_PatentView,
                        args=(
                              year,
                              alice_rejections_application_with_doc_num,
                              output_path,
                              'Alice'
                              )
                        )
    pool.close()
    pool.join()

    print('End rejected Claim Full Text extraction for Alice only')
    r'''
    
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Linear Execution
    for year in range(min_year_global, max_year_global+1):
        print('\t Start rejected claim extraction for Alice only for year '\
              + str(year) + '\n')
        application_claim_PatentView(year,
                                     alice_rejections_application_with_doc_num,
                                     output_path,
                                     'Alice')                        
 
    print('End main routine')

