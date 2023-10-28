# -*- coding: utf-8 -*-
"""
Author: Dominik Jurek

DATE: 10/27/2023
METHOD: Imports extracted claim texts for training data and 
        build training and control data set for the NLP method.
"""

#######################################################
#   Load Environment
#######################################################

import pandas as pd
import numpy as np
import re
import os
import shutil


import pickle

import requests
from io import BytesIO
import zipfile

import multiprocessing as mp

# Set seed
RANDOM_SEED = 42

# Number or cores
CORES = mp.cpu_count()


#####################################
# Training Data Import              #
#####################################
def training_data_import(nclasses: int = 4,
                         home_directory: str = os.getcwd(),
                         specified_uspc_class: list = ['705'],
                         use_specified_uspc_class: bool = False,
                         text_source: str = 'claim_extraction',
                         replacement: bool = False,
                         Issued_patent_control: bool = False):
    r'''
    METHOD: Import data from 'claim_text_extraction_for_training' saved in 
                text_source subdirectory.  Build balanced data frame for
                model training with valid and invalid claim texts.
    INPUT:  nclasses (int): the top n USPC classes of rejected claims to be used.
            home_directory (str): path that contains 'text_source' subfolder.
            text_source (str): path to full text claim extraction of treated and valid claims.
            specified_uspc_class (list): list of strings with specify USPC class to be used instead
                of nclasses most frequent in rejected data.  Unused if use_specified_uspc_class=False.                
            use_specified_uspc_class (bool): use list of classes in 'specified_uspc_class' instead
                of nclasses most frequent in rejected data.                
            replacement (bool): use drawing with replacement from control claims for sample balancing
                (usually only relevant when using pre-grant applications as control).
            Issued_patent_control (bool): use claim of issued patents in PatentsView as control.
    OUTPUT: training_data (pd.DataFrame): df with 'app_id', 'claim_text', 'treated' and binary variable
                identifying invalid claims.
            mainclasses (list of int): main USPC classes to be used classification
    r'''
    #############################
    # Import files              #
    #############################

    #===================================================
    # Load Claims

    text_documents = os.listdir(text_source)

    Alice_claims_files = [f for f in text_documents if \
                          bool(re.search('_Alice', f)) & \
                          ~bool(re.search('Bilski', f)) &  \
                          ~bool(re.search('ControlPatents', f))]
        # -> if we are considering all pre-Alice cases and have downloaded
        # all the relevant data, Bilski would be the starting point.


    Alice_claims = pd.DataFrame()
    for claim_file in Alice_claims_files:
        Alice_claims = pd.concat([Alice_claims, 
                                  pd.read_csv(text_source + '//' + claim_file,
                                              encoding='utf-8',
                                              low_memory=False)], \
                                 axis = 0, ignore_index = True)

    #========================================
    # Import rejections, office action, and application data from USPTO

    #----------------------------------
    # Application Data
    #----------------------------------
    # only the 2017 vintage has the application status codes that are 
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

    #----------------------------------
    # Rejection Data
    #----------------------------------
    if ('rejections_uspto_research_2017.csv' in os.listdir(USPTO_data_directory)):
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Local
        rejections = pd.read_csv(USPTO_data_directory
                                 + '/rejections_uspto_research_2017.csv', low_memory=False)
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

    #----------------------------------
    # Office Actions Data
    #----------------------------------
    if ('office_actions_uspto_research_2017.csv' in os.listdir(USPTO_data_directory)):
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Local
        office_actions = pd.read_csv(USPTO_data_directory
                                     + '/office_actions_uspto_research_2017.csv', 
                                     low_memory=False)
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


    #####################################
    # Construct training dataset        #
    #####################################

    office_actions['ifw_number'] = office_actions.ifw_number.astype(str).\
        apply(lambda s: re.sub(r'^0*', '', str(s).split('.')[0])) # internal identifier
    rejections['ifw_number'] = rejections.ifw_number.astype(str).\
        apply(lambda s: re.sub(r'^0*', '', str(s).split('.')[0]))

    rejections['app_id'] = pd.to_numeric(rejections.app_id, 
                                         downcast = 'integer', errors = 'coerce')
    office_actions['app_id'] = pd.to_numeric(office_actions.app_id, 
                                             downcast = 'integer', errors = 'coerce')

    # Office action data
    alice_rejections = pd.merge(office_actions,
                                rejections[rejections.alice_in == 1],
                                how = 'inner',
                                on = ['ifw_number', 'app_id']) 
    # => app_id not needed, but avoids duplicates columns

    print('\t\t Number of Alice Rejections Raw %.0f' % len(alice_rejections), flush=True)
   
    #====================================
    # Limit to cases with clear Alice identification
    #====================================
    # Source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3024621
    #         https://www.uspto.gov/sites/default/files/documents/Variable%20tables_v20171120.pdf
    # - Exclude mismatches between paragraph and action sentence
    # - Exclude rejection based on lack of novelty, 102
    # - Exclude rejection based on obviousness, 103
    # - Exclude rejections based on does not meet requirements
    #   regarding the adequacy of the disclosure of the invention, 112
    # - Exclude double patenting rejections
    alice_condition = (alice_rejections.rejection_fp_mismatch != 1) & \
        (alice_rejections.rejection_102 != 1) & \
        (alice_rejections.rejection_103 != 1) & \
        (alice_rejections.rejection_112 != 1) & \
        (alice_rejections.rejection_dp != 1)

    print('\t\t Number of Alice Rejections with no other injections %.0f' % len(alice_rejections[alice_condition]), flush=True)
    print('\t\t Number of unique application ids with Alice Rejections with no other injections %.0f' % len(alice_rejections[alice_condition].app_id.unique()), flush=True)

    # Merge with claim fulltext
    alice_text_rejections = pd.merge(alice_rejections[alice_condition],
                                     Alice_claims,
                                     how = 'inner',
                                     on = 'app_id')


    # Filter out observations with claims being mentioned as rejected
    alice_text_rejections['affected_claim'] = alice_text_rejections.\
        apply(lambda row: str(row['claim_int']).split('.')[0] in row['claim_numbers'].split(','), 
              axis = 1)

    alice_treated = alice_text_rejections[alice_text_rejections['affected_claim']].copy()
    alice_treated = alice_treated.drop(['rejection_101','alice_in', 'bilski_in', 'mayo_in',
                                        'myriad_in', 'dep_reference', 'affected_claim',
                                        'uspc_class', 'uspc_subclass'], axis = 1)

    print('\t\t Number of Alice Treated independent claims %.0f' % len(alice_treated), flush=True)
    print('\t\t Number of unique application ids with Alice Treated independent claims %.0f' % len(alice_treated.app_id.unique()), flush=True)
    #--------------------------------------
    # Merge with application data
    application_data['app_id'] = pd.to_numeric(application_data.application_number,
                                                   downcast = 'integer', errors = 'coerce')


    alice_application_treated = pd.merge(alice_treated,
                                         application_data,
                                         how = 'inner',
                                         on = 'app_id')


    #----------------------------------------
    # Control for pg-publication number to be correct
    alice_application_treated['pgpub_number'] = alice_application_treated.\
        pgpub_number.astype(str).apply(lambda s: re.sub(r'^0*', '', str(s).split('.')[0]))
    alice_application_treated['earliest_pgpub_number_cleaned'] = alice_application_treated.earliest_pgpub_number.astype(str). \
        apply(lambda x: re.sub(r'^US', '', x)).apply(lambda x: re.sub(r'[a-zA-Z]{1,1}\d{0,1}$', '', x))

    publication_condition = (alice_application_treated.pgpub_number == alice_application_treated.earliest_pgpub_number_cleaned) | \
                            (alice_application_treated.pgpub_number == '')

    alice_application_treated = alice_application_treated[publication_condition]
    print('\t\t Number of treated claims from the right pre-grant publication document %.0f' % len(alice_application_treated), flush=True)

    #--------------------------------------
    alice_application_treated['appl_status_code'] = pd.to_numeric(
        alice_application_treated.appl_status_code,
        errors='coerce', downcast='integer')

    alice_application_treated.appl_status_code.value_counts().head(10)
    # 150.0  => Patented Case
    # 161.0  => Abandoned -- Failure to Respond to an Office Action
    # 124.0  => On Appeal -- Awaiting Decision by the Board of Appeals
    # 41.0   => Non Final Action Mailed
    # 163.0  => Abandoned --
    # 61.0   => Final Rejection Mailed
    # 30.0   => Docketed New Case - Ready for Examination
    # 93.0   => Notice of Allowance Mailed -- Application Received in Office of Publications
    # 71.0   => Response to Non-Final Office Action Entered and Forwarded to Examiner
    # 120.0  => Notice of Appeal Filed

    # Filter for out final rejections and abandonment
    type_condition = (alice_application_treated['appl_status_code'].isin([161, 163, 61]))

    alice_application_treated = alice_application_treated[type_condition]
    print('\t\t Number of treated claims from abandoned application status %.0f' % len(alice_application_treated), flush=True)

    #--------------------------------------
    # Restrict to reasonable dates
    alice_application_treated['filing_date_dt'] = pd.to_datetime(
        alice_application_treated.filing_date, errors='coerce')
    alice_application_treated['mail_dt_dt'] = pd.to_datetime(
        alice_application_treated.mail_dt, errors='coerce')
    alice_application_treated['appl_status_date_dt'] = pd.to_datetime(
        alice_application_treated.appl_status_date, errors='coerce')
    alice_application_treated['earliest_pgpub_date_dt'] = pd.to_datetime(
        alice_application_treated.earliest_pgpub_date, errors='coerce')

    date_condition = (alice_application_treated['mail_dt_dt'] > alice_application_treated['filing_date_dt']) & \
                        (alice_application_treated['appl_status_date_dt'] > alice_application_treated['mail_dt_dt']) & \
                            (alice_application_treated['earliest_pgpub_date_dt'] > alice_application_treated['filing_date_dt'])

    alice_application_treated = alice_application_treated[date_condition]
    print('\t\t Number of treated claims purged for illogical dates %.0f' % len(alice_application_treated), flush=True)

    #-----------------------------
    # Remove cancelled claims
    cancelled_condition = alice_application_treated.claim_text.\
            apply(lambda x: not(bool(re.search(r'canceled', re.sub('[^A-Za-z]','', x))) & bool(len(re.sub('[^A-Za-z]','', x)) < 20)))

    alice_application_treated = alice_application_treated[cancelled_condition]
    print('\t\t Number of uncancelled treated claims %.0f' % len(alice_application_treated), flush=True)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if Issued_patent_control==True:
        print('\t Use for control claim construction the PatentsView claim texts', flush=True)
        #======================================
        # Control Claims using Application Extraction
        #======================================

        # Load Control claims
        text_documents = os.listdir(text_source)
        Alice_Issued_patent_control_files = [f for f in text_documents if \
                                                  bool(re.search('_Alice', f)) & \
                                                      bool(re.search('ControlPatents', f)) & \
                                                        ~bool(re.search('Bilski', f)) &  \
                                                          ~bool(re.search('PatentsView_PregrantApp', f))]

        Alice_Issued_patent_control = pd.DataFrame()
        for claim_file in Alice_Issued_patent_control_files:
            Alice_Issued_patent_control = pd.concat(
                [Alice_Issued_patent_control,
                 pd.read_csv(text_source + '//' + claim_file,
                             encoding='utf-8',
                             low_memory=False)], \
                    axis = 0, ignore_index = True)

        # Use patent number that are integers for utility patents
        Alice_Issued_patent_control['patent_num'] = pd.to_numeric(
            Alice_Issued_patent_control.patent_id, \
                downcast = 'integer', errors = 'coerce')
        application_data['patent_num'] = pd.to_numeric(
            application_data.patent_number, \
                downcast = 'integer', errors = 'coerce')

        # Already controlled for being patented cases and no office actions being associated
        alice_text_controls = pd.merge(application_data,
                                       Alice_Issued_patent_control,
                                       how = 'inner',
                                       on = 'patent_num')

        print('\t\t Number of control claims from PatentsView %.0f' % len(alice_text_controls), flush=True)

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    else:
        print('\t Use for control claim construction the application claim texts', flush=True)

        # Load Control claims
        text_documents = os.listdir(text_source)
        Alice_ApplicationFile_control_claims_files = [f for f in text_documents if \
                                                      bool(re.search('_Alice', f)) & \
                                                          bool(re.search('ControlPatents', f)) & \
                                                            ~bool(re.search('Bilski', f)) &  \
                                                              bool(re.search('PatentsView_PregrantApp', f))]

        Alice_ApplicationFile_control_claims = pd.DataFrame()
        for claim_file in Alice_ApplicationFile_control_claims_files:
            Alice_ApplicationFile_control_claims = pd.concat(
                [Alice_ApplicationFile_control_claims,
                 pd.read_csv(text_source + '//' + claim_file,
                             encoding='utf-8',
                             low_memory=False)], \
                    axis = 0, ignore_index = True)


        Alice_ApplicationFile_control_claims['app_id'] = pd.to_numeric(
            Alice_ApplicationFile_control_claims.app_id, \
                downcast = 'integer', errors = 'coerce')

        alice_ApplicationFile_text_controls = pd.merge(
            application_data,
            Alice_ApplicationFile_control_claims,
            how = 'inner',
            on = 'app_id')

        # Already controlled for being patented cases and no office actions being associated
        print('\t\t Number of raw control claims from Application Files %.0f' % len(alice_ApplicationFile_text_controls), flush=True)

        #------------------------------------
        # Remove cancelled claims
        cancelled_condition = alice_ApplicationFile_text_controls.claim_text.\
            apply(lambda x: not(bool(re.search(r'canceled', re.sub('[^A-Za-z]','', x))) & bool(len(re.sub('[^A-Za-z]','', x)) < 20)))

        alice_ApplicationFile_text_controls = alice_ApplicationFile_text_controls[
            cancelled_condition]
        print('\t\t Number of uncancelled control claims from Application Files %.0f' % len(alice_ApplicationFile_text_controls), flush=True)

        #------------------------------------
        # Control for pg-publication number to be correct
        alice_ApplicationFile_text_controls['pgpub_number'] = alice_ApplicationFile_text_controls.\
            pgpub_number.astype(str).\
            apply(lambda s: re.sub(r'^0*', '', str(s).split('.')[0]))
        alice_ApplicationFile_text_controls['earliest_pgpub_number_cleaned'] = alice_ApplicationFile_text_controls.\
            earliest_pgpub_number.astype(str). \
            apply(lambda x: re.sub(r'^US', '', x)).\
                apply(lambda x: re.sub(r'[a-zA-Z]{1,1}\d{0,1}$', '', x))

        publication_control_patent_condition = (alice_ApplicationFile_text_controls.pgpub_number == alice_ApplicationFile_text_controls.earliest_pgpub_number_cleaned) | \
                                                (alice_ApplicationFile_text_controls.pgpub_number == '')

        alice_text_controls = alice_ApplicationFile_text_controls[publication_control_patent_condition]

        alice_text_controls = alice_text_controls.drop(['claim_num'], axis=1)
        print('\t\t Number of control claims with right pgpub file from Application Files %.0f' % len(alice_text_controls), flush=True)

    #======================================
    # Sample balancing
    #======================================

    alice_application_treated = alice_application_treated.\
        drop_duplicates(['claim_text', 'app_id', 'earliest_pgpub_number'])
    alice_text_controls = alice_text_controls.\
        drop_duplicates(['claim_text', 'app_id', 'earliest_pgpub_number'])
    print(' \t Alice unique treated claims before class selections: \n' + str(len(alice_application_treated)), flush=True)
    print(' \t Unique control claims before class selections: \n' + str(len(alice_text_controls)), flush=True)

    #--------------------------------------
    # Restrict to same classes as controls

    alice_application_treated['uspc_class_str'] = alice_application_treated.\
        uspc_class.astype(str).\
        apply(lambda s: re.sub(r'^0*', '', str(s).split('.')[0]))

    alice_application_treated.uspc_class_str.value_counts().head(10)
    # Source: https://www.uspto.gov/web/patents/classification/selectnumwithtitle.htm
    # 705    => Data processing: financial, business practice, management, or cost/price determination
    # 463    => Amusement devices: games
    # 702    => Data processing: measuring, calibrating, or testing
    # 434    => Education and demonstration
    # 709    => Electrical computers and digital processing systems: multicomputer data transferring
    # 273    => Amusement devices: games
    # 716    => Computer-aided design and analysis of circuits and semiconductor masks
    # 703    => Data processing: structural design, modeling, simulation, and emulation
    # 435    => Chemistry: molecular biology and microbiology
    # 701    => Data processing: vehicles, navigation, and relative location

    print('\t\t Count of Alice classes\n' +
          str(alice_application_treated.uspc_class_str.value_counts().head(10)), flush=True)

    print('\t\t Cumsum of Alice classes\n' +
          str(alice_application_treated.uspc_class_str.value_counts(normalize=True).head(10).cumsum()), flush=True)

    alice_text_controls['uspc_class_str'] = alice_text_controls.uspc_class.astype(str).\
        apply(lambda s: re.sub(r'^0*', '', str(s).split('.')[0]))
    alice_text_controls.uspc_class_str.value_counts().head(10)

    #----------------------------------------
    # Select control and treated patents from top n main classes or specific given classes
    if use_specified_uspc_class==True:
        main_classes = list(set(specified_uspc_class))
    else:
        main_classes = list(set(alice_application_treated.uspc_class_str.value_counts(). \
                                nlargest(nclasses).reset_index()['index']))
    print('\t USPC main classes for patent: \n' + str(main_classes), flush=True)

    alice_text_treated = alice_application_treated[
        alice_application_treated.uspc_class_str.isin(main_classes)].copy()
    alice_text_controls = alice_text_controls[
        alice_text_controls.uspc_class_str.isin(main_classes)].copy()
    print('\t\t Number of treated claims in main class %.0f and control claims %.0f' % (len(alice_text_treated), len(alice_text_controls)), flush=True)

    #--------------------------------------
    # cast dates
    alice_text_controls['filing_date_dt'] = pd.to_datetime(
        alice_text_controls.filing_date, errors='coerce')
    alice_text_controls['appl_status_date_dt'] = pd.to_datetime(
        alice_text_controls.appl_status_date, errors='coerce')
    alice_text_controls['patent_issue_date_dt'] = pd.to_datetime(
        alice_text_controls.patent_issue_date, errors='coerce')

    # Restrict control patents to those granted after the Alice decision
    alice_decision_date = np.datetime64('2014-06-19')

    timed_alice_text_controls = alice_text_controls[
        alice_text_controls.patent_issue_date_dt > alice_decision_date].copy()
    print('\t\t Number of control claims issued after the Alice decision %.0f' % len(timed_alice_text_controls), flush=True)
    print(' \t Control USPC Classes count: \n' + str(timed_alice_text_controls.uspc_class_str.value_counts()), flush=True)
    print(' \t Control Filing years count: \n' + str(timed_alice_text_controls.filing_date_dt.dt.year.value_counts()), flush=True)

    #-------------------------------------------
    # Resample to match uspc and filing year distribution as the treated claims
    alice_text_treated['filing_data_year'] = alice_text_treated.filing_date_dt.dt.year
    alice_groups = alice_text_treated.groupby(['uspc_class_str', 'filing_data_year']).size().reset_index(name='count')

    print(' \t Alice USPC Classes count: \n' + str(alice_text_treated.uspc_class_str.value_counts(normalize=True)), flush=True)
    print(' \t Alice Filing years count: \n' + str(alice_text_treated.filing_date_dt.dt.year.value_counts(normalize=True)), flush=True)
    print(' \t Alice year and type counts: \n' + str(alice_groups), flush=True)

    #--------------------------------------------
    # Create relative frequency by filing year and uspc class
    alice_groups['relative_freq'] = alice_groups['count'] / alice_groups['count'].agg(sum)

    timed_alice_text_controls['filing_data_year'] = timed_alice_text_controls.filing_date_dt.dt.year
    timed_alice_text_controls = pd.merge(timed_alice_text_controls,
                                        alice_groups,
                                        on=['uspc_class_str', 'filing_data_year'],
                                        how='inner')


    # resample control patents by weight, allower for replace to have more balance in set
    sample_length = len(alice_text_treated)
    balance_alice_text_controls = timed_alice_text_controls.sample(
        n=sample_length,
        weights=timed_alice_text_controls.relative_freq,
        random_state=RANDOM_SEED,
        replace=replacement)

    balance_alice_text_controls[['uspc_class_str']].value_counts(normalize=True)
    alice_text_treated[['uspc_class_str']].value_counts(normalize=True)

    balance_alice_text_controls[['filing_data_year']].value_counts(normalize=True)
    alice_text_treated[['filing_data_year']].value_counts(normalize=True)


    balance_alice_text_controls = balance_alice_text_controls.\
        rename(columns={'claim_number':'claim_int'})
    balance_alice_text_controls['unique_claim_id'] = balance_alice_text_controls.\
        apply(lambda x: str(x['app_id']) + '_' + str(x['claim_int']), axis=1)

    print(' \t Unique App_id + Claim Number repetitions in balanced controls: \n' +
        str(balance_alice_text_controls.unique_claim_id.value_counts().head(10)), flush=True)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print(' \t Alice USPC Classes control group: \n' + str(balance_alice_text_controls[['uspc_class_str']].value_counts(normalize=True)), flush=True)
    print(' \t Alice USPC Classes treated group: \n' + str(alice_text_treated[['uspc_class_str']].value_counts(normalize=True)), flush=True)

    print(' \t Alice Filing years control group: \n' + str(balance_alice_text_controls[['filing_data_year']].value_counts(normalize=True)), flush=True)
    print(' \t Alice Filing years treated group: \n' + str(alice_text_treated[['filing_data_year']].value_counts(normalize=True)), flush=True)

    print(' \t Count Alice USPC Classes control group: \n' + str(balance_alice_text_controls[['uspc_class_str']].value_counts()), flush=True)
    print(' \t Count Alice USPC Classes treated group: \n' + str(alice_text_treated[['uspc_class_str']].value_counts()), flush=True)

    print(' \t Count Alice Filing years control group: \n' + str(balance_alice_text_controls[['filing_data_year']].value_counts()), flush=True)
    print(' \t Count Alice Filing years treated group: \n' + str(alice_text_treated[['filing_data_year']].value_counts()), flush=True)


    print(' \t Alice year and type counts for balanced control group: \n' +
          str(balance_alice_text_controls.groupby(['uspc_class_str', 'filing_data_year']).size().\
              reset_index(name='count')), flush=True)

    #=======================================
    # Construct dataframe with treated and untreated claims

    treated_claims_text = alice_text_treated[['app_id', 'claim_text']].copy()
    treated_claims_text['treated'] = 1

    control_claims_text = balance_alice_text_controls[['app_id', 'claim_text']].copy()
    control_claims_text['treated'] = 0

    training_data = pd.concat([treated_claims_text, control_claims_text], \
                              axis = 0, ignore_index = True)

    # Cleaning the entries and remove digits from the beginning
    training_data['claim_text'] = training_data.apply(lambda row: \
                                                      re.sub(r'^\d{1,3}\.{0,1}\s', \
                                                             '', row['claim_text']), \
                                                          axis = 1)
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print(' \t Alice treated data length: ' + str(len(alice_text_treated)), flush=True)
    print(' \t Unbalanced Control data length: ' + str(len(timed_alice_text_controls)), flush=True)
    print(' \t Training data length: ' + str(len(training_data)), flush=True)

    
    return(training_data, main_classes)

#########################
# Build for testing
#########################
if __name__ == '__main__':
    # For the build, we can define different variations for control data.
    # For illustration, I keep below in the comment the specifications
    # of different data construction methods.  Thus, I set the model version
    # of the file names to the origin of the control data.

    # Note, 'issued_patents_control' version using patent claims for controls should
    #   use non-resampling in the training dataset construction, while 
    # 'ApplicationControls' version should use resampling (smaller number of eligible controls)

    # Change the working directory
    home_directory = '/Users/dominikjurek/Library/CloudStorage/Dropbox/University/PhD Berkeley/Research/Alice Project/NLP Patent Classification/Alice NLP Python Code/Testing Github files'
    os.chdir(home_directory)

    # Define directory for other data that are not PatentsView
    USPTO_data_directory = 'USPTO_raw_data'
    if not os.path.exists(USPTO_data_directory):
        os.makedirs(USPTO_data_directory)

    # how many classes should be included in the training dataset
    TOP_NCLASSES = 4

    output_version='issued_patents_control'
    
    model_data, main_classes = training_data_import(
        nclasses=TOP_NCLASSES,
        home_directory=os.getcwd(),
        specified_uspc_class='xxx',
        use_specified_uspc_class=False,
        text_source='claim_extraction',
        replacement=False,
        Issued_patent_control=True)
    
    
    r'''
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    output_version='ApplicationControls'
    
    model_data, main_classes = training_data_import(
        nclasses=TOP_NCLASSES,
        home_directory=os.getcwd(),
        specified_uspc_class='xxx',
        use_specified_uspc_class=False,
        text_source='claim_extraction',
        replacement=True,
        Issued_patent_control=False)
    
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    output_version='PatentsControls_only705'
    
    model_data, main_classes = training_data_import(
        nclasses=1,
        home_directory=os.getcwd(),
        specified_uspc_class=['705'],
        use_specified_uspc_class=True,
        text_source='claim_extraction',
        replacement=False,
        Issued_patent_control=True)
    
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    output_version='ApplicationControls_only705'
    
    model_data, main_classes = training_data_import(
        nclasses=1,
        home_directory=os.getcwd(),
        specified_uspc_class=['705'],
        use_specified_uspc_class=True,
        text_source='claim_extraction',
        replacement=True,
        Issued_patent_control=False)
    r'''
    
    #======================================
    # Save the model data and the main categories as .pkl files
    model_data.to_pickle('model_data_' + str(output_version) + '.pkl')
    
    with open('main_classes_' + str(output_version) + '.pkl', 'wb') as fp:
        pickle.dump(main_classes, fp)
    
    
    
    
