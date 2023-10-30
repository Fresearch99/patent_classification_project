#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR: Dominik Jurek

DATE: 2023/10/29

NETHOD: The PatnentView_raw_data and USPTO_raw_data folder of the patent_classification_project
        are too large to effectively be stored in a replication capsula.  Edit them
        by only removing excess rows.
"""

import pandas as pd
import os
import csv

home_directory = 'working directory'
os.chdir(home_directory)


PatentsView_directory = 'PatentsView_raw_data'
USPTO_data_directory = 'USPTO_raw_data'

edited_PatentsView_directory = 'edited_PatentsView_raw_data'
edited_USPTO_data_directory = 'edited_USPTO_raw_data'


if not os.path.exists(edited_PatentsView_directory):
    os.makedirs(edited_PatentsView_directory)

if not os.path.exists(edited_USPTO_data_directory):
    os.makedirs(edited_USPTO_data_directory)


global_min_year = 2010
global_max_year = 2016

global_min_data = '2010-01-01'
global_max_data = '2017-01-01'

# =============================================================================
# Edite PatentsView data
# =============================================================================
os.listdir(PatentsView_directory)
application_pregrant_publication = pd.read_csv(
                PatentsView_directory + '/application_pregrant_publication.tsv', 
                delimiter="\t", quoting=csv.QUOTE_NONNUMERIC, low_memory=False,
                dtype={'pgpub_id':'object',
                       'filing_date':'object'})

edited_application_pregrant_publication = application_pregrant_publication[
    (application_pregrant_publication.filing_date>=global_min_data) \
        & (application_pregrant_publication.filing_date<global_max_data)]

edited_application_pregrant_publication.to_csv(
    edited_PatentsView_directory + '/application_pregrant_publication.tsv', 
    sep="\t", quoting=csv.QUOTE_NONNUMERIC, index=False)    
    
# List of publication IDs that fit the descriptions
pgpub_id_list = list(set(edited_application_pregrant_publication.pgpub_id))

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    
patent = pd.read_csv(
                PatentsView_directory + '/patent.tsv', 
                delimiter="\t", quoting=csv.QUOTE_NONNUMERIC, low_memory=False,
                dtype={'patent_id':'object',
                       'patent_date':'object'})

edited_patent = patent[
    (patent.patent_date>=global_min_data) \
        & (patent.patent_date<global_max_data)]

edited_patent.to_csv(
    edited_PatentsView_directory + '/patent.tsv', 
    sep="\t", quoting=csv.QUOTE_NONNUMERIC, index=False)    
    
# List of patents that fit the descriptions
patent_id_list = list(set(edited_patent.patent_id))

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    
cpc_current_pregrant_publication = pd.read_csv(
                PatentsView_directory + '/cpc_current_pregrant_publication.tsv', 
                delimiter="\t", quoting=csv.QUOTE_NONNUMERIC, low_memory=False,
                dtype={'pgpub_id':'object'})

edited_cpc_current_pregrant_publication = cpc_current_pregrant_publication[
    cpc_current_pregrant_publication.pgpub_id.isin(pgpub_id_list)]

edited_cpc_current_pregrant_publication.to_csv(
    edited_PatentsView_directory + '/cpc_current_pregrant_publication.tsv', 
    sep="\t", quoting=csv.QUOTE_NONNUMERIC, index=False)    


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    
cpc_current_PatentsView = pd.read_csv(
                PatentsView_directory + '/cpc_current_PatentsView.tsv', 
                delimiter="\t", quoting=csv.QUOTE_NONNUMERIC, low_memory=False,
                dtype={'patent_id':'object'})
                
edited_cpc_current_PatentsView = cpc_current_PatentsView[
    cpc_current_PatentsView.patent_id.isin(patent_id_list)]

edited_cpc_current_PatentsView.to_csv(
    edited_PatentsView_directory + '/cpc_current_PatentsView.tsv', 
    sep="\t", quoting=csv.QUOTE_NONNUMERIC, index=False)    


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    
uspc_current_PatentsView = pd.read_csv(
                PatentsView_directory + '/uspc_current_PatentsView.tsv', 
                delimiter="\t", quoting=csv.QUOTE_NONNUMERIC, low_memory=False,
                dtype={'patent_id':'object'})

edited_uspc_current_PatentsView = uspc_current_PatentsView[
    uspc_current_PatentsView.patent_id.isin(patent_id_list)]

edited_uspc_current_PatentsView.to_csv(
    edited_PatentsView_directory + '/uspc_current_PatentsView.tsv', 
    sep="\t", quoting=csv.QUOTE_NONNUMERIC, index=False)    

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    
uspc_pregrant_publication = pd.read_csv(
                PatentsView_directory + '/uspc_pregrant_publication.tsv', 
                delimiter="\t", quoting=csv.QUOTE_NONNUMERIC, low_memory=False,
                dtype={'pgpub_id':'object'})

edited_uspc_current_pregrant_publication = uspc_pregrant_publication[
    uspc_pregrant_publication.pgpub_id.isin(pgpub_id_list)]

edited_uspc_current_pregrant_publication.to_csv(
    edited_PatentsView_directory + '/uspc_pregrant_publication.tsv', 
    sep="\t", quoting=csv.QUOTE_NONNUMERIC, index=False)    


# =============================================================================
# Edite USPTO files
# =============================================================================
os.listdir(USPTO_data_directory)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
application_data_2017 = pd.read_csv(
                USPTO_data_directory + '/application_data_2017.csv', 
                low_memory=False,
                dtype={'filing_data':'object'})

edited_application_data_2017 = application_data_2017[
    (application_data_2017.filing_date>=global_min_data) \
        & (application_data_2017.filing_date<global_max_data)]

edited_application_data_2017.to_csv(
    edited_USPTO_data_directory + '/application_data_2017.csv', index=False) 

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
application_data_2020 = pd.read_csv(
                USPTO_data_directory + '/application_data_2020.csv', 
                low_memory=False,
                dtype={'app_id':'object',
                       'filing_data':'object'})

edited_application_data_2020 = application_data_2020[
    (application_data_2020.filing_date>=global_min_data) \
        & (application_data_2020.filing_date<global_max_data)]

edited_application_data_2020.to_csv(
    edited_USPTO_data_directory + '/application_data_2020.csv', index=False) 

# List of application that fit the descriptions
app_id_list = list(set(edited_application_data_2020.application_number))

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
office_actions_uspto_research_2017 = pd.read_csv(
                USPTO_data_directory + '/office_actions_uspto_research_2017.csv', 
                low_memory=False,
                dtype={'app_id':'object'})

edited_office_actions_uspto_research_2017 = office_actions_uspto_research_2017[
    (office_actions_uspto_research_2017.app_id.isin(app_id_list))]

edited_office_actions_uspto_research_2017.to_csv(
    edited_USPTO_data_directory + '/office_actions_uspto_research_2017.csv', index=False) 

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
rejections_uspto_research_2017 = pd.read_csv(
                USPTO_data_directory + '/rejections_uspto_research_2017.csv', 
                low_memory=False,
                dtype={'app_id':'object'})

edited_rejections_uspto_research_2017 = rejections_uspto_research_2017[
    (rejections_uspto_research_2017.app_id.isin(app_id_list))]

edited_rejections_uspto_research_2017.to_csv(
    edited_USPTO_data_directory + '/rejections_uspto_research_2017.csv', index=False) 










