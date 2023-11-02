# PyPatentAlice - NLP-based identification of invalid patents
The *2014 Alice Corp. v. CLS Bank International Supreme Court decision*[^1] (*Alice* decision) weakened the enforceability of existing software patents and limited the patentability of new software-related innovations.  Using claim texts from patents and reject applications due to *Alice*, this project provides in the package *PyPatentAlice* Python methods that build and train an NLP-based classification model to identify claim texts that are invalid after the *Alice* decision.  The model can be used to predict for a wide set of patents and patent portfolios whether they are invalid after *Alice*, allowing researchers to estimate the causal effects of software patentability on economic, innovation, and technology outcomes.

The [Context_Alice_NLP_Method](https://github.com/Fresearch99/patent_classification_project/blob/main/Context_Alice_NLP_Method.pdf) paper provides additional context for the *Alice* decision and explains how a comprehensive version of this classification project can be used for the causal identification in economic analysis.  This patent classification model is used to identify the causal effect of the *Alice* decision on innovation, job creation, and market entry of firms in a current U.S. Census project[^2] and the paper [*Patents, Innovation, and Market Entry*](https://www.census.gov/library/working-papers/2023/adrm/CES-WP-23-45.html).  

I welcome any feedback and I am looking forward to further developing this project in the future!  All data are in the public domain and sourced from the [United States Patent and Trademark Office (USPTO)](www.uspto.gov) and [PatentsView](https://patentsview.org/download/data-download-tables).  Special thanks go to Lu et al. (2017)[^3] who identified rejected application claims due to *Alice*.   


## Structure of the project
The file [**walkthrough patent classification**](https://github.com/Fresearch99/patent_classification_project/blob/main/walkthrough%20patent%20classification.ipynb)  is the main reference for this project.  This executable Jupyter Notebook provides background information and details on how to run the patent classification.    

The folder *PyPatentAlice* is the main package of this project and contains all relevant modules.  The files are executable, but can also be imported as in [**walkthrough patent classification**](https://github.com/Fresearch99/patent_classification_project/blob/main/walkthrough%20patent%20classification.ipynb).  The modules and steps in these files build upon each other, thus the sequence cannot be changed.  There are many specification options in the files and the current setting should be considered as an example variant.  I encourage everyone to explore and adjust the files as needed. 

The modules are run in the following order:
1. **claim_text_extractions_for_training.py**: loads patent application texts from USPTO websites that are related to rejected applications due to *Alice* and control texts from patents from the same patent class and cohort.  
2. **training_data_build.py**: uses the downloaded application and patent texts to prepare training data for the NLP model fitting.
3. **NLP_model_building.py**: builds, fits, and evaluates NLP classification models to estimate whether a patent claim text is invalid after *Alice*.
4. **patent_claim_classification.py**: classifies claim texts of issued patents in PatentsViews to determine whether they are invalid after *Alice*.
5. **application_pregrant_pub_claim_classification.py**: uses the trained NLP model to classify claim texts of pre-grant application publications in PatentsView.  This method can be run independently from **patent_claim_classification.py**.
6. **classification_testing.py**: an optional program that statistically tests whether the classification method can identify invalid patents after *Alice*.  This is an example of an econometric specification that identifies the causal effect of software patentability.

The *example_data* folder contains the following example outputs from the classification method.  The outputs can be used for additional data analysis and exploration:
- **example_data_claim_classification.csv**: classification outcomes as in the output file 'FullText__patents_cpcAffected__predicted__TFIDF_poly2_issued_patents_control.csv'.  'patent_id' and 'claim_sequence' identify the respective patent claim that was classified, '0' and '1' gives the estimated probabilities for being valid and invalid, respectively, and 'predicted_label' is a dummy variable that is one for claims that are predicted to be invalid.
- **example_data_classified_patents.csv**: the dataset is constructed in **classification_testing.py**.  'patent_id' identifies the issued patent and 'issue_date_dt' the day on which the patent was issued.  The file also contains additional columns with the issue year, quarter, and the end day of the quarter the patent was issuance, as well as the current CPC group and subgroup for the patent.  Finally, 'Treated' is the 'predicted_label' for the first claim in the respective patent and is one for invalid patents.
- **example_data_for_analysis.csv**: aggregated count of issued patents created in **classification_testing.py**.  The patent data of **example_data_classified_patents.csv** are aggregated by CPC group, patent issuance quarter, and treatment status, 'Count' is the number of issued patents within each of these groups.  'Post' is a dummy for periods after the *Alice* decision in June 2014, and 'log_Count' is the log-transformation of 'Count'.  This dataset is used for the regression analysis in the last part of **classification_testing.py**.

This project also includes an edited version of the eventual output folder with *working directory*.  I recommend using the same folder when running your project and setting the *home_directory* parameter in the files accordingly. 
- *LIME_issued_patents_control*: contains outputs for the [LIME](https://github.com/marcotcr/lime) evaluation outputs of the trained model.  LIME approximates the words that are most relevant for the classification as valid or invalid.
- *TFIDF_SVC_issued_patents_control*: contains variations of the trained main model that is used for the classification, a [support vector machine classification model](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) based on a [TF-IDF vector matrix](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) of the valid and invalid training claim data.  The folder also includes performance outputs for each model.
- *Wordcloud_issued_patents_control*: contains [word clouds](https://github.com/amueller/word_cloud) of the word frequencies in valid and invalid claims.
- **main_classes_issued_patents_control.pkl**: ["pickled"](https://docs.python.org/3/library/pickle.html) main USPC classes of rejected applications that are used to train the classification model.  The USPC classes can be re-used to identify patents and application publications that can be classified by the model.
- **model_data_issued_patents_control.pkl**: ["pickled"](https://docs.python.org/3/library/pickle.html) balanced training data of valid and invalid claim texts that can be used to fit an NLP classification model.  

The project also contains the file **patent_classification_edit_raw_files_for_capsule.py** which edits the raw USPTO and PatentsView files by removing rows that are outside of the date range 2010 to 2016 for the execution in the [reproducible capsule](https://codeocean.com/capsule/5756567/tree).  This limits the storage space requirement for the capsule.

[^1]: 573 U.S. 208 (2014).
[^2]: FSRDC Project Number 2669.
[^3]: Lu, Qiang and Myers, Amanda F. and Beliveau, Scott, USPTO Patent Prosecution Research Data: Unlocking Office Action Traits (November 20, 2017). USPTO Economic Working Paper No. 2017-10.

## Setup
Clone the project:

```
git clone https://github.com/Fresearch99/patent_classification_project.git
```

or download the GitHub project and unzip:
```
https://github.com/Fresearch99/patent_classification_project/archive/refs/heads/main.zip
```

The project is also linked to a reproducible capsule on Code Ocean.  The capsule contains a shortened version of the raw data and details on the execution environment.  The [walkthrough file for capsule execution](https://codeocean.com/capsule/5756567/tree) was only slightly edited to allow for results to be stored at the end.

```
https://codeocean.com/capsule/5756567/tree
```

Change the *home_directory* parameter within each file to where the output data should be stored (I recommend using the *working directory* that is part of the package download).  Note that the total project output can be very large (more than 40GB).

If not already present, the programs will download and save required PatentsView and USPTO data to the *home_directory*.  The only exception is *application_data_2020.csv* which should be downloaded manually.  The execution can be accelerated if the raw PatentsView and USPTO data are already downloaded and unzipped to the right folder location.  Rename the files accordingly as described below.

- *USPTO_raw_data* - Folder with all relevant [USPTO research datasets](https://www.uspto.gov/ip-policy/economic-research/research-datasets):
  - *application_data_2017.csv* - [2017 vintage PatEx application research data](https://bulkdata.uspto.gov/data/patent/pair/economics/2017/application_data.csv.zip).
  - *application_data_2020.csv* - [2020 vintage PatEx application research data](https://bulkdata.uspto.gov/data/patent/pair/economics/2020/application_data.csv.zip), this dataset should be downloaded manually!
  - *office_actions_uspto_research_2017.csv* - [office action research dataset](https://bulkdata.uspto.gov/data/patent/office/actions/bigdata/2017/office_actions.csv.zip).
  - *rejections_uspto_research_2017.csv* - [rejections in office action research dataset](https://bulkdata.uspto.gov/data/patent/office/actions/bigdata/2017/rejections.csv.zip).
- *PatentsView_raw_data* - Folder with relevant [data tables from PatentsView](https://patentsview.org/download/data-download-tables):
  - *application_pregrant_publication.tsv* - [published pre-grant application data](https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/pg_published_application.tsv.zip).
  - *cpc_current_PatentsView.tsv* - [current CPC classifications of granted patents](https://s3.amazonaws.com/data.patentsview.org/download/g_cpc_current.tsv.zip).
  - *cpc_current_pregrant_publication.tsv* - [current CPC classifications for all applications](https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/pg_cpc_at_issue.tsv.zip).
  - *patent.tsv* - [data on granted patents](https://s3.amazonaws.com/data.patentsview.org/download/g_patent.tsv.zip), this file is optional and only used for the classification testing.
  - *uspc_current_PatentsView.tsv* - [current USPC classifications for granted patents](https://s3.amazonaws.com/data.patentsview.org/download/uspc_current.tsv.zip), this is a legacy file and probably will not be available forever, replace with [uspc_at_issue](https://s3.amazonaws.com/data.patentsview.org/download/g_uspc_at_issue.tsv.zip) if needed.
  - *uspc_pregrant_publication.tsv* - [current USPC classifications for published pre-grant applications](https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/pg_uspc_at_issue.tsv.zip), this is a legacy file and needs to be replaced by [uspc_at_issue](https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/pg_uspc_at_issue.tsv.zip) if needed.


## Dependencies
The project was built and runs in Python (v 3.11.0).  Specifically, I recommend using the [Anaconda distribution](https://www.anaconda.com/download) for this project with conda v.23.7.3.  The project is supported in the [reproducible capsule](https://codeocean.com/capsule/5756567/tree) under Python v3.9.7, miniconda 4.12.0.

Below is the list of the library dependencies for the project.

| Library dependencies|
|---------------------|
|gensim=4.3.0         |
|joblib=1.2.0         |
|lime=0.2.0.1         |
|linearmodels=0.0.0   |
|matplotlib=3.7.1     |
|nltk=3.7             |
|numpy=1.24.3         |
|pandas=1.5.3         |
|Requests=2.31.0      |
|scikit_learn=1.2.2   |
|seaborn=0.13.0       |
|statsmodels=0.13.5   |
|wordcloud=1.9.2      |

## Current Version
Version: 1.0
Date: October 28, 2023
Author: Dominik Jurek


