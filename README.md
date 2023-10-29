# PyPatentAlice - NLP-based identification of invalid patents
The *2014 Alice Corp. v. CLS Bank International Supreme Court decision* (*Alice* decision) weakened the enforceability of existing software patents and limited the patentability of new software-related innovations.  Using claim texts from patents and reject application due to *Alice*, this project provides Python methods that build and train a NLP-based classification method to identify claim texts that are invalid after the *Alice* decision.  This can be used to predict for a wide set of patent portfolios whether they are treated, allowing for causal identification of the effects of the patentability of software inventions.

I welcome any feedback and am looking forward to further developing this project in the future!  For this project, all data are sourced from the [United States Patent and Trademark Office (USPTO)](www.uspto.gov), and [PatentsView](https://patentsview.org/download/data-download-tables) and in the public domain.   


## Structure of the project
This project contains multiple subfolders and programs that are used to classify patent claim text.  There are many specification options in the files and the current setting should be considered an example variant.  I encourage everyone to explore and adjust the files as needed.  

The file [**walkthrough patent classification**](https://github.com/Fresearch99/patent_classification_project/blob/main/walkthrough%20patent%20classification.ipynb)  is the main reference for this project.  This executable Jupyter notebook provides background information and details on how to run the patent classification.    

The folder *PyPatentAlice* is the main package of this project and contains all relevant modules.  The files are executable, but can also be imported as in [**walkthrough patent classification**](https://github.com/Fresearch99/patent_classification_project/blob/main/walkthrough%20patent%20classification.ipynb).  The models and steps in this file below build up on each other, thus the sequence cannot be changed.
1. **claim_text_extractions_for_training.py**: loads patent application texts from USPTO websites that are related to rejected applications due to *Alice* and control texts from patents from the same class and cohort.  
2. **training_data_build.py**: uses the downloaded application and patent texts to prepare training data for the NLP model fitting.
3. **NLP_model_building.py**: builds, fits, and evaluates NLP classification models to estimate whether a patent claim text is invalid after *Alice*.
4. **patent_claim_classification.py**: classifies claim texts of issued patents in PatentsViews whether they are invalid after *Alice*.
5. **application_pregrant_pub_claim_classification.py**: uses the trained NLP model to classify claim texts of application pre-grant publications in PatentsView.  This method can be run independently from **patent_claim_classification.py**.
6. **classification_testing.py**: an optional program that statistically tests whether the classification method can identify invalid patents after *Alice*.  This is an example for an econometric specification that allows for the causal identification of the effect of patentability.

The *example_data* folder contains the following short example outputs from the classification method.  They can be used for additional data analysis:
- **example_data_claim_classification.csv**: classification outputs from the output file 'FullText__patents_cpcAffected__predicted__TFIDF_poly2_issued_patents_control.csv', with 'patent_id' and 'claim_sequence' identifying the respective patent claim that was classified, '0' and '1' giving the probabilities for being valid and invalid, respectively, and 'predicted_label' is a dummy variable that is one for claims that are predicted to be invalid.
- **example_data_classified_patents.csv**: the dataset constructed in **classification_testing.py**.  'patent_id' identifies the issued patent, 'issue_date_dt' and the day on which it was issued.  The file also contains additional columns with the issue year, quarter, and the end day of the quarter of patent issuance, as well as the current CPC group and subgroup for the patent.  Finally, 'Treated' is the 'predicted_label' for the first claim in the respective patent and is one for invalid patents.
- **example_data_for_analysis.csv**: aggregate patent issuance counts by CPC group created in **classification_testing.py**.  The patent data in **example_data_classified_patents.csv** are aggregated by CPC group, patent issuance quarter, and treatment status, 'Count' is the number of issued patents within each of these groups.  'Post' is a dummy for periods after the *Alice* decision in June 2014, and 'log_Count' is the log-transformation of the 'Count'.  This dataset is used for the regression analysis in the last part of **classification_testing.py**.



The package also contains the working directory for the project. In the working directory, relevant downloaded patent data, as well as the classification outputs and the analysis outputs such as wordclouds and performance reports for NLP models are saved. The storage space required can become quite big (for example, my local working directoy take up almost 40 GB). Especially the 'PatentsView_raw_data' folder can take up a lot of space since large data files with patent information are downloaded from PatentsView (https://patentsview.org/download/data-download-tables) and stored here.

If storage limitation are a concern, limit the executions to only the variants of the patent classification that are truly needed. You can also adjust the code to store only relevant information such as CPC-based classications instead of both USPC and CPC-based. Finally, to really save storage space, pre-download the relevant patent information from PatentsView, edit the data file to only include the needed columns, and adjust the code to load only the pre-edited PatentsView data.

[^1]: 573 U.S. 208 (2014)

## Setup
Download the GitHub project and unzip.  Change the *home_directory* variable within each file, depending on where the output data are to be stored (I recommend using the *working directory* that is part of the package-download as location).  Note that the total project output can be very large (more than 40GB), if possible pre-download the required PatentsView and USPTO data into the home_directory location.

The project requires in the home_directory the following (large) data folder with raw patent data.  Most of the files will be downloaded when executing the project (except *application_data_2020.csv* which should be downloaded manually), but the execution can be accelerated if the files are already downloaded and unzipped.  Rename the files accordingly as described after downloading.

- *USPTO_raw_data* - Folder with all relevant [USPTO research datasets](https://www.uspto.gov/ip-policy/economic-research/research-datasets):
  - *application_data_2017.csv* - [2017 vintage PatEx application research data](https://bulkdata.uspto.gov/data/patent/pair/economics/2017/application_data.csv.zip).
  - *application_data_2020.csv* - [2020 vintage PatEx application research data](https://bulkdata.uspto.gov/data/patent/pair/economics/2020/application_data.csv.zip), this dataset should be downloaded manually!
  - *office_actions_uspto_research_2017.csv* - [office action research dataset](https://bulkdata.uspto.gov/data/patent/office/actions/bigdata/2017/office_actions.csv.zip).
  - *rejections_uspto_research_2017.csv* - [rejections in office action research dataset](https://bulkdata.uspto.gov/data/patent/office/actions/bigdata/2017/rejections.csv.zip).
- *PatentsView_raw_data* - Folder with relevant [data table from PatentsView](https://patentsview.org/download/data-download-tables).
  - *application_pregrant_publication.tsv* - [published application data](https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/pg_published_application.tsv.zip).
  - *cpc_current_PatentsView.tsv* - [current CPC classifications of granted patents](https://s3.amazonaws.com/data.patentsview.org/download/g_cpc_current.tsv.zip).
  - *cpc_current_pregrant_publication.tsv* - [current CPC classification data for all applications](https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/pg_cpc_at_issue.tsv.zip).
  - *patent.tsv* - [data on granted patents](https://s3.amazonaws.com/data.patentsview.org/download/g_patent.tsv.zip), this file is optional and only used for the classification testing.
  - *uspc_current_PatentsView.tsv* - [current USPC classification for granted patents](https://s3.amazonaws.com/data.patentsview.org/download/uspc_current.tsv.zip), this is a legacy file and probably will not be available forever, replace it with [uspc_at_issue](https://s3.amazonaws.com/data.patentsview.org/download/g_uspc_at_issue.tsv.zip) is needed.
  - *uspc_pregrant_publication.tsv* - [current USPC classification published applications](https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/pg_uspc_at_issue.tsv.zip), similar as above this is a legacy file and my need to be replace by [uspc_at_issue](https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/pg_uspc_at_issue.tsv.zip).


## Dependencies
The project was built and runs in Python (v 3.11.0).  Specifically, I recommend using the [Anaconda distribution](https://www.anaconda.com/download) for this project.  This project was built with conda (v.23.7.3).

Below is the list of the library dependencies of the project.

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
