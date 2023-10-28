# patent_classification_project - NLP-based identification of invalid patents
The *2014 Alice Corp. v. CLS Bank International Supreme Court decision* (*Alice* decision) weakened the enforceability of existing software patents and limited the patentability of new software-related innovations.  Using claim texts from patents and reject application due to *Alice*, this project provides Python methods that build and train a NLP-based classification method to identify claim texts that are invalid after the *Alice* decision.  This can be used to predict for a wide set of patent portfolios whether they are treated, allowing for causal identification of the effects of the patentability of software inventions.

I welcome any feedback and am looking forward to further developing this project in the future!


##Structure of the project
This project contains multiple subfolders and programs that are used to classify patent claim text.    



This notebook is a walkthrough of the Alice patent classification project from the definition of the training data and fitting of the NLP models to the classification of patent claims of exisiting patents. The methods for each step are in the patent_classification_project package. The different modules of the package are also executable files and contain beside a main method also other useful comments and code variations.

The models and steps in this file below build up on each other. The sequence of the execution is as follows:

claim_text_extractions_for_training.py: load patent application texts from USPTO websites that are related to rejected applications due to Alice and control texts from patents the same class and cohort.
training_data_build.py: uses the downloaded application and patent texts to prepare training data for the NLP model fitting.
NLP_model_building.py: build, fit, and evaluate the NLP classification model to estimate whether a patent claim text is invalid after Alice.
patent_claim_classification.py: classify claim texts of issued patent in PatentsViews whether they are invalid after Alice.
The package also contains the method application_pregrant_pub_claim_classification.py which uses the trained NLP model to classify claim texts of application pre-grant publications in PatentsView.

The module classification_testing.py is an optional program file that I run as the last step in this file to test that the classification method indeed performs better than using just CPC groups to identify patents that are likely invalid after Alice. The module loads the classification outcomes and defines a patent as treated if the first claim is classified as invalid. The first claim of a patent is ususally the broadest in the patent and thus deterimnes the overall scope of the patent. I then plot the patent issuances for treated and untreated patents over time and run several difference-in-differences regression models that show how patents classified as invalid are indeed less likely to be issued after Alice.

The methods that load claim texts from USPTO sites and classify the text can be run parallel since PatentsView has separate files for respective pre-grant publication and issue years. In the current modules of the package, I commented out the parallel version since I will run the files on my local desktop machine. The parallel methods are most useful on servers when classifying many more patents than in this example.

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

| Package dependencies |
|----------------------|
|gensim==4.3.0         |
|joblib==1.2.0         |
|lime==0.2.0.1         |
|linearmodels==0.0.0   |
|matplotlib==3.7.1     |
|nltk==3.7             |
|numpy==1.24.3         |
|pandas==1.5.3         |
|Requests==2.31.0      |
|scikit_learn==1.2.2   |
|seaborn==0.13.0       |
|statsmodels==0.13.5   |
|wordcloud==1.9.2      |
