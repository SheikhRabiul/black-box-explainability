process_data.py -> preprocess data by running following 4 files
	data_extract.py -> read the dataset; store it in sqlite database.
	data_transform.py -> clean data; sample data
	data_load.py -> move cleaned and sampled data from raw tables to main table. Also write into data/data_preprocessed.csv.
	data_conversion.py -> remove unimportant features; labelEncode; onHotEncode;Scale; resample minority class;
		save the fully processed data as numpy array 

process.py -> make the necessary ready with appropriate generalized feature set and run classifier.
		#change the classifier name in the above file one by one from below. 
		classifier_et -> Extra Trees (ET)
		classifier_rf.py -> Random Forest (RF)
		classifier_gradient_boosting.py - > Gradient Boosting (GB)
		classifier_svm.py -> Support Vector Machine (SVM)
		classifier_ann.py -> Artificial Neural Network (ANN)
		  

