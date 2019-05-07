# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:04:57 2019

@author: Sheikh Rabiul Islam
Purpose: running classifiers.

"""
import sys
classifier_file_name = "classifier_et.py"
classifier_output_file_name = "output_et.txt"


sys.stdout = open(classifier_output_file_name, 'w')  #comment this line in case you want to see output on the console.

import time

for i in range(0,25):
    i_s = str(i)
	#change the directory according to the directory structure of your project.
    runfile('E:/projects/code-explainability/data_conversion_alt.py', args= i_s, wdir='E:/projects/code-explainability')
    start = time.time()
    time.sleep(3)
    exec(open(classifier_file_name).read())
    end = time.time()
