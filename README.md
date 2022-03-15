# Word-Similairty-Scoring-Flask-App
This Flask Application utilizes the basic functionality of the Universal Sentence Encoder model and BERT Sentence Tranformer model to compare text. The output renders similarity scores of inputted text by each model. Thus, comparing the accuracy of each model through the inputted text. 

Goal

Practice and explore the Flask Application capabilities. Also, explore the similarity comparisons of the BERT encoding model, Universal Sentence Encoder embedding model, and standard TF-IDF.
Similarity scoring algorithm for BERT and USE are done through a cosine similairty built using Numpy.

Dependencies-Modeling

Tensorflow, Tensorflow_hub, Sentence_Transfomer, Sklearn

Notes

No runtime improvements were made, average runtime ~15seconds. 
No stop words were implemented.
Special characters are stripped through regular expressions.
Runs on localhost development server. 
Practice purposes only.
