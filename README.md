# DiscHPO
This repository contains the implementation of the DiscHPO model, developed as part of our contribution to the BioCreative VIII Track 3 shared task.

# Notebooks Dscription
Notebook | Description | 
--- | --- | 
HPO_DataAnalysis | Train and validation sets Data analysis |
HPO_Preprocessing| Script used for preprocessing to reach a format compatable for NER model input |
HPO_NER | NER model a modified version of https://github.com/ToluClassics/Notebooks/blob/main/T5_Ner_Finetuning.ipynb |
HPO_NER_Large | NER model Used for the Large versions of T5 models |
HPO_NER_LoRa | NER model with added LoRa adaptation |
HPO_Postprocessing | Process the results back to the original dataset format (with numrical offsets)|
HPO_Training_Dataset | Creating a dataset to train the linking model, I also added function to delete UnobservableHPOTerms, I also added function to append ObservedHPO to training set
HPO_Synonyms | Augmented training set with synonyms from Hp_term dataset and appended to it all mentions from training set with ther respective ID
HPO_Linking+FineTuning | Implementation of our normalisation component, here we Finetuning sentence transformer using our created training set
HPO_Linking+Training | We train clinicalRoBERTa from the scratch with our data to make it sentence transformer
Resolving EvalScript Errors |  Created to solve errors resulting while using the official Evaluation script such as adding missing obs, deleting duplications ..etc
HPO_DiscResultFilter |  
HPO_AlignedDiscNEs | used to calculate the results for Discontinuoues span, by taking the "alignmentPheNorm" that resulted from the official evaluation script, and drop non-discNEs then calculate the metrics(EM,P,R,F1)
BioCreativeVIIITask3EvaluationScript | Organisers Official Evaluation Script

# Notebooks Order:
(1) Start with HPO_Preprocessing.ipynb run from start to end of preprocessing.

(2) Run HPO_NER.ipynb : just change model name and tokeniser.

(3) Take NER results (e.g. T5-Large),put it in "content/biocreative/result", open HPO_Postprocessing, run it all over the dataset.

(4) take TBL-Model.tsv and put it in "content/biocreative/result", in  HPO_Linking_fintuning.ipynb put the path of file and run linking. ((Skip training Block)).

(5) For evaluation, run the normalisation results on Resolving EvalScript Errors.ipynb.


