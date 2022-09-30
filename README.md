# RMV-VAE

# Representation Learning to Effectively Integrate and Interpret Omics Data
Scripts are currently organised by experiment:

* BRCA_ER: VAE on breast cancer data - predicting ER
* BRCA_Survival: RMV-VAE on breast cancer data - predicting ER
* BRCA_ER_NORM: VAE on breast cancer data - predicting survival
* BRCA_Surv_NORM: RMV-VAE on breast cancer data - predicting survival
* PAAD: VAE on pancreatic cancer data - predicting survival
* PAAD_NORM: RMV-VAE on pancreatic cancer data - predicting survival


To run these experiments we used data from TCGA available here: https://xenabrowser.net/datapages/


To run this script on your data you'll need:

- Two or more omics datasets 
- Optional - clinical data to predict outcomes 

To perform attribute regularisation you can use one of the existing scripts from any RMV-VAE folder and substitute our datasets with yours.


