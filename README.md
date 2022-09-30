# RMV-VAE

# Representation Learning to Effectively Integrate and Interpret Omics Data
Scripts are currently organised by experiment:

![alt text](https://github.com/saramasarone/RMV-VAE/blob/main/diagram.png)

* BRCA_ER: VAE on breast cancer data - predicting ER
* BRCA_Survival: RMV-VAE on breast cancer data - predicting ER
* BRCA_ER_NORM: VAE on breast cancer data - predicting survival
* BRCA_Surv_NORM: RMV-VAE on breast cancer data - predicting survival
* PAAD: VAE on pancreatic cancer data - predicting survival
* PAAD_NORM: RMV-VAE on pancreatic cancer data - predicting survival


To run this script on your data you'll need:

- Two or more omics datasets 
- Optional - clinical data to predict outcomes 

To perform attribute regularisation you can use one of the existing scripts from any RMV-VAE folder and substitute our datasets with yours.


