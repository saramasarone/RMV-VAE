## RMV-VAE

## Representation Learning to Effectively Integrate and Interpret Omics Data
pdf: https://openreview.net/pdf?id=FRE7FT9DDAj

Presented at NeurIPS 2022, AI for Science Workshop

![alt text](https://github.com/saramasarone/RMV-VAE/blob/main/diagram.png)

Scripts are currently organised by experiment:
* BRCA_ER: VAE on breast cancer data - predicting ER
* BRCA_Survival: RMV-VAE on breast cancer data - predicting ER
* BRCA_ER_NORM: VAE on breast cancer data - predicting survival
* BRCA_Surv_NORM: RMV-VAE on breast cancer data - predicting survival
* PAAD: VAE on pancreatic cancer data - predicting survival
* PAAD_NORM: RMV-VAE on pancreatic cancer data - predicting survival


To run these scripts on your data you'll need:

- Two or more omics datasets 
- Optional - clinical data to predict outcomes 

To perform attribute regularisation you can use one of the existing scripts from any RMV-VAE folder and substitute our datasets with yours. You'll have to replace the gene name by the variable you intend to normalise the data by. We presented the work regularising by two genes but you can use any variable present in the dataset.

### Contacts
email @ smasarone@turing.ac.uk :)

OpenReview: https://openreview.net/forum?id=FRE7FT9DDAj 
