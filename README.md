# SKET
This repository contains the source code for the Semantic Knowledge Extractor Tool (SKET). <br /> SKET is an unsupervised hybrid knowledge extraction system that combines a rule-based expert system with pre-trained machine learning models to extract cancer-related information from pathology reports.

## Installation 

<b>CAVEAT</b>: the package has been tested on Python 3.7.7 with unix-based systems and win64 systems. There are no guarantees that it works with different configurations.

Clone this repository

```bash
git clone https://github.com/ExaNLP/sket.git
```

Install all the requirements:

```bash
pip install -r requirements.txt
```

Then install any ```core``` model from scispacy v0.3.0 (default is ```en_core_sci_sm```):

```bash
pip install </path/to/download>
```

The required scispacy models are available at: https://github.com/allenai/scispacy/tree/v0.3.0

## Datasets

Users can go into the ```datasets``` folder and place their datasets within the corresponding use case folders. Use cases are: Colon Cancer (colon), Cervix Uterine Cancer (cervix), and Lung Cancer (lung). 

Datasets can be provided in two formats:

### XLS Format

Users can provide ```.xls``` or ```.xlsx``` files with the first row consisting of column headers (i.e., fields) and the rest of data inputs. 

### JSON Format

Users can provide ```.json``` files structured in two ways: <br />

As a dict containing a ```reports``` field consisting of multiple key-value reports; 

```bash
{'reports': [{k: v, ...}, ...]}
```

As a dict containing a single key-value report.

```bash
{k: v, ...}
```

SKET concatenates data from all the fields before translation. Users can alterate this behavior by filling ```./sket/rep_proc/rules/report_fields.txt``` with target fields, one per line. Users can also provide a custom file to SKET, as long as it contains one field per line (more on this below).

Users can provide <i>special</i> headers that are treated differently from regular text by SKET. These fields are: <br />
```id```: when specified, the ```id``` field is used to identify the corresponding report. Otherwise, ```uuid``` is used.
```gender```: when specified, the ```gender``` field is used to provide patient's information within RDF graphs. Otherwise, ```gender``` is set to None.
```age```: when specified, the ```age``` field is used to provide patient's information within RDF graphs. Otherwise, ```age``` is set to None.

## Dataset Statistics

Users can compute dataset statistics to uderstand the distribution of concepts extracted by SKET for each use case. For instance, if a user wants to compute statistics for Colon Cancer, they can run 

```bash
python compute_stats.py --outputs ./outputs/concepts/refined/colon/*.json --use_case colon
```

## Pretrain

SKET can be deployed with different pretrained models, i.e., fastText and BERT. In our experiments, we employed the BioWordVec fastText model and the Bio + Clinical BERT model. <br />
<b>BioWordVec</b> can be downloaded from https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin <br />
<b>Bio + Clinical BERT</b> model can be automatically downloaded at run time by setting the ```biobert``` SKET parameter equal to 'emilyalsentzer/Bio_ClinicalBERT'

Users can pass different pretrained models depending on their preferences. 


## Usage
  
Users can deploy SKET using ```run_med_sket.py```. We release within ```./examples``` three sample datasets that can be used as toy examples to play with SKET. SKET can be deployed with different configurations and using different combinations of matching models. 

Furthermore, SKET exhibits a tunable ```threshold``` parameter that can be tuned to decide the hardness of the entity linking component. The higher the ```threshold```, the more precise the model -- at the expense of recall -- and vice versa. Users can fine-tune this parameter to obtain the desired trade-off between precision and recall. Note that ```threshold``` must always be lower than or equal to the number of considered matching models. Otherwise, the entity linking component does not return any concept.

The available matching models, in form of SKET parameters, are: <br />
```biow2v```: the scispacy pretrained word embeddings. Set this parameter to ```True``` to use them. <br />
```biofast```: the fastText model. Set this parameter to ```/path/to/fastText/file``` to use fastText. <br />
```biobert```: the BERT model. Set this parameter to ```bert-name``` to use BERT (see https://huggingface.co/transformers/pretrained_models.html for model IDs). <br />
```str_match```: the Gestalt Pattern Matching (GPM) model. Set this parameter to ```True``` to use GPM.

When using BERT, users can also set ```gpu``` parameter to the corresponding GPU number to fasten SKET execution.

For instance, a user can run the following script to obtain concepts, labels, and RDF graphs on the test.xlsx sample dataset:

```bash
python run_med_sket.py \
  	--src_lang it \
    --use_case colon \
    --spacy_model en_core_sci_sm \
    --w2v_model True \
    --string_model True \
    --thr 2.0 \
    --store True \
    --dataset ./examples/test.xlsx
```

In this case, we set the ```src_lang``` to ```it``` as the source language of reports is Italian. Therefore, SKET needs to translate reports from Italian to English before performing information extraction.
