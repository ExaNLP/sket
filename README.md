# SKET
This repository contains the source code for the Semantic Knowledge Extractor Tool (SKET). <br /> SKET is an unsupervised hybrid knowledge extraction system that combines a rule-based expert system with pre-trained machine learning models to extract cancer-related information from pathology reports.

## Usage 

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
