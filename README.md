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
