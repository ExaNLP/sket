import os
import json


def assign_gpu(tknz_out, gpu):
	"""
	Assign tokenizer tensors to GPU(s)

	Params:
		tknz_out (dict): dict containing tokenizer tensors within CPU
		gpu (int): gpu device

	Returns: the dict containing tokenizer tensors within GPU(s)
	"""

	if type(gpu) == int:
		device = 'cuda:' + str(gpu)
	else:
		device = 'cpu'
	tokens_tensor = tknz_out['input_ids'].to(device)
	token_type_ids = tknz_out['token_type_ids'].to(device)
	attention_mask = tknz_out['attention_mask'].to(device)
	# assign GPU(s) tokenizer tensors to output dict
	output = {
		'input_ids': tokens_tensor,
		'token_type_ids': token_type_ids,
		'attention_mask': attention_mask
	}
	return output


def en_sanitize_record(record, use_case):  # @smarchesin TODO: define sanitize use-case functions that read replacements from file
	"""
	Sanitize record to avoid translation errors

	Params:
		record (str): target record

	Returns: the sanitized record
	"""

	if record:
		if use_case == 'colon':
			record = record.replace('octopus', 'polyp')
			record = record.replace('hairy', 'villous')
			record = record.replace('villous adenoma-tubule', 'tubulo-villous adenoma')
			record = record.replace('villous adenomas-tubule', 'tubulo-villous adenoma')
			record = record.replace('villous adenomas tubule', 'tubulo-villous adenoma')
			record = record.replace('tubule adenoma-villous', 'tubulo-villous adenoma')
			record = record.replace('tubular adenoma-villous', 'tubulo-villous adenoma')
			record = record.replace('villous adenoma tubule-', 'tubulo-villous adenoma ')
			record = record.replace('villous adenoma tubule', 'tubulo-villous adenoma')
			record = record.replace('tubulovilloso adenoma', 'tubulo-villous adenoma')
			record = record.replace('blind', 'caecum')
			record = record.replace('cecal', 'caecum')
			record = record.replace('rectal', 'rectum')
			record = record.replace('sigma', 'sigmoid')
			record = record.replace('hyperplasia', 'hyperplastic')  # MarianMT translates 'iperplastico' as 'hyperplasia' instead of 'hyperplastic'
			record = record.replace('proximal colon', 'right colon')
		if use_case == 'cervix':
			record = record.replace('octopus', 'polyp')
			record = record.replace('his cassock', 'lamina propria')
			record = record.replace('tunica propria', 'lamina propria')
			record = record.replace('l-sil', 'lsil')
			record = record.replace('h-sil', 'hsil')
			record = record.replace('cin ii / iii', 'cin23')
			record = record.replace('cin iii', 'cin3')
			record = record.replace('cin ii', 'cin2')
			record = record.replace('cin i', 'cin1')
			record = record.replace('cin-iii', 'cin3')
			record = record.replace('cin-ii', 'cin2')
			record = record.replace('cin-i', 'cin1')
			record = record.replace('cin1-2', 'cin1 cin2')
			record = record.replace('cin2-3', 'cin2 cin3')
			record = record.replace('cin-1', 'cin1')
			record = record.replace('cin-2', 'cin2')
			record = record.replace('cin-3', 'cin3')
			record = record.replace('cin 2 / 3', 'cin23')
			record = record.replace('cin 2/3', 'cin23')
			record = record.replace('cin 1-2', 'cin1 cin2')
			record = record.replace('cin 2-3', 'cin2 cin3')
			record = record.replace('cin 1', 'cin1')
			record = record.replace('cin 2', 'cin2')
			record = record.replace('cin 3', 'cin3')
			record = record.replace('ii-iii cin', 'cin2 cin3')
			record = record.replace('i-ii cin', 'cin1 cin2')
			record = record.replace('iii cin', 'cin3')
			record = record.replace('ii cin', 'cin2')
			record = record.replace('i cin', 'cin1')
			record = record.replace('port biopsy', 'portio biopsy')
	return record


def nl_sanitize_record(record, use_case):
	"""
	Sanitize record to avoid translation errors
	Params:
		record (str): target record
	Returns: the sanitized record
	"""

	if record:
		if use_case == 'cervix':
			record = record.replace('cin ii - iii', 'cin2 cin3')
			record = record.replace('cin ii-iii', 'cin2 cin3')
			record = record.replace('cin ii en  iii', 'cin2 cin3')
			record = record.replace('cin i - iii', 'cin1 cin3')
			record = record.replace('cin i-iii', 'cin1 cin3')
			record = record.replace('cin i en iii', 'cin1 cin3')
			record = record.replace('cin i - ii', 'cin1 cin2')
			record = record.replace('cin i-ii', 'cin1 cin2')
			record = record.replace('cin i en ii', 'cin1 cin2')
			record = record.replace('cin ii / iii', 'cin23')
			record = record.replace('cin iii', 'cin3')
			record = record.replace('cin ii', 'cin2')
			record = record.replace('cin i', 'cin1')
			record = record.replace('cin-iii', 'cin3')
			record = record.replace('cin-ii', 'cin2')
			record = record.replace('cin-i', 'cin1')
			record = record.replace('ii-iii cin', 'cin2 cin3')
			record = record.replace('i-ii cin', 'cin1 cin2')
			record = record.replace('iii cin', 'cin3')
			record = record.replace('ii cin', 'cin2')
			record = record.replace('i cin', 'cin1')
			record = record.replace('kin ii - iii', 'kin2 kin3')
			record = record.replace('kin ii-iii', 'kin2 kin3')
			record = record.replace('kin ii en  iii', 'kin2 kin3')
			record = record.replace('kin i - iii', 'kin1 kin3')
			record = record.replace('kin i-iii', 'kin1 kin3')
			record = record.replace('kin i en iii', 'kin1 kin3')
			record = record.replace('kin i - ii', 'kin1 kin2')
			record = record.replace('kin i-ii', 'kin1 kin2')
			record = record.replace('kin i en ii', 'kin1 kin2')
			record = record.replace('kin ii / iii', 'kin2 kin3')
			record = record.replace('kin iii', 'kin3')
			record = record.replace('kin ii', 'kin2')
			record = record.replace('kin i', 'kin1')
			record = record.replace('kin-iii', 'kin3')
			record = record.replace('kin-ii', 'kin2')
			record = record.replace('kin-i', 'kin1')
			record = record.replace('ii-iii kin', 'kin2 kin3')
			record = record.replace('i-ii kin', 'kin1 kin2')
			record = record.replace('iii kin', 'kin3')
			record = record.replace('ii kin', 'kin2')
			record = record.replace('i kin', 'kin1')
	return record


def sanitize_code(code):
	"""
	Sanitize code removing unnecessary characters

	Params:
		code (str): target code

	Returns: the sanitized code
	"""

	if code:
		code = code.replace('-', '')
		code = code.ljust(7, '0')
	return code


def sanitize_codes(codes):
	"""
	Sanitize codes by splitting and removing unnecessarsy characters

	Params:
		codes (list): target codes

	Returns: the sanitized codes
	"""

	codes = codes.split(';')
	codes = [sanitize_code(code) for code in codes]
	return codes


def read_rules(rules):
	"""
	Read rules stored within file

	Params: 
		rules (str): path to rules file

	Returns: a dict of trigger: [candidates] representing rules for each use-case
	"""

	with open(rules, 'r') as file:
		lines = file.readlines()

	rules = {'colon': {}, 'cervix': {}, 'celiac': {}, 'lung': {}}
	for line in lines:
		trigger, candidates, position, mode, use_cases = line.strip().split('\t')
		use_cases = use_cases.split(',')
		for use_case in use_cases:
			rules[use_case][trigger] = (candidates.split(','), position, mode)
	return rules


def read_dysplasia_mappings(mappings):
	"""
	Read dysplasia mappings stored within file

	Params:
		mappings (str): path to dysplasia mappings file

	Returns: a dict of {trigger: grade} representing mappings for each use-case
	"""

	with open(mappings, 'r') as file:
		lines = file.readlines()

	mappings = {'colon': {}, 'cervix': {}, 'celiac': {}, 'lung': {}}
	for line in lines:
		trigger, grade, use_cases = line.strip().split('\t')
		use_cases = use_cases.split(',')
		for use_case in use_cases:
			mappings[use_case][trigger] = grade.split(',')
	return mappings


def read_cin_mappings(mappings):
	"""
	Read cin mappings stored within file

	Params:
		mappings (str): path to cin mappings file

	Returns: a dict of {trigger: grade} representing mappings for cervical intraephitelial neoplasia
	"""

	with open(mappings, 'r') as file:
		lines = file.readlines()

	mappings = {}
	for line in lines:
		trigger, grade = line.strip().split('\t')
		mappings[trigger] = grade
	return mappings


def read_hierarchies(hrels):
	"""
	Read hierarchy relations stored within file
	
	Params:
		hrels (str): hierarchy relations file path
		
	Returns: the list of hierarchical relations
	"""
	
	with open(hrels, 'r') as f:
		rels = f.readlines()
	return [rel.strip() for rel in rels]


def read_report_fields(rfields):
	"""
	Read considered report fields stored within file

	Params:
		rfields (str): report fields file path

	Returns: the list of report fields
	"""

	with open(rfields, 'r') as f:
		fields = f.read().splitlines()
	return [field.strip() for field in fields if field]


def store_concepts(concepts, out_path, indent=4, sort_keys=False):
	"""
	Store report concepts 

	Params:
		concepts (dict): report concepts
		out_path (str): output file-path w/o extension
		indent (int): indentation level
		sort_keys (bool): sort keys

	Returns: True
	"""

	os.makedirs(os.path.dirname(out_path), exist_ok=True)

	with open(out_path, 'w') as out:
		json.dump(concepts, out, indent=indent, sort_keys=sort_keys)
	return True


def load_concepts(concept_fpath):
	"""
	Load stored concepts

	Params:
		concept_fpath (str): file-path to stored concepts

	Returns: the dict containing the report (stored) concepts
	"""

	with open(concept_fpath, 'r') as f:
		concepts = json.load(f)
	return concepts


def store_labels(labels, out_path, indent=4, sort_keys=False):
	"""
	Store report labels 

	Params:
		labels (dict): report labels
		out_path (str): output file-path w/o extension
		indent (int): indentation level
		sort_keys (bool): sort keys

	Returns: True
	"""

	os.makedirs(os.path.dirname(out_path), exist_ok=True)

	with open(out_path, 'w') as out:
		json.dump(labels, out, indent=indent, sort_keys=sort_keys)
	return True


def load_labels(label_fpath):
	"""
	Load stored labels

	Params:
		label_fpath (str): file-path to stored labels

	Returns: the dict containing the report (stored) labels
	"""

	with open(label_fpath, 'r') as f:
		labels = json.load(f)
	return labels


# AOEC RELATED FUNCTIONS

def aoec_colon_concepts2labels(report_concepts):
	"""
	Convert the concepts extracted from colon reports to the set of pre-defined labels used for classification
	
	Params:
		report_concepts (dict(list)): the dict containing for each colon report the extracted concepts
		
	Returns: a dict containing for each colon report the set of pre-defined labels where 0 = absence and 1 = presence
	"""
	
	report_labels = dict()
	# loop over reports
	for rid, rconcepts in report_concepts.items():
		# assign pre-defined set of labels to current report
		report_labels[rid] = {'cancer': 0, 'hgd': 0, 'lgd': 0, 'hyperplastic': 0, 'ni': 0}
		# textify diagnosis section
		diagnosis = ' '.join([concept[1].lower() for concept in rconcepts['Diagnosis']])
		# update pre-defined labels w/ 1 in case of label presence
		if 'colon adenocarcinoma' in diagnosis:  # update cancer
			report_labels[rid]['cancer'] = 1
		if 'dysplasia' in diagnosis:  # diagnosis contains dysplasia
			if 'mild' in diagnosis:  # update lgd
				report_labels[rid]['lgd'] = 1
			if 'moderate' in diagnosis:  # update lgd
				report_labels[rid]['lgd'] = 1
			if 'severe' in diagnosis:  # update hgd
				report_labels[rid]['hgd'] = 1
		if 'hyperplastic polyp' in diagnosis:  # update hyperplastic
			report_labels[rid]['hyperplastic'] = 1
		if sum(report_labels[rid].values()) == 0:  # update ni
			report_labels[rid]['ni'] = 1   
	return report_labels


def aoec_colon_labels2binary(report_labels):
	"""
	Convert the pre-defined labels extracted from colon reports to binary labels used for classification
	
	Params:
		report_labels (dict(list)): the dict containing for each colon report the pre-defined labels
		
	Returns: a dict containing for each colon report the set of binary labels where 0 = absence and 1 = presence
	"""
	
	binary_labels = dict()
	# loop over reports
	for rid, rlabels in report_labels.items():
		# assign binary labels to current report
		binary_labels[rid] = {'cancer_or_dysplasia': 0, 'other': 0}
		# update binary labels w/ 1 in case of label presence
		if rlabels['cancer'] == 1 or rlabels['lgd'] == 1 or rlabels['hgd'] == 1:  # update cancer_or_dysplasia label
			binary_labels[rid]['cancer_or_dysplasia'] = 1
		else:  # update other label
			binary_labels[rid]['other'] = 1  
	return binary_labels


def aoec_cervix_concepts2labels(report_concepts):
	"""
	Convert the concepts extracted from cervix reports to the set of pre-defined labels used for classification
	
	Params:
		report_concepts (dict(list)): the dict containing for each cervix report the extracted concepts
		
	Returns: a dict containing for each cervix report the set of pre-defined labels where 0 = absence and 1 = presence
	"""
	
	report_labels = dict()
	# loop over reports
	for rid, rconcepts in report_concepts.items():
		# assign pre-defined set of labels to current report
		report_labels[rid] = {
			'cancer_scc_inv': 0, 'cancer_scc_insitu': 0, 'cancer_adeno_inv': 0, 'cancer_adeno_insitu': 0,
			'lgd': 0, 'hgd': 0,
			'hpv': 0, 'koilocytes': 0,
			'glands_norm': 0, 'squamous_norm': 0
		}
		# textify diagnosis section
		diagnosis = ' '.join([concept[1].lower() for concept in rconcepts['Diagnosis']])
		# update pre-defined labels w/ 1 in case of label presence
		if 'cervical squamous cell carcinoma' in diagnosis:
			report_labels[rid]['cancer_scc_inv'] = 1
		if 'squamous carcinoma in situ' in diagnosis or 'squamous intraepithelial neoplasia' in diagnosis:
			report_labels[rid]['cancer_scc_insitu'] = 1
		if 'cervical adenocarcinoma' in diagnosis:
			if 'cervical adenocarcinoma in situ' in diagnosis:
				report_labels[rid]['cancer_adeno_insitu'] = 1
			else:
				report_labels[rid]['cancer_adeno_inv'] = 1
		if 'low grade cervical squamous intraepithelial neoplasia' in diagnosis:
			report_labels[rid]['lgd'] = 1
		if 'squamous carcinoma in situ' in diagnosis or \
			'squamous intraepithelial neoplasia' in diagnosis or \
			'cervical squamous intraepithelial neoplasia 2' in diagnosis or \
			'cervical intraepithelial neoplasia grade 2/3' in diagnosis:
			report_labels[rid]['hgd'] = 1
		if 'human papilloma virus infection' in diagnosis:
			report_labels[rid]['hpv'] = 1
		if 'koilocytotic squamous cell' in diagnosis:
			report_labels[rid]['koilocytes'] = 1
		if sum(report_labels[rid].values()) == 0:
			report_labels[rid]['glands_norm'] = 1
			report_labels[rid]['squamous_norm'] = 1
	return report_labels


def aoec_cervix_labels2aggregates(report_labels):
	"""
	Convert the pre-defined labels extracted from cervix reports to coarse- and fine-grained aggregated labels
		Params:
			report_labels (dict(list)): the dict containing for each cervix report the pre-defined labels
		Returns: two dicts containing for each cervix report the set of aggregated labels where 0 = absence and 1 = presence
	"""

	coarse_labels = dict()
	fine_labels = dict()
	# loop over reports
	for rid, rlabels in report_labels.items():
		# assign aggregated labels to current report
		coarse_labels[rid] = {'cancer': 0, 'dysplasia': 0, 'normal': 0}
		fine_labels[rid] = {'cancer_adeno': 0, 'cancer_scc': 0, 'dysplasia': 0, 'glands_norm': 0, 'squamous_norm': 0}
		# update aggregated labels w/ 1 in case of label presence
		if rlabels['cancer_adeno_inv'] == 1 or rlabels['cancer_adeno_insitu'] == 1:
			coarse_labels[rid]['cancer'] = 1
			fine_labels[rid]['cancer_adeno'] = 1
		if rlabels['cancer_scc_inv'] == 1 or rlabels['cancer_scc_insitu'] == 1:
			coarse_labels[rid]['cancer'] = 1
			fine_labels[rid]['cancer_scc'] = 1
		if rlabels['lgd'] == 1 or rlabels['hgd'] == 1:
			coarse_labels[rid]['dysplasia'] = 1
			fine_labels[rid]['dysplasia'] = 1
		if rlabels['glands_norm'] == 1:
			coarse_labels[rid]['normal'] = 1
			fine_labels[rid]['glands_norm'] = 1
		if rlabels['squamous_norm'] == 1:
			coarse_labels[rid]['normal'] = 1
			fine_labels[rid]['squamous_norm'] = 1
	return coarse_labels, fine_labels


def aoec_lung_concepts2labels(report_concepts):
	"""
	Convert the concepts extracted from lung reports to the set of pre-defined labels used for classification

	Params:
		report_concepts (dict(list)): the dict containing for each lung report the extracted concepts

	Returns: a dict containing for each lung report the set of pre-defined labels where 0 = absence and 1 = presence
	"""

	report_labels = dict()
	# loop over reports
	for rid, rconcepts in report_concepts.items():
		# assign pre-defined set of labels to current report
		report_labels[rid] = {
			'cancer_scc': 0, 'cancer_nscc_adeno': 0, 'cancer_nscc_squamous': 0, 'cancer_nscc_large': 0, 'no_cancer': 0}
		# textify diagnosis section
		diagnosis = ' '.join([concept[1].lower() for concept in rconcepts['Diagnosis']])
		# update pre-defined labels w/ 1 in case of label presence
		if 'small cell lung carcinoma' in diagnosis:
			report_labels[rid]['cancer_scc'] = 1
		if 'lung adenocarcinoma' in diagnosis or 'clear cell adenocarcinoma' in diagnosis or 'metastatic neoplasm' in diagnosis:
			report_labels[rid]['cancer_nscc_adeno'] = 1
		if 'non-small cell squamous lung carcinoma' in diagnosis:
			report_labels[rid]['cancer_nscc_squamous'] = 1
		if 'lung large cell carcinoma' in diagnosis:
			report_labels[rid]['cancer_nscc_large'] = 1
		if sum(report_labels[rid].values()) == 0:
			report_labels[rid]['no_cancer'] = 1
	return report_labels


# RADBOUD RELATED FUNCTIONS

def radboud_colon_concepts2labels(report_concepts):
	"""
	Convert the concepts extracted from reports to the set of pre-defined labels used for classification
	
	Params:
		report_concepts (dict(list)): the dict containing for each report the extracted concepts

	Returns: a dict containing for each report the set of pre-defined labels where 0 = abscence and 1 = presence
	"""
	
	report_labels = dict()
	# loop over reports
	for rid, rconcepts in report_concepts.items():
		report_labels[rid] = dict()
		# assign pre-defined set of labels to current report
		report_labels[rid]['labels'] = {'cancer': 0, 'hgd': 0, 'lgd': 0, 'hyperplastic': 0, 'ni': 0}
		# textify diagnosis section
		diagnosis = ' '.join([concept[1].lower() for concept in rconcepts['concepts']['Diagnosis']])
		# update pre-defined labels w/ 1 in case of label presence
		if 'colon adenocarcinoma' in diagnosis:  # update cancer
			report_labels[rid]['labels']['cancer'] = 1
		if 'dysplasia' in diagnosis:  # diagnosis contains dysplasia
			if 'mild' in diagnosis:  # update lgd
				report_labels[rid]['labels']['lgd'] = 1
			if 'moderate' in diagnosis:  # update lgd
				report_labels[rid]['labels']['lgd'] = 1
			if 'severe' in diagnosis:  # update hgd
				report_labels[rid]['labels']['hgd'] = 1
		if 'hyperplastic polyp' in diagnosis:  # update hyperplastic
			report_labels[rid]['labels']['hyperplastic'] = 1
		if sum(report_labels[rid]['labels'].values()) == 0:  # update ni
			report_labels[rid]['labels']['ni'] = 1   
		if 'slide_ids' in rconcepts:
			report_labels[rid]['slide_ids'] = rconcepts['slide_ids']
	return report_labels


def radboud_colon_labels2binary(report_labels):
	"""
	Convert the pre-defined labels extracted from reports to binary labels used for classification
	
	Params:
		report_labels (dict(list)): the dict containing for each report the pre-defined labels

	Returns: a dict containing for each report the set of binary labels where 0 = abscence and 1 = presence
	"""
	
	binary_labels = dict()
	# loop over reports
	for rid, rlabels in report_labels.items():
		binary_labels[rid] = dict()
		# assign binary labels to current report
		binary_labels[rid]['labels'] = {'cancer_or_dysplasia': 0, 'other': 0}
		# update binary labels w/ 1 in case of label presence
		if rlabels['labels']['cancer'] == 1 or rlabels['labels']['lgd'] == 1 or rlabels['labels']['hgd'] == 1:  # update cancer_or_adenoma label
			binary_labels[rid]['labels']['cancer_or_dysplasia'] = 1
		else:  # update other label
			binary_labels[rid]['labels']['other'] = 1
		if 'slide_ids' in rlabels:
			binary_labels[rid]['slide_ids'] = rlabels['slide_ids']
	return binary_labels


def radboud_cervix_concepts2labels(report_concepts):
	"""
	Convert the concepts extracted from cervix reports to the set of pre-defined labels used for classification

	Params:
		report_concepts (dict(list)): the dict containing for each cervix report the extracted concepts

	Returns: a dict containing for each cervix report the set of pre-defined labels where 0 = absence and 1 = presence
	"""

	report_labels = dict()
	# loop over reports
	for rid, rconcepts in report_concepts.items():
		# assign pre-defined set of labels to current report
		report_labels[rid] = {
			'cancer_scc_inv': 0, 'cancer_scc_insitu': 0, 'cancer_adeno_inv': 0, 'cancer_adeno_insitu': 0,
			'lgd': 0, 'hgd': 0,
			'hpv': 0, 'koilocytes': 0,
			'glands_norm': 0, 'squamous_norm': 0
		}
		# textify diagnosis section
		diagnosis = ' '.join([concept[1].lower() for concept in rconcepts['concepts']['Diagnosis']])
		# update pre-defined labels w/ 1 in case of label presence
		if 'cervical squamous cell carcinoma' in diagnosis:
			report_labels[rid]['cancer_scc_inv'] = 1
		if 'squamous carcinoma in situ' in diagnosis or 'squamous intraepithelial neoplasia' in diagnosis:
			report_labels[rid]['cancer_scc_insitu'] = 1
		if 'cervical adenocarcinoma' in diagnosis:
			if 'cervical adenocarcinoma in situ' in diagnosis:
				report_labels[rid]['cancer_adeno_insitu'] = 1
			else:
				report_labels[rid]['cancer_adeno_inv'] = 1
		if 'low grade cervical squamous intraepithelial neoplasia' in diagnosis:
			report_labels[rid]['lgd'] = 1
		if 'squamous carcinoma in situ' in diagnosis or \
			'squamous intraepithelial neoplasia' in diagnosis or \
			'cervical squamous intraepithelial neoplasia 2' in diagnosis or \
			'cervical intraepithelial neoplasia grade 2/3' in diagnosis:
			report_labels[rid]['hgd'] = 1
		if 'human papilloma virus infection' in diagnosis:
			report_labels[rid]['hpv'] = 1
		if 'koilocytotic squamous cell' in diagnosis:
			report_labels[rid]['koilocytes'] = 1
		if sum(report_labels[rid].values()) == 0:
			report_labels[rid]['glands_norm'] = 1
			report_labels[rid]['squamous_norm'] = 1
		if 'slide_ids' in rconcepts:
			report_labels[rid]['slide_ids'] = rconcepts['slide_ids']
	return report_labels


def radboud_cervix_labels2aggregates(report_labels):
	"""
	Convert the pre-defined labels extracted from cervix reports to coarse- and fine-grained aggregated labels

		Params:
			report_labels (dict(list)): the dict containing for each cervix report the pre-defined labels
		Returns: two dicts containing for each cervix report the set of aggregated labels where 0 = absence and 1 = presence
	"""

	coarse_labels = dict()
	fine_labels = dict()
	# loop over reports
	for rid, rlabels in report_labels.items():
		# assign aggregated labels to current report
		coarse_labels[rid] = {'cancer': 0, 'dysplasia': 0, 'normal': 0}
		fine_labels[rid] = {'cancer_adeno': 0, 'cancer_scc': 0, 'dysplasia': 0, 'glands_norm': 0, 'squamous_norm': 0}
		# update aggregated labels w/ 1 in case of label presence
		if rlabels['cancer_adeno_inv'] == 1 or rlabels['cancer_adeno_insitu'] == 1:
			coarse_labels[rid]['cancer'] = 1
			fine_labels[rid]['cancer_adeno'] = 1
		if rlabels['cancer_scc_inv'] == 1 or rlabels['cancer_scc_insitu'] == 1:
			coarse_labels[rid]['cancer'] = 1
			fine_labels[rid]['cancer_scc'] = 1
		if rlabels['lgd'] == 1 or rlabels['hgd'] == 1:
			coarse_labels[rid]['dysplasia'] = 1
			fine_labels[rid]['dysplasia'] = 1
		if rlabels['glands_norm'] == 1:
			coarse_labels[rid]['normal'] = 1
			fine_labels[rid]['glands_norm'] = 1
		if rlabels['squamous_norm'] == 1:
			coarse_labels[rid]['normal'] = 1
			fine_labels[rid]['squamous_norm'] = 1
		if 'slide_ids' in rlabels:
			coarse_labels[rid]['slide_ids'] = rlabels['slide_ids']
			fine_labels[rid]['slide_ids'] = rlabels['slide_ids']
	return coarse_labels, fine_labels


# GENERAL-PURPOSE FUNCTIONS

def colon_concepts2labels(report_concepts):
	"""
	Convert the concepts extracted from colon reports to the set of pre-defined labels used for classification

	Params:
		report_concepts (dict(list)): the dict containing for each colon report the extracted concepts

	Returns: a dict containing for each colon report the set of pre-defined labels where 0 = absence and 1 = presence
	"""

	report_labels = dict()
	# loop over reports
	for rid, rconcepts in report_concepts.items():
		# assign pre-defined set of labels to current report
		report_labels[rid] = {'cancer': 0, 'hgd': 0, 'lgd': 0, 'hyperplastic': 0, 'ni': 0}
		# textify diagnosis section
		diagnosis = ' '.join([concept[1].lower() for concept in rconcepts['Diagnosis']])
		# update pre-defined labels w/ 1 in case of label presence
		if 'colon adenocarcinoma' in diagnosis:  # update cancer
			report_labels[rid]['cancer'] = 1
		if 'dysplasia' in diagnosis:  # diagnosis contains dysplasia
			if 'mild' in diagnosis:  # update lgd
				report_labels[rid]['lgd'] = 1
			if 'moderate' in diagnosis:  # update lgd
				report_labels[rid]['lgd'] = 1
			if 'severe' in diagnosis:  # update hgd
				report_labels[rid]['hgd'] = 1
		if 'hyperplastic polyp' in diagnosis:  # update hyperplastic
			report_labels[rid]['hyperplastic'] = 1
		if sum(report_labels[rid].values()) == 0:  # update ni
			report_labels[rid]['ni'] = 1
	return report_labels


def colon_labels2binary(report_labels):
	"""
	Convert the pre-defined labels extracted from colon reports to binary labels used for classification

	Params:
		report_labels (dict(list)): the dict containing for each colon report the pre-defined labels

	Returns: a dict containing for each colon report the set of binary labels where 0 = absence and 1 = presence
	"""

	binary_labels = dict()
	# loop over reports
	for rid, rlabels in report_labels.items():
		# assign binary labels to current report
		binary_labels[rid] = {'cancer_or_dysplasia': 0, 'other': 0}
		# update binary labels w/ 1 in case of label presence
		if rlabels['cancer'] == 1 or rlabels['lgd'] == 1 or rlabels['hgd'] == 1:  # update cancer_or_dysplasia label
			binary_labels[rid]['cancer_or_dysplasia'] = 1
		else:  # update other label
			binary_labels[rid]['other'] = 1
	return binary_labels


def cervix_concepts2labels(report_concepts):
	"""
	Convert the concepts extracted from cervix reports to the set of pre-defined labels used for classification

	Params:
		report_concepts (dict(list)): the dict containing for each cervix report the extracted concepts

	Returns: a dict containing for each cervix report the set of pre-defined labels where 0 = absence and 1 = presence
	"""

	report_labels = dict()
	# loop over reports
	for rid, rconcepts in report_concepts.items():
		# assign pre-defined set of labels to current report
		report_labels[rid] = {
			'cancer_scc_inv': 0, 'cancer_scc_insitu': 0, 'cancer_adeno_inv': 0, 'cancer_adeno_insitu': 0,
			'lgd': 0, 'hgd': 0,
			'hpv': 0, 'koilocytes': 0,
			'glands_norm': 0, 'squamous_norm': 0
		}
		# textify diagnosis section
		diagnosis = ' '.join([concept[1].lower() for concept in rconcepts['Diagnosis']])
		# update pre-defined labels w/ 1 in case of label presence
		if 'cervical squamous cell carcinoma' in diagnosis:
			report_labels[rid]['cancer_scc_inv'] = 1
		if 'squamous carcinoma in situ' in diagnosis or 'squamous intraepithelial neoplasia' in diagnosis:
			report_labels[rid]['cancer_scc_insitu'] = 1
		if 'cervical adenocarcinoma' in diagnosis:
			if 'cervical adenocarcinoma in situ' in diagnosis:
				report_labels[rid]['cancer_adeno_insitu'] = 1
			else:
				report_labels[rid]['cancer_adeno_inv'] = 1
		if 'low grade cervical squamous intraepithelial neoplasia' in diagnosis:
			report_labels[rid]['lgd'] = 1
		if 'squamous carcinoma in situ' in diagnosis or \
				'squamous intraepithelial neoplasia' in diagnosis or \
				'cervical squamous intraepithelial neoplasia 2' in diagnosis or \
				'cervical intraepithelial neoplasia grade 2/3' in diagnosis:
			report_labels[rid]['hgd'] = 1
		if 'human papilloma virus infection' in diagnosis:
			report_labels[rid]['hpv'] = 1
		if 'koilocytotic squamous cell' in diagnosis:
			report_labels[rid]['koilocytes'] = 1
		if sum(report_labels[rid].values()) == 0:
			report_labels[rid]['glands_norm'] = 1
			report_labels[rid]['squamous_norm'] = 1
	return report_labels


def cervix_labels2aggregates(report_labels):
	"""
	Convert the pre-defined labels extracted from cervix reports to coarse- and fine-grained aggregated labels
		Params:
			report_labels (dict(list)): the dict containing for each cervix report the pre-defined labels
		Returns: two dicts containing for each cervix report the set of aggregated labels where 0 = absence and 1 = presence
	"""

	coarse_labels = dict()
	fine_labels = dict()
	# loop over reports
	for rid, rlabels in report_labels.items():
		# assign aggregated labels to current report
		coarse_labels[rid] = {'cancer': 0, 'dysplasia': 0, 'normal': 0}
		fine_labels[rid] = {'cancer_adeno': 0, 'cancer_scc': 0, 'dysplasia': 0, 'glands_norm': 0, 'squamous_norm': 0}
		# update aggregated labels w/ 1 in case of label presence
		if rlabels['cancer_adeno_inv'] == 1 or rlabels['cancer_adeno_insitu'] == 1:
			coarse_labels[rid]['cancer'] = 1
			fine_labels[rid]['cancer_adeno'] = 1
		if rlabels['cancer_scc_inv'] == 1 or rlabels['cancer_scc_insitu'] == 1:
			coarse_labels[rid]['cancer'] = 1
			fine_labels[rid]['cancer_scc'] = 1
		if rlabels['lgd'] == 1 or rlabels['hgd'] == 1:
			coarse_labels[rid]['dysplasia'] = 1
			fine_labels[rid]['dysplasia'] = 1
		if rlabels['glands_norm'] == 1:
			coarse_labels[rid]['normal'] = 1
			fine_labels[rid]['glands_norm'] = 1
		if rlabels['squamous_norm'] == 1:
			coarse_labels[rid]['normal'] = 1
			fine_labels[rid]['squamous_norm'] = 1
	return coarse_labels, fine_labels


def lung_concepts2labels(report_concepts):
	"""
	Convert the concepts extracted from lung reports to the set of pre-defined labels used for classification

	Params:
		report_concepts (dict(list)): the dict containing for each lung report the extracted concepts

	Returns: a dict containing for each lung report the set of pre-defined labels where 0 = absence and 1 = presence
	"""

	report_labels = dict()
	# loop over reports
	for rid, rconcepts in report_concepts.items():
		# assign pre-defined set of labels to current report
		report_labels[rid] = {
			'cancer_scc': 0, 'cancer_nscc_adeno': 0, 'cancer_nscc_squamous': 0, 'cancer_nscc_large': 0, 'no_cancer': 0}
		# textify diagnosis section
		diagnosis = ' '.join([concept[1].lower() for concept in rconcepts['Diagnosis']])
		# update pre-defined labels w/ 1 in case of label presence
		if 'small cell lung carcinoma' in diagnosis:
			report_labels[rid]['cancer_scc'] = 1
		if 'lung adenocarcinoma' in diagnosis or 'clear cell adenocarcinoma' in diagnosis or 'metastatic neoplasm' in diagnosis:
			report_labels[rid]['cancer_nscc_adeno'] = 1
		if 'non-small cell squamous lung carcinoma' in diagnosis:
			report_labels[rid]['cancer_nscc_squamous'] = 1
		if 'lung large cell carcinoma' in diagnosis:
			report_labels[rid]['cancer_nscc_large'] = 1
		if sum(report_labels[rid].values()) == 0:
			report_labels[rid]['no_cancer'] = 1
	return report_labels
