import pandas as pd
import math
import string
import re
import json
import uuid
import copy
import roman

from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from transformers import MarianMTModel, MarianTokenizer

from ..utils import utils


class ReportProc(object):

	def __init__(self, src_lang, use_case, fields_path=None):
		"""
		Set translator and build regular expression to split text based on bullets

		Params:
			src_lang (str): considered source language
			use_case (str): considered use case
			fields_path (str): report fields file path

		Returns: None
		"""

		self.use_case = use_case

		if fields_path:  # read report fields file
			self.fields = utils.read_report_fields(fields_path)
		else:  # no report fields file provided
			self.fields = utils.read_report_fields('./sket/rep_proc/rules/report_fields.txt')

		if src_lang != 'en':  # set NMT model
			self.nmt_name = 'Helsinki-NLP/opus-mt-' + src_lang + '-en'
			self.tokenizer = MarianTokenizer.from_pretrained(self.nmt_name)
			self.nmt = MarianMTModel.from_pretrained(self.nmt_name)
		else:  # no NMT model required
			self.nmt_name = None
			self.tokenizer = None
			self.nmt = None

		# build regex for bullet patterns
		self.en_roman_regex = re.compile('((?<=(^i-ii(\s|:|\.)))|(?<=(^i-iii(\s|:|\.)))|(?<=(^ii-iii(\s|:|\.)))|(?<=(^i-iv(\s|:|\.)))|(?<=(^ii-iv(\s|:|\.)))|(?<=(^iii-iv(\s|:|\.)))|(?<=(^i and ii(\s|:|\.)))|(?<=(^i and iii(\s|:|\.)))|(?<=(^ii and iii(\s|:|\.)))|(?<=(^i and iv(\s|:|\.)))|(?<=(^ii and iv(\s|:|\.)))|(?<=(^iii and iv(\s|:|\.)))|(?<=(^i(\s|:|\.)))|(?<=(^ii(\s|:|\.)))|(?<=(^iii(\s|:|\.)))|(?<=(^iv(\s|:|\.)))|(?<=(\si-ii(\s|:|\.)))|(?<=(\si-iii(\s|:|\.)))|(?<=(\sii-iii(\s|:|\.)))|(?<=(\si-iv(\s|:|\.)))|(?<=(\sii-iv(\s|:|\.)))|(?<=(\siii-iv(\s|:|\.)))|(?<=(\si and ii(\s|:|\.)))|(?<=(\si and iii(\s|:|\.)))|(?<=(\sii and iii(\s|:|\.)))|(?<=(\si and iv(\s|:|\.)))|(?<=(\sii and iv(\s|:|\.)))|(?<=(\siii and iv(\s|:|\.)))|(?<=(\si(\s|:|\.)))|(?<=(\sii(\s|:|\.)))|(?<=(\siii(\s|:|\.)))|(?<=(\siv(\s|:|\.))))(.*?)((?=(\si+(\s|:|\.|-)))|(?=(\siv(\s|:|\.|-)))|(?=($)))')
		self.nl_roman_regex = re.compile('((?<=(^i-ii(\s|:|\.)))|(?<=(^i-iii(\s|:|\.)))|(?<=(^ii-iii(\s|:|\.)))|(?<=(^i-iv(\s|:|\.)))|(?<=(^ii-iv(\s|:|\.)))|(?<=(^iii-iv(\s|:|\.)))|(?<=(^i en ii(\s|:|\.)))|(?<=(^i en iii(\s|:|\.)))|(?<=(^ii en iii(\s|:|\.)))|(?<=(^i en iv(\s|:|\.)))|(?<=(^ii en iv(\s|:|\.)))|(?<=(^iii en iv(\s|:|\.)))|(?<=(^i(\s|:|\.)))|(?<=(^ii(\s|:|\.)))|(?<=(^iii(\s|:|\.)))|(?<=(^iv(\s|:|\.)))|(?<=(\si-ii(\s|:|\.)))|(?<=(\si-iii(\s|:|\.)))|(?<=(\sii-iii(\s|:|\.)))|(?<=(\si-iv(\s|:|\.)))|(?<=(\sii-iv(\s|:|\.)))|(?<=(\siii-iv(\s|:|\.)))|(?<=(\si en ii(\s|:|\.)))|(?<=(\si en iii(\s|:|\.)))|(?<=(\sii en iii(\s|:|\.)))|(?<=(\si en iv(\s|:|\.)))|(?<=(\sii en iv(\s|:|\.)))|(?<=(\siii en iv(\s|:|\.)))|(?<=(\si(\s|:|\.)))|(?<=(\sii(\s|:|\.)))|(?<=(\siii(\s|:|\.)))|(?<=(\siv(\s|:|\.))))(.*?)((?=(\si+(\s|:|\.|-)))|(?=(\siv(\s|:|\.|-)))|(?=($)))')
		self.bullet_regex = re.compile("^[-(]?\s*[\d,]+\s*[:)-]?")
		self.ranges_regex = re.compile("^\(?\s*(\d\s*-\s*\d|\d\s*\.\s*\d)\s*\)?")

	# COMMON FUNCTIONS

	def update_usecase(self, use_case):
		"""
		Update use case

		Params:
			use_case (str): considered use case

		Returns: None
		"""

		self.use_case = use_case

	def update_nmt(self, src_lang):
		"""
		Update NMT model changing source language

		Params:
			src_lang (str): considered source language

		Returns: None
		"""

		if src_lang != 'en':  # update NMT model
			self.nmt_name = 'Helsinki-NLP/opus-mt-' + src_lang + '-en'
			self.tokenizer = MarianTokenizer.from_pretrained(self.nmt_name)
			self.nmt = MarianMTModel.from_pretrained(self.nmt_name)
		else:  # no NMT model required
			self.nmt_name = None
			self.tokenizer = None
			self.nmt = None

	def update_report_fields(self, fields_path):
		"""
		Update report fields changing current ones

		Params:
			fields_path (str): report fields file

		Returns: None
		"""

		self.fields = utils.read_report_fields(fields_path)

	def load_dataset(self, reports_path, sheet, header): 
		"""
		Load reports dataset

		Params:
			reports_path (str): reports.xlsx fpath
			sheet (str): name of the excel sheet to use 
			header (int): row index used as header
		
		Returns: the loaded dataset
		"""

		if reports_path.split('.')[-1] == 'xlsx':  # requires openpyxl engine
			dataset = pd.read_excel(io=reports_path, sheet_name=sheet, header=header, engine='openpyxl')
		else:
			dataset = pd.read_excel(io=reports_path, sheet_name=sheet, header=header)
		# remove rows w/ na
		dataset.dropna(axis=0, how='all', inplace=True)

		return dataset

	def translate_text(self, text):
		"""
		Translate text from source to destination -- text is lower-cased before and after translation

		Params:
			text (str): target text

		Returns: translated text
		"""

		if type(text) == str:
			trans_text = self.nmt.generate(**self.tokenizer(text.lower(), return_tensors="pt", padding=True))[0]
			trans_text = self.tokenizer.decode(trans_text, skip_special_tokens=True)
		else:
			trans_text = ''
		return trans_text.lower()

	# AOEC SPECIFIC FUNCTIONS

	def aoec_process_data(self, dataset):
		"""
		Read AOEC reports and extract the required fields 

		Params:
			dataset (pandas DataFrame): target dataset

		Returns: a dict containing the required reports fields
		"""

		reports = dict()
		print('acquire data')
		# acquire data and translate text
		for report in tqdm(dataset.itertuples()):
			reports[str(report._1).strip()] = {
				'diagnosis_nlp': report.Diagnosi,
				'materials': report.Materiali,
				'procedure': report.Procedura if type(report.Procedura) == str else '',
				'topography': report.Topografia if type(report.Topografia) == str else '',
				'diagnosis_struct': report._5 if type(report._5) == str else '',
				'age': int(report.Età) if not math.isnan(report.Età) else 0,
				'gender': report.Sesso if type(report.Sesso) == str else ''
			}
		return reports

	def aoec_split_diagnoses(self, diagnoses, int_id, debug=False):
		"""
		Split the section 'diagnoses' within AOEC reports relying on bullets (i.e. '1', '2', etc.)

		Params:
			diagnoses (str): the 'diagnoses' section of AOEC reports
			int_id (int): the internal id specifying the current diagnosis
			debug (bool): whether to keep flags for debugging

		Returns: the part of the 'diagnoses' section related to the current internalid
		"""

		current_iids = []
		dgnss = {}
		# split diagnosis on new lines
		dlines = diagnoses.split('\n')
		# loop over lines
		for line in dlines:
			line = line.strip()
			if line:  # line contains text
				# look for range first
				rtext = self.ranges_regex.findall(line)
				if rtext:  # range found
					bullets = re.findall('\d+', rtext[0])
					bullets = list(map(int, bullets))
					bullets = range(bullets[0], bullets[1]+1)
					current_iids = deepcopy(bullets)
				else:  # ranges not found
					# look for bullets
					btext = self.bullet_regex.findall(line)
					if btext:  # bullets found
						bullets = re.findall('\d+', btext[0])
						bullets = list(map(int, bullets))
						current_iids = deepcopy(bullets)
				# associate current line to the corresponding ids
				for iid in current_iids:
					if iid in dgnss:  # iid assigned before
						dgnss[iid] += ' ' + line
					else:  # new idd
						dgnss[iid] = line
		if int_id in dgnss:  # return the corresponding diagnosis
			return dgnss[int_id]
		elif not current_iids:  # no bullet found -- return the whole diagnoses field (w/o \n to avoid problems w/ FastText)
			return diagnoses.replace('\n', ' ')
		else:  # return the whole diagnoses field (w/o \n to avoid problems w/ FastText) -- something went wrong
			if debug:
				print('\n\nSomething went wrong -- return the whole diagnoses field but print data:')
				print('Internal ID: {}'.format(int_id))
				print('Raw Field: {}'.format(diagnoses))
				print('Processed Field: {}\n\n'.format(dgnss))
			return diagnoses.replace('\n', ' ')

	def aoec_process_data_v2(self, dataset, debug=False):
		"""
		Read AOEC reports and extract the required fields (v2 used for batches from 2nd onwards)

		Params:
			dataset (pandas DataFrame): target dataset
			debug (bool): whether to keep flags for debugging

		Returns: a dict containing the required report fields
		"""

		reports = dict()
		print('acquire data and split it based on diagnoses')
		# acquire data and split it based on diagnoses
		for report in tqdm(dataset.itertuples()):
			rid = str(report.FILENAME).strip() + '_' + str(report.IDINTERNO).strip()
			reports[rid] = {
				'diagnosis_nlp': self.aoec_split_diagnoses(report.TESTODIAGNOSI, report.IDINTERNO, debug=debug),
				'materials': report.MATERIALE,
				'procedure': report.SNOMEDPROCEDURA if type(report.SNOMEDPROCEDURA) == str else '',
				'topography': report.SNOMEDTOPOGRAFIA if type(report.SNOMEDTOPOGRAFIA) == str else '',
				'diagnosis_struct': report.SNOMEDDIAGNOSI if type(report.SNOMEDDIAGNOSI) == str else '',
				'birth_date': report.NATOIL if report.NATOIL else '',
				'visit_date': report.DATAORAFINEVALIDAZIONE if report.DATAORAFINEVALIDAZIONE else '',
				'gender': report.SESSO if type(report.SESSO) == str else '',
				'image': report.FILENAME,
				'internalid': report.IDINTERNO
			}

		return reports

	def aoec_translate_reports(self, reports):
		"""
		Translate processed reports

		Params:
			reports (dict): processed reports

		Returns: translated reports
		"""

		trans_reports = copy.deepcopy(reports)
		print('translate text')
		# translate text
		for rid, report in tqdm(trans_reports.items()):
			trans_reports[rid]['diagnosis_nlp'] = self.translate_text(report['diagnosis_nlp'])
			trans_reports[rid]['materials'] = self.translate_text(report['materials'])
		return trans_reports

	# RADBOUD SPECIFIC FUNCTIONS

	def radboud_split_conclusions(self, conclusions):
		"""
		Split the section 'conclusions' within reports relying on bullets (i.e. 'i', 'ii', etc.)

		Params:
			conclusions (str): the 'conclusions' section of radboud reports

		Returns: a dict containing the 'conclusions' section divided as a bullet list
		"""

		sections = defaultdict(str)
		# use regex to identify bullet-divided sections within 'conclusions'
		for groups in self.nl_roman_regex.findall(conclusions):
			# identify the target bullet for the given section
			bullet = [group for group in groups[:65] if group and any(char.isalpha() or char.isdigit() for char in group)][0].strip()
			if 'en' in bullet:  # composite bullet
				bullets = bullet.split(' en ')
			elif '-' in bullet:  # composite bullet
				bullets = bullet.split('-')
			else:  # single bullet
				bullets = [bullet]
			# loop over bullets and concatenate corresponding sections
			for bullet in bullets:
				if groups[65] != 'en':  # the section is not a conjunction between two bullets (e.g., 'i and ii')
					sections[bullet.translate(str.maketrans('', '', string.punctuation)).upper()] += ' ' + groups[65]  # store them using uppercased roman numbers as keys - required to make Python 'roman' library working
		if bool(sections):  # 'sections' contains split sections
			return sections
		else:  # 'sections' is empty - assign the whole 'conclusions' to 'sections'
			sections['whole'] = conclusions
			return sections

	def radboud_process_data(self, dataset, debug=False):
		"""
		Read Radboud reports and extract the required fields

		Params:
			dataset (pandas DataFrame): target dataset
			debug (bool): whether to keep flags for debugging


		Returns: a dict containing the required report fields
		"""

		proc_reports = dict()
		skipped_reports = []
		unsplitted_reports = 0
		misplitted_reports = 0
		report_conc_keys = {report.Studynumber: report.Conclusion for report in dataset.itertuples()}
		for report in tqdm(dataset.itertuples()):
			rid = str(report.Studynumber).strip()
			if type(report.Conclusion) == str:  # split conclusions and associate to each block the corresponding conclusion
				# deepcopy rdata to avoid removing elements from input reports
				raw_conclusions = report.Conclusion
				# split conclusions into sections
				conclusions = self.radboud_split_conclusions(utils.nl_sanitize_record(raw_conclusions.lower(), self.use_case))
				pid = '_'.join(rid.split('_')[:-1])  # remove block and slide ids from report id - keep patient id
				related_ids = [rel_id for rel_id in report_conc_keys.keys() if pid in rel_id]  # get all the ids related to the current patient
				# get block ids from related_ids
				block_ids = []
				for rel_id in related_ids:
					if 'B' not in rel_id:  # skip report as it does not contain block ID
						skipped_reports.append(rel_id)
						continue
					if 'v' not in rel_id.lower() and '-' not in rel_id:  # report does not contain special characters
						block_part = rel_id.split('_')[-1]
						if len(block_part) < 4:  # slide ID not available
							block_ids.append(rel_id)
						else:  # slide ID available
							block_ids.append(rel_id[:-2])
					elif 'v' in rel_id.lower():  # report contains slide ID first variant (i.e., _V0*)
						block_part = rel_id.split('_')[-2]
						if len(block_part) < 4:  # slide ID not available
							block_ids.append('_'.join(rel_id.split('_')[:-1]))
						else:  # slide ID available
							block_ids.append('_'.join(rel_id.split('_')[:-1])[:-2])
					elif '-' in rel_id:  # report contains slide ID second variant (i.e., -*)
						block_part = rel_id.split('_')[-1].split('-')[0]
						if len(block_part) < 4:  # slide ID not available
							block_ids.append(rel_id.split('-')[0])
						else:  # slide ID available
							block_ids.append(rel_id.split('-')[0][:-2])
					else:
						print('something went wrong w/ current report')
						print(rel_id)

				if not block_ids:  # Block IDs not found -- skip it
					continue

				if 'whole' in conclusions:  # unable to split conclusions - either single conclusion or not appropriately specified
					if len(block_ids) > 1:  # conclusions splits not appropriately specified or wrong
						unsplitted_reports += 1
					for bid in block_ids:
						# create dict to store block diagnosis and slide ids
						proc_reports[bid] = dict()
						# store conclusion - i.e., the final diagnosis
						proc_reports[bid]['diagnosis'] = conclusions['whole']
						# store slide ids associated to the current block diagnosis
						slide_ids = []
						for sid in report_conc_keys.keys():
							if bid in sid:  # Block ID found within report ID
								if 'v' not in sid.lower() and '-' not in sid:  # report does not contain special characters
									block_part = sid.split('_')[-1]
									if len(block_part) < 4:  # slide ID not available
										continue
									else:  # slide ID available
										slide_ids.append(sid[-2:])
								elif 'v' in sid.lower():  # report contains slide ID first variant (i.e., _V0*)
									block_part = sid.split('_')[-2]
									if len(block_part) < 4:  # slide ID not available
										slide_ids.append(sid.split('_')[-1])
									else:  # slide ID available
										slide_ids.append(sid.split('_')[-2][-2:] + '_' + sid.split('_')[-1])
								elif '-' in sid:  # report contains slide ID second variant (i.e., -*)
									block_part = sid.split('_')[-1].split('-')[0]
									if len(block_part) < 4:  # slide ID not available
										slide_ids.append(sid.split('-')[1])
									else:  # slide ID available
										slide_ids.append(sid.split('-')[0][-2:] + '-' + sid.split('-')[1])
						proc_reports[bid]['slide_ids'] = slide_ids
				else:
					block_ix2id = {int(block_id[-1]): block_id for block_id in block_ids}
					if len(conclusions) < len(block_ids):  # fewer conclusions have been identified than the actual number of blocks - store and fix later
						misplitted_reports += 1
						# get conclusions IDs
						cix2id = {roman.fromRoman(cid): cid for cid in conclusions.keys()}
						# loop over Block IDs and associate the given conclusions to the corresponding blocks when available
						for bix, bid in block_ix2id.items():
							# create dict to store block diagnosis and slide ids
							proc_reports[bid] = dict()
							if bix in cix2id:  # conclusion associated with the corresponding block
								# store conclusion - i.e., the final diagnosis
								proc_reports[bid]['diagnosis'] = conclusions[cix2id[bix]]
								# store slide ids associated to the current block diagnosis
								slide_ids = []
								for sid in report_conc_keys.keys():
									if bid in sid:  # Block ID found within report ID
										if 'v' not in sid.lower() and '-' not in sid:  # report does not contain special characters
											block_part = sid.split('_')[-1]
											if len(block_part) < 4:  # slide ID not available
												continue
											else:  # slide ID available
												slide_ids.append(sid[-2:])
										elif 'v' in sid.lower():  # report contains slide ID first variant (i.e., _V0*)
											block_part = sid.split('_')[-2]
											if len(block_part) < 4:  # slide ID not available
												slide_ids.append(sid.split('_')[-1])
											else:  # slide ID available
												slide_ids.append(sid.split('_')[-2][-2:] + '_' + sid.split('_')[-1])
										elif '-' in sid:  # report contains slide ID second variant (i.e., -*)
											block_part = sid.split('_')[-1].split('-')[0]
											if len(block_part) < 4:  # slide ID not available
												slide_ids.append(sid.split('-')[1])
											else:  # slide ID available
												slide_ids.append(sid.split('-')[0][-2:] + '-' + sid.split('-')[1])
								proc_reports[bid]['slide_ids'] = slide_ids
							else:  # unable to associate diagnosis with the corresponding block -- associate the entire conclusion
								# store slide ids associated to the current block diagnosis
								slide_ids = []
								# get patient ID to store conclusions field
								pid = '_'.join(bid.split('_')[:3])
								wconc = [report_conc_keys[sid] for sid in report_conc_keys.keys() if pid in sid and type(report_conc_keys[sid]) == str]
								# store the whole 'conclusions' field
								proc_reports[bid]['diagnosis'] = wconc[0]
								for sid in report_conc_keys.keys():
									if bid in sid:  # Block ID found within report ID
										if 'v' not in sid.lower() and '-' not in sid:  # report does not contain special characters
											block_part = sid.split('_')[-1]
											if len(block_part) < 4:  # slide ID not available
												continue
											else:  # slide ID available
												slide_ids.append(sid[-2:])
										elif 'v' in sid.lower():  # report contains slide ID first variant (i.e., _V0*)
											block_part = sid.split('_')[-2]
											if len(block_part) < 4:  # slide ID not available
												slide_ids.append(sid.split('_')[-1])
											else:  # slide ID available
												slide_ids.append(sid.split('_')[-2][-2:] + '_' + sid.split('_')[-1])
										elif '-' in sid:  # report contains slide ID second variant (i.e., -*)
											block_part = sid.split('_')[-1].split('-')[0]
											if len(block_part) < 4:  # slide ID not available
												slide_ids.append(sid.split('-')[1])
											else:  # slide ID available
												slide_ids.append(sid.split('-')[0][-2:] + '-' + sid.split('-')[1])
								proc_reports[bid]['slide_ids'] = slide_ids
					else:  # associate the given conclusions to the corresponding blocks
						# loop over conclusions and fill proc_reports
						for cid, cdata in conclusions.items():
							block_ix = roman.fromRoman(cid)  # convert conclusion id (roman number) into corresponding arabic number (i.e., block index)
							if block_ix in block_ix2id:  # block with bloc_ix present within dataset
								# create dict to store block diagnosis and slide ids
								proc_reports[block_ix2id[block_ix]] = dict()
								# store conclusion - i.e., the final diagnosis
								proc_reports[block_ix2id[block_ix]]['diagnosis'] = cdata
								# store slide ids associated to the current block diagnosis
								slide_ids = []
								for sid in report_conc_keys.keys():
									if block_ix2id[block_ix] in sid:  # Block ID found within report ID
										if 'v' not in sid.lower() and '-' not in sid:  # report does not contain special characters
											block_part = sid.split('_')[-1]
											if len(block_part) < 4:  # slide ID not available
												continue
											else:  # slide ID available
												slide_ids.append(sid[-2:])
										elif 'v' in sid.lower():  # report contains slide ID first variant (i.e., _V0*)
											block_part = sid.split('_')[-2]
											if len(block_part) < 4:  # slide ID not available
												slide_ids.append(sid.split('_')[-1])
											else:  # slide ID available
												slide_ids.append(sid.split('_')[-2][-2:] + '_' + sid.split('_')[-1])
										elif '-' in sid:  # report contains slide ID second variant (i.e., -*)
											block_part = sid.split('_')[-1].split('-')[0]
											if len(block_part) < 4:  # slide ID not available
												slide_ids.append(sid.split('-')[1])
											else:  # slide ID available
												slide_ids.append(sid.split('-')[0][-2:] + '-' + sid.split('-')[1])
								proc_reports[block_ix2id[block_ix]]['slide_ids'] = slide_ids
		if debug:
			print('number of missplitted reports: {}'.format(misplitted_reports))
			print('number of unsplitted reports: {}'.format(unsplitted_reports))
			print('skipped reports:')
			print(skipped_reports)
		return proc_reports

	def radboud_process_data_v2(self, dataset):
		"""
		Read Radboud reports and extract the required fields (v2 used for anonymized datasets)

		Params:
			dataset (pandas DataFrame): target dataset

		Returns: a dict containing the required report fields
		"""

		proc_reports = dict()
		for report in tqdm(dataset.itertuples()):
			if 'Microscopy' in report._fields:  # first batch of Radboud reports
				rid = str(report.Studynumber).strip()
			else:  # subsequent anonymized batches of Radboud reports
				rid = str(report._3).strip() + '_A'  # '_A' stands for anonymized report
			if report.Conclusion:  # split conclusions and associate to each block the corresponding conclusion
				# split conclusions into sections
				conclusions = self.radboud_split_conclusions(utils.nl_sanitize_record(report.Conclusion.lower(), self.use_case))

				if 'whole' in conclusions:  # unable to split conclusions - either single conclusion or not appropriately specified
					# create block id
					bid = rid + '_1'
					# create dict to store block diagnosis
					proc_reports[bid] = dict()
					# store conclusion - i.e., the final diagnosis
					proc_reports[bid]['diagnosis'] = conclusions['whole']

				else:
					# get conclusions IDs
					cid2ix = {cid: roman.fromRoman(cid) for cid in conclusions.keys()}
					for cid, cix in cid2ix.items():
						# create block id
						bid = rid + '_' + str(cix)
						# create dict to store block diagnosis
						proc_reports[bid] = dict()
						# store conclusion - i.e., the final diagnosis
						proc_reports[bid]['diagnosis'] = conclusions[cid]
		return proc_reports

	def radboud_translate_reports(self, reports):
		"""
		Translate processed reports

		Params:
			reports (dict): processed reports

		Returns: translated reports
		"""

		trans_reports = copy.deepcopy(reports)
		print('translate text')
		# translate text
		for rid, report in tqdm(trans_reports.items()):
			trans_reports[rid]['diagnosis'] = self.translate_text(report['diagnosis'])
		return trans_reports

	# GENERAL-PURPOSE FUNCTIONS

	def read_xls_reports(self, dataset):
		"""
		Read reports from xls file

		Params:
			dataset (str): target dataset

		Returns: a list containing dataset report(s)
		"""

		if dataset.split('.')[-1] == 'xlsx':  # read input file as xlsx object
			ds = pd.read_excel(io=dataset, header=0, engine='openpyxl')
		else:  # read input file as xls object
			ds = pd.read_excel(io=dataset, header=0)

		reports = []
		for report in tqdm(ds.itertuples(index=False)):  # convert raw dataset into list containing report(s)
			reports.append({field: report[ix] for ix, field in enumerate(report._fields)})
		# return report(s)
		return reports

	def read_json_reports(self, dataset):
		"""
		Read reports from JSON file

		Params:
			dataset (str): target dataset

		Returns: a list containing dataset report(s)
		"""

		with open(dataset, 'r') as dsf:
			ds = json.load(dsf)

		if 'reports' in ds:  # dataset consists of several reports
			reports = ds['reports']
		else:  # dataset consists of single report
			reports = [ds]
		# return report(s)
		return reports

	def read_stream_reports(self, dataset):
		"""
		Read reports from stream input

		Params:
			dataset (dict): target dataset

		Returns: a list containing dataset report(s)
		"""

		if 'reports' in dataset:  # dataset consists of several reports
			reports = dataset['reports']
		else:  # dataset consists of single report
			reports = [dataset]
		# return report(s)
		return reports

	def process_data(self, dataset, debug=False):
		"""
		Read reports and extract the required fields

		Params:
			dataset (dict): target dataset
			debug (bool): whether to keep flags for debugging

		Returns: a dict containing the required report fields
		"""

		if type(dataset) == str:  # dataset passed as input file
			if dataset.split('.')[-1] == 'json':  # read input file as JSON object
				reports = self.read_json_reports(dataset)
			elif dataset.split('.')[-1] == 'xlsx' or dataset.split('.')[-1] == 'xls':  # read input file as xlsx or xls object
				reports = self.read_xls_reports(dataset)
			else:  # raise exception
				print('Format required for input: JSON, xls, or xlsx.')
				raise Exception
		else:  # dataset passed as stream dict
			reports = self.read_stream_reports(dataset)

		proc_reports = {}
		# process reports and concat fields
		for report in reports:
			if 'id' in report:
				rid = report.pop('id')  # use provided id
			else:
				rid = str(uuid.uuid4())  # generate uuid

			if 'age' in report:  # get age from report
				age = report.pop('age')
			else:  # set age to None
				age = None

			if 'gender' in report:  # get gender from report
				gender = report.pop('gender')
			else:  # set gender to None
				gender = None

			if self.fields:  # report fields specified -- restrict to self.fields
				fields = [field for field in report.keys() if field in self.fields]
			else:  # report fields not specified -- keep report fields
				fields = [field for field in report.keys()]
			report_fields = [report[field] if report[field].endswith('.') else report[field] + '.' for field in fields]
			text = ' '.join(report_fields)

			# prepare processed report
			proc_reports[rid] = {'text': text, 'age': age, 'gender': gender}
		return proc_reports

	def translate_reports(self, reports):
		"""
		Translate reports

		Params:
			reports (dict): reports

		Returns: translated reports
		"""

		trans_reports = copy.deepcopy(reports)
		print('translate text')
		# translate text
		for rid, report in tqdm(trans_reports.items()):
			trans_reports[rid]['text'] = self.translate_text(report['text'])
		return trans_reports
