import pandas as pd
import math
import string
import re
import uuid
import copy
import roman

from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from transformers import MarianMTModel, MarianTokenizer

from ..utils import utils


class ReportProc(object):

	def __init__(self, src_lang, use_case):
		"""
		Set translator and build regular expression to split text based on bullets

		Params:
			src_lang (str): considered source language
			use_case (str): considered use case

		Returns: None
		"""

		self.use_case = use_case

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

	def load_dataset(self, reports_path, sheet, header): 
		"""
		Load reports dataset

		Params:
			reports_path (str): reports.xlsx fpath
			sheet (str): name of the excel sheet to use 
			header (int): row index used as header
		
		Returns: the loaded datasrt
		"""

		if reports_path.split('.')[-1] == 'xlsx':  # requires openpyxl engine
			dataset = pd.read_excel(io=reports_path, sheet_name=sheet, header=header, engine='openpyxl')
		else:
			dataset = pd.read_excel(io=reports_path, sheet_name=sheet, header=header)
		return dataset

	def translate_text(self, text):
		"""
		Translate text from source to destination

		Params:
			text (str): target text

		Returns: translated text
		"""

		if type(text) == str:
			trans_text = self.nmt.generate(**self.tokenizer(text, return_tensors="pt", padding=True))[0]
			trans_text = self.tokenizer.decode(trans_text, skip_special_tokens=True)
		else:
			trans_text = ''
		return trans_text

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
			reports[report._1] = {
				'diagnosis_nlp': report.Diagnosi,
				'materials': report.Materiali,
				'procedure': report.Procedura if type(report.Procedura) == str else '',
				'topography': report.Topografia if type(report.Topografia) == str else '',
				'diagnosis_struct': report._5 if type(report._5) == str else '',
				'age': int(report.Età) if not math.isnan(report.Età) else 0,
				'gender': report.Sesso if type(report.Sesso) == str else ''
			}
		return reports

	def aoec_split_diagnoses(self, diagnoses, int_id):
		"""
		Split the section 'diagnoses' within AOEC reports relying on bullets (i.e. '1', '2', etc.)

		Params:
			diagnoses (str): the 'diagnoses' section of AOEC reports
			int_id (int): the internal id specifying the current diagnosis

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
			print('\n\nSomething went wrong -- return the whole diagnoses field but print data:')
			print('Internal ID: {}'.format(int_id))
			print('Raw Field: {}'.format(diagnoses))
			print('Processed Field: {}\n\n'.format(dgnss))
			return diagnoses.replace('\n', ' ')

	def aoec_process_data_v2(self, dataset):
		"""
		Read AOEC reports and extract the required fields (v2 used for batches from 2nd onwards)

		Params:
			dataset (pandas DataFrame): target dataset

		Returns: a dict containing the required report fields
		"""

		reports = dict()
		print('acquire data and split it based on diagnoses')
		# acquire data and split it based on diagnoses
		for report in tqdm(dataset.itertuples()):
			rid = report.FILENAME + '_' + str(report.IDINTERNO)
			reports[rid] = {
				'diagnosis_nlp': self.aoec_split_diagnoses(report.TESTODIAGNOSI, report.IDINTERNO),
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

		# print('split data based on diagnoses')
		# split data based on diagnoses
		# for rid, report in tqdm(reports.items()):
			# reports[rid]['diagnosis_nlp'] = self.aoec_split_diagnoses(report['diagnoses'], report['internalid'])

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

	def radboud_process_data(self, dataset):  # @smarchesin TODO: once testing is finished, remove unsplitted_ and misplitted_reports
		"""
		Read Radboud reports and extract the required fields

		Params:
			dataset (pandas DataFrame): target dataset

		Returns: a dict containing the required report fields
		"""

		proc_reports = dict()
		skipped_reports = []
		unsplitted_reports = 0
		misplitted_reports = 0
		report_conc_keys = {report.Studynumber: report._2 for report in dataset.itertuples()}
		for report in tqdm(dataset.itertuples()):
			rid = report.Studynumber
			if type(report._2) == str:  # split conclusions and associate to each block the corresponding conclusion
				# deepcopy rdata to avoid removing elements from input reports
				raw_conclusions = report._2
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
		print('number of misplitted reports: {}'.format(misplitted_reports))
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
			rid = str(report._3) + '_A'  # @smarchesin TODO: can we assume 'A' stands for anonymized report?
			if report.Conclusion:  # split conclusions and associate to each block the corresponding conclusion
				# split conclusions into sections
				conclusions = self.radboud_split_conclusions(utils.nl_sanitize_record(report.conclusion.lower(), self.use_case))

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

	def process_data(self, dataset):
		"""
		Read reports and extract the required fields

		Params:
			dataset (dict): target dataset

		Returns: a dict containing the required report fields
		"""

		proc_reports = {}
		for report in dataset['reports']:
			if 'id' in report:
				rid = report.pop('id')  # use provided id
			else:
				rid = str(uuid.uuid4())  # generate uuid
			proc_reports[rid] = report
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
