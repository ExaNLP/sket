import numpy as np
import torch
import spacy
import fasttext
import itertools
import re

from tqdm import tqdm
from textdistance import ratcliff_obershelp
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from .normalizer import MinMaxNormalizer
from ..utils import utils
from ..negex.negation import Negex


class NERD(object):

	def __init__(self, biospacy="en_core_sci_lg", biow2v=True, str_match=False, biofast=None, biobert=None, rules=None, dysplasia_mappings=None, cin_mappings=None, gpu=None):
		"""
		Load models and rules

		Params:
			biospacy (str): full spaCy pipeline for biomedical data
			biow2v (bool): whether to use biospacy to perform semantic matching or not
			str_match (bool): string matching
			biofast (str): biomedical fasttext model
			biobert (str): biomedical bert model
			rules (str): hand-crafted rules file path
			dysplasia_mappings (str): dysplasia mappings file path
			cin_mappings (str): cin mappings file path
			gpu (int): use gpu when using BERT

		Returns: None
		"""

		# prepare spaCy model
		self.nlp = spacy.load(biospacy)
		# prepare PhraseMatcher model
		self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
		# prepare Negex model
		self.negex = Negex(self.nlp, language="en_clinical", chunk_prefix=["free of", "free from"])  # chunk_prefix allows to match also negations chunked together w/ entity mentions
		self.negex.add_patterns(preceding_negations=["free from"])  # @smarchesin TODO: read negations from file if the number of patterns rises
		self.negex.remove_patterns(following_negations=["free"])  # 'free' pattern clashes w/ 'free of' and 'free from' -- @smarchesin TODO: is there a way to fix this without removing 'free'?

		# load hand-crafted rules
		if rules:  # custom hand-crafted rules file path
			self.rules = utils.read_rules(rules)
		else:  # default hand-crafted rules file path
			self.rules = utils.read_rules('./sket/nerd/rules/rules.txt')
		# set patterns for PhraseMatcher 
		self.patterns = {use_case: {trigger: [self.nlp(candidate) for candidate in candidates[0]] for trigger, candidates in rules.items()} for use_case, rules in self.rules.items()}

		# add expand_entity_mentions to spaCy processing pipeline
		self.nlp.add_pipe(self.expand_entity_mentions, name='expand_entities', after='ner')
		# add negation detector to spaCy pipeline
		self.nlp.add_pipe(self.negex, last=True)

		self.biow2v = biow2v
		if str_match:  # prepare string matching model
			self.gpm = ratcliff_obershelp  # gpm == gestalt pattern matching
		else:
			self.gpm = None
		if biofast:  # prepare fasttext model
			self.biofast_model = fasttext.load_model(biofast)
		else:
			self.biofast_model = None
		if biobert:  # prepare bert model
			self.bert_tokenizer = AutoTokenizer.from_pretrained(biobert)
			self.bert_model = AutoModel.from_pretrained(biobert)
			self.gpu = gpu
			if type(self.gpu) == int:
				device = 'cuda:' + str(self.gpu)
			else:
				device = 'cpu'
			self.bert_model = self.bert_model.to(device)
		else:
			self.bert_model = None

		# load dysplasia mappings
		if dysplasia_mappings:  # custom dysplasia mappings file path
			self.dysplasia = utils.read_dysplasia_mappings(dysplasia_mappings)
		else:  # default dysplasia mappings file path
			self.dysplasia = utils.read_dysplasia_mappings('./sket/nerd/rules/dysplasia_mappings.txt')
		# load cin mappings
		if cin_mappings:  # custom cin mappings file path
			self.cin = utils.read_cin_mappings(cin_mappings)
		else:  # default cin mappings file path
			self.cin = utils.read_cin_mappings('./sket/nerd/rules/cin_mappings.txt')
		# define set of ad hoc linking functions
		self.ad_hoc_linking = {
			'colon': self.ad_hoc_colon_linking,
			'cervix': self.ad_hoc_cervix_linking,
			'lung': self.ad_hoc_lung_linking
		}
		# set parameter to None before choosing use case
		self.use_case_ad_hoc_linking = None
		# define set of ad hoc post processing concept operations
		self.ad_hoc_post_processing = {
			'colon': self.ad_hoc_colon_post_processing,
			'cervix': self.ad_hoc_cervix_post_processing,
			'lung': self.ad_hoc_lung_post_processing
		}
		# set parameter to None before choosing use case
		self.use_case_ad_hoc_post_processing = None
		# set parameter to store the hand-crated rules restricted to a specific use-case (updated w/ self.set_rules() func)
		self.use_case_rules = dict()
		# set parameter to store dysplasia  mappings restricted to a specific use-case
		self.use_case_dysplasia = dict()

	# COMMON FUNCTIONS

	def restrict2use_case(self, use_case):  # @smarchesin TODO: remove all the triggers within PhraseMacher by looping over use cases - then consider only the use case ones
		"""
		Restrict hand crafted rules to the considered use-case

		Params: 
			use_case (str): the considered use case

		Returns: the updated rules, candidates, and mappings
		"""

		# restrict hand crafted rules
		self.use_case_rules = self.rules[use_case]
		self.use_case_dysplasia = self.dysplasia[use_case]
		self.use_case_ad_hoc_linking = self.ad_hoc_linking[use_case]
		self.use_case_ad_hoc_post_processing = self.ad_hoc_post_processing[use_case]
		# remove triggers within PhraseMatcher
		for use_case_patterns in self.patterns.values():
			for trigger in use_case_patterns.keys():
				if trigger in self.matcher:  # trigger found within PhraseMatcher -- remove it
					self.matcher.remove(trigger)
		# add triggers within PhraseMatcher for the specified use case
		for trigger, candidates in self.patterns[use_case].items():
			self.matcher.add(trigger, None, *candidates)

	def expand_entity_mentions(self, doc):
		"""
		Expand entity mentions relying on hand-crafted rules

		Params:
			doc (spacy.tokens.doc.Doc): text processed w/ spaCy models

		Returns: a new set of entities for doc
		"""

		spans = list()
		# loop over restricted entities and expand entity mentions based on hand-crafted rules
		for ent in doc.ents:
			# identify triggers for current entity mention
			triggers = [trigger for trigger in self.use_case_rules.keys() if (trigger in ent.text)]
			if triggers:  # current entity presents a trigger
				# keep longest trigger as candidate trigger - e.g., adenocarcinoma instead of carcinoma
				trigger = max(triggers, key=len)
				candidates, location, mode = self.use_case_rules[trigger]
				# check whether the entity mention contains any rule's candidate and exclude those candidates already contained within the entity mention
				target_candidates = [candidate for candidate in candidates if (candidate not in ent.text)]
				# search target candidates within preceding, subsequent or both tokens
				if location == 'PRE':  # candidates are matched on preceding tokens 
					if mode == 'EXACT':  # candidates are matched by exact matching immediately preceding tokens  
						spans = self.pre_exact_match(doc, ent, target_candidates, spans)
					elif mode == 'LOOSE':  # candidates are matched by finding matches within preceding tokens
						spans = self.pre_loose_match(doc, ent, trigger, target_candidates, spans)
					else:  # wrong or mispelled mode - return exception
						print("The mode is wrong or mispelled in the rules.txt file")
						raise Exception
				elif location == 'POST':  # candidates are matched on subsequent tokens 
					if mode == 'EXACT':  # candidates are matched by exact matching immediately subsequent tokens
						spans = self.post_exact_match(doc, ent, target_candidates, spans)
					elif mode == 'LOOSE':  # candidates are matched by finding matches within subsequent tokens
						spans = self.post_loose_match(doc, ent, trigger, target_candidates, spans)
					else:  # wrong or mispelled mode - return exception
						print("The mode is wrong or mispelled in the rules.txt file")
						raise Exception
				elif location == 'BOTH':  # candidates are matched on preceding and subsequent tokens
					if mode == 'EXACT':  # candidates are matched by exact matching immediately preceding and subsequent tokens
						spans = self.pre_exact_match(doc, ent, target_candidates, spans)
						spans = self.post_exact_match(doc, ent, target_candidates, spans)	
					elif mode == 'LOOSE':  # candidates are matched by finding matches within preceding and subsequent tokens
						spans = self.pre_loose_match(doc, ent, trigger, target_candidates, spans)
						spans = self.post_loose_match(doc, ent, trigger, target_candidates, spans)
					else:  # wrong or mispelled mode - return exception
						print("The mode is wrong or mispelled in the rules.txt file")
						raise Exception
				else:  # error in the rules.txt file
					print("The positional information is wrong or mispelled in the rules.txt file")
					raise Exception
			else:  # current entity does not present a trigger
				spans.append([ent.start, ent.end])
		if spans:  # doc contains valid entity mentions
			# merge entities w/ overlapping spans
			merged_spans = self.merge_spans(spans)
			doc.ents = [Span(doc, span[0], span[1], label='ENTITY') for span in merged_spans]
		return doc

	def pre_exact_match(self, doc, ent, candidates, spans):
		"""
		Perform exact matching between entity mention and preceding candidates and return the extended span (i.e., entity mention + candidate)

		Params: 
			doc (spacy.tokens.doc.Doc): text processed w/ spaCy models
			ent (spacy.tokens.doc.Doc.ents): entity mention found by NER
			candidates (list(string)): list of candidates associated to the trigger
			spans (list(list)): list of span ranges [start, end]

		Returns: the list of expanded spans given the entity mentions 
		""" 

		matched_candidate_ix = None
		ix = self.skip_pre_punct(doc, ent.start-1)  # returns previous token index if token.is_punct != True, otherwise None
		if type(ix) == int:   
			for candidate in candidates:  # loop over candidates 
				num_tokens = len(candidate.split())  # number of tokens to inspect
				pre_tokens = doc[max(0, ix-num_tokens+1):ix+1]
				if candidate == pre_tokens.text:  # exact match between candidate and tokens
					matched_candidate_ix = pre_tokens.start
		if matched_candidate_ix:
			# expand entity mention
			spans.append([matched_candidate_ix, ent.end])
		else:
			# keep current entity mention as is
			spans.append([ent.start, ent.end])
		return spans

	def skip_pre_punct(self, doc, ix):
		"""
		Get (recursively) the index of the precedent token where token.is_alpha == True (closing punctuation not allowed)

		Params:
			doc (spacy.tokens.doc.Doc): the processed document
			ix (int): the current index

		Returns: the correct token index or None if skip_punct meets EOS
		"""

		if ix == -1 or doc[ix].text == '.':  # BOS or closing punctuation
			return None
		elif not doc[ix].is_punct:  # base case
			return ix
		else:  # recursive case
			return self.skip_pre_punct(doc, ix-1) 

	def post_exact_match(self, doc, ent, candidates, spans):
		"""
		Perform exact matching between entity mention and subsequent candidates and return the extended span (i.e., entity mention + candidate)

		Params: 
			doc (spacy.tokens.doc.Doc): text processed w/ spaCy models
			ent (spacy.tokens.doc.Doc.ents): entity mention found by NER
			candidates (list(string)): list of candidates associated to the trigger
			spans (list(list)): list of span ranges [start, end]

		Returns: the list of expanded spans given the entity mentions 
		"""

		matched_candidate_ix = None
		ix = self.skip_post_punct(doc, ent.end)  # returns next token index if token.is_punct != True, otherwise None
		if type(ix) == int:  
			for candidate in candidates:  # loop over candidates
				num_tokens = len(candidate.split())  # number of tokens to inspect
				post_tokens = doc[ix:ix+num_tokens]
				if candidate == post_tokens.text:  # exact match between candidate and tokens
					matched_candidate_ix = post_tokens.end
		if matched_candidate_ix:
			# expand entity mention
			spans.append([ent.start, matched_candidate_ix])
		else:
			# keep current entity mention as is
			spans.append([ent.start, ent.end])
		return spans

	def skip_post_punct(self, doc, ix):
		"""
		Get (recursively) the index of the posterior token where token.is_alpha == True (closing punctuation not allowed)
		
		Params:
			doc (spacy.tokens.doc.Doc): the processed document
			ix (int): the current index
			
		Returns: the correct token index or None if skip_punct meets EOS
		"""
		
		if ix == len(doc) or doc[ix].text == '.':  # EOS or closing punctuation
			return None
		elif not doc[ix].is_punct:  # base case
			return ix
		else:  # recursive case
			return self.skip_post_punct(doc, ix+1)

	def pre_loose_match(self, doc, ent, trigger, candidates, spans):
		"""
		Perform loose matching between entity mention and preceding candidates and return the extended span (i.e., entity mention + candidate)

		Params: 
			doc (spacy.tokens.doc.Doc): text processed w/ spaCy models
			ent (spacy.tokens.doc.Doc.ents): entity mention found by NER
			trigger (string): token triggered for the entity mention
			candidates (list(string)): list of candidates associated to the trigger
			spans (list(list)): list of span ranges [start, end]

		Returns: the list of expanded preceding spans given the entity mentions 
		"""

		matched_candidates = list()
		ix = self.get_pre_tokens(ent)  # returns previous token index if not token.is_punct == True, otherwise None
		if type(ix) == int:  
			# perform matching over doc and return matches
			matches = self.matcher(doc)
			for m_id, m_start, m_end in matches:
				if self.matcher.vocab.strings[m_id] != trigger:  # match w/ different trigger
					continue
				if (m_start < ix) or (m_end > ent.start):  # match out of bounds
					continue
				if doc[m_start:m_end].text not in candidates:  # match out of candidates
					continue
				matched_candidates.append(m_start)  # match found - store starting index
		if matched_candidates:
			# keep earliest candidate index for entity mention's expansion
			fix = min(matched_candidates)
			# expand entity mention
			spans.append([fix, ent.end])
		else:
			# keep current entity mention as is
			spans.append([ent.start, ent.end])
		return spans

	@staticmethod
	def get_pre_tokens(ent):
		"""
		Get index of the first token

		Params:
			ent (spacy.tokens.span.Span): the current entity

		Returns: the correct token index or None if skip_punct meets BOS
		"""

		sent_ix = ent.sent.start
		if ent.start == sent_ix:  # entity mention is the first token in the sentence - skip it
			return None
		else:  # return index of the first token in sentence
			return sent_ix

	def post_loose_match(self, doc, ent, trigger, candidates, spans):
		"""
		Perform loose matching between entity mention and subsequent candidates and return the extended span (i.e., entity mention + candidate)

		Params: 
			doc (spacy.tokens.doc.Doc): text processed w/ spaCy models
			ent (spacy.tokens.doc.Doc.ents): entity mention found by NER
			trigger (string): token triggered for the entity mention
			candidates (list(string)): list of candidates associated to the trigger
			spans (list(list)): list of span ranges [start, end]

		Returns: the list of expanded subsequent spans given the entity mentions 
		"""

		matched_candidates = list()
		ix = self.get_post_tokens(ent)
		if type(ix) == int:  # returns next token index if not token.is_punct == True, otherwise None
			# perform matching over doc and return matches
			matches = self.matcher(doc)
			for m_id, m_start, m_end in matches:
				if self.matcher.vocab.strings[m_id] != trigger:  # match w/ different trigger
					continue
				if (m_start < ent.end) or (m_end > ix):  # match out of bounds
					continue
				if doc[m_start:m_end].text not in candidates:  # match out of candidates
					continue
				matched_candidates.append(m_end)  # match found - store closing index
		if matched_candidates:
			# keep latest candidate index for entity mention's expansion
			lix = max(matched_candidates)
			# expand entity mention
			spans.append([ent.start, lix])
		else:
			# keep current entity mention as is
			spans.append([ent.start, ent.end])
		return spans

	@staticmethod
	def get_post_tokens(ent):
		"""
		Get (recursively) the index of the subsequent token where token.is_alpha == True
		
		Params:
			ent (spacy.tokens.span.Span): the current entity
			
		Returns: the correct token index or None if skip_punct meets EOS
		"""
		
		sent_ix = ent.sent.end
		if ent.end == sent_ix:  # entity mention is the last token in the sentence - skip it
			return None
		else:  # return index of the last token in sentence
			return sent_ix

	@staticmethod
	def merge_spans(spans):
		"""
		Merge spans w/ overlapping ranges
		
		Params:
			spans (list(list)): list of span ranges [start, end]
			
		Returns: a list of merged span ranges [start, end]
		"""
		
		spans.sort(key=lambda span: span[0]) 
		merged_spans = [[spans[0][0], spans[0][1]]]  # avoid copying by reference
		for current in spans:
			previous = merged_spans[-1]
			if current[0] < previous[1]:
				previous[1] = max(previous[1], current[1])
			else:
				merged_spans.append(current)
		return merged_spans

	def extract_entity_mentions(self, text, keep_negated=False):
		"""
		Extract entity mentions identified within text.

		Params:
			text (str): text to be processed.
			keep_negated (bool): keep negated entity mentions

		Returns: a list of named/unnamed detected entity mentions
		"""

		doc = self.nlp(text)
		if keep_negated:  # keep negated mentions
			return [mention for mention in doc.ents]
		else: 
			return [mention for mention in doc.ents if mention._.negex is False]

	def text_similarity(self, mention, label):
		"""
		Compute different similarity measures between entity mention and concept label

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text
			label (spacy.tokens.doc.Doc | np.array(728) | (spacy.tokens.doc.Doc, np.array(728))): concept label from ontology

		Returns: sim scores and names for each considered method
		"""

		sim_scores = []
		sim_names = []
		if self.biow2v: # compute word2vec sim scores
			word2vec_scores = mention.similarity(label[0])
			sim_scores += [word2vec_scores]
			sim_names += ['word2vec']

		if self.gpm:  # compute string matching sim scores
			string_scores = self.gpm.normalized_similarity(mention.text, label[0].text)
			sim_scores += [string_scores]
			sim_names += ['gpm']

		if self.biofast_model:  # compute FastText sim scores
			fasttext_scores = cosine_similarity(
				[self.biofast_model.get_sentence_vector(mention.text)], [self.biofast_model.get_sentence_vector(label[0].text)]
			)
			sim_scores += [fasttext_scores[0]]
			sim_names += ['fasttext']

		if self.bert_model:  # compute Bert sim scores
			tokens = utils.assign_gpu(self.bert_tokenizer(mention.text, return_tensors="pt"), self.gpu)  # get tokens
			embs = self.bert_model(**tokens)[0]  # get BERT last layer hidden states
			pooled_mention = torch.mean(embs, 1).cpu().detach().numpy()  # compute pooling to obtain mention embedding
			pooled_mention = pooled_mention.reshape(1, -1)  # reshape to perform cosine similarity
			if len(label) == 1:
				pooled_label = label[0].reshape(1, -1)  # reshape to perform cosine similarity
			else:
				pooled_label = label[1].reshape(1, -1)  # reshape to perform cosine similarity
			bert_scores = cosine_similarity(pooled_mention, pooled_label)[0]
			sim_scores += [bert_scores]
			sim_names += ['bert']

		if len(sim_scores) == 0:
			print('No semantic matching method selected.\nPlease select any combination of: "biow2v", "str_match", "biofast", and "biobert"')
			raise Exception
		return sim_scores, sim_names

	def associate_mention2candidate(self, mention, labels, sim_thr=0.7):
		"""
		Associate entity mention to candidate concept label

		Params:
			mention (spacy.token.span.Span): entity mention extracted from text
			labels (list(spacy.tokens.doc.Doc) | dict(label: np.array(728)) | dict(label: (spacy.tokens.doc.Doc, np.array(728)))): list of concept labels from reference ontology
			sim_thr (float): keep candidates with sim score greater than or equal to sim_thr

		Returns: candidate ontology concept (or None)
		"""

		# perform sim scores between entity mention and onto labels
		scores_and_labels = {lid: self.text_similarity(mention, ldata)[0] for lid, ldata in labels.items()}
		# get scores and ids
		scores = np.array(list(scores_and_labels.values()))
		labels = list(scores_and_labels.keys())
		# set normalizers for sim methods
		norms = {i: MinMaxNormalizer(scores[:, i]) for i in range(0, scores.shape[1])}
		# perform combSUM over scores w/ norms
		comb_scores = np.array([norms[i](scores[:, i]) for i in range(0, scores.shape[1])]).sum(axis=0)
		if np.argwhere(comb_scores >= sim_thr).size != 0:  # return (mention, label) pair
			# keep label w/ highest comb_score
			label = labels[np.argsort(-comb_scores[:])[0]]
			return [[mention.text, label]]
		else:  # return (mention, None) pair
			return [[mention.text, None]]

	def link_mentions_to_concepts(self, mentions, labels, use_case_ontology, sim_thr=0.7, raw=False):
		"""
		Link identified entity mentions to ontology concepts 

		Params:
			mentions (list(spacy.token.span.Span)): list of entity mentions extracted from text
			labels (list(spacy.token.span.Span)): list of concept labels from reference ontology
			use_case_ontology (pandas DataFrame): reference ontology restricted to the use case considered
			sim_thr (float): keep candidates with sim score greater than or equal to sim_thr
			raw (bool): whether to return concepts within semantic areas or mentions+concepts

		Returns: a dict of identified ontology concepts {semantic_area: [iri, mention, label], ...}
		"""

		# link mentions to concepts
		mentions_and_concepts = [self.use_case_ad_hoc_linking(mention, labels, sim_thr) for mention in mentions]
		mentions_and_concepts = list(itertools.chain.from_iterable(mentions_and_concepts))
		# post process mentions and concepts based on the considered use case
		mentions_and_concepts = self.use_case_ad_hoc_post_processing(mentions_and_concepts)
		# extract linked data from ontology
		linked_data = [(mention_and_concept[0], use_case_ontology.loc[use_case_ontology['label'].str.lower() == mention_and_concept[1]][['iri', 'label', 'semantic_area_label']].values[0].tolist()) for mention_and_concept in mentions_and_concepts if mention_and_concept[1] is not None]
		# filter out linked data 'semantic_area_label' == None
		linked_data = [linked_datum for linked_datum in linked_data if linked_datum[1][2] is not None]
		if raw:  # return mentions+concepts
			return linked_data
		else:  # return concepts within semantic areas
			# return linked concepts divided into semantic areas
			linked_concepts = {area: [] for area in set(use_case_ontology['semantic_area_label'].tolist()) if area is not None}
			for linked_datum in linked_data:
				linked_concepts[str(linked_datum[1][2])].append([linked_datum[1][0], linked_datum[1][1]])
			return linked_concepts

	# COLON SPECIFIC LINKING FUNCTIONS

	def ad_hoc_colon_linking(self, mention, labels, sim_thr=0.7):
		"""
		Perform set of colon ad hoc linking functions 

		Params: 
			mention (spacy.tokens.span.Span): entity mention extracted from text
			labels (list(spacy.token.span.Span)): list of concept labels from reference ontology
			sim_thr (float): keep candidates with sim score greater than or equal to sim_thr

		Returns: matched ontology concept label(s)
		"""

		if 'dysplasia' in mention.text:  # mention contains 'dysplasia'
			return self.link_colon_dysplasia(mention)
		elif 'carcinoma' in mention.text:  # mention contains 'carcinoma'
			return self.link_colon_adenocarcinoma(mention)
		elif 'hyperplastic' in mention.text:  # mention contains 'hyperplastic'
			return self.link_colon_hyperplastic_polyp(mention)
		elif 'biopsy' in mention.text:  # mention contains 'biopsy'
			return self.link_colon_biopsy(mention, labels, sim_thr)
		elif 'colon' == mention.text:  # mention matches 'colon' -- @smarchesin TODO: w/ better similarity self.link_colon_nos should be deprecated
			return self.link_colon_nos(mention)
		elif 'polyp' == mention.text:  # mention matches 'polyp' -- @smarchesin TODO: w/ better similarity self.link_colon_polyp should be deprecated
			return self.link_colon_polyp(mention)
		else:  # none of the ad hoc functions was required -- perform similarity-based linking
			return self.associate_mention2candidate(mention, labels, sim_thr)
			# return [[mention.text, None]]

	def link_colon_dysplasia(self, mention):
		"""
		Identify (when possible) the colon dysplasia grade and link the dysplasia mention to the correct concept 
		
		Params:
			mention (spacy.tokens.span.Span): (dysplasia) entity mention extracted from text
		
		Returns: matched ontology concept label(s)
		"""
		
		dysplasia_mention = mention.text
		# identify dysplasia grades within mention
		grades = [self.use_case_dysplasia[trigger] for trigger in self.use_case_dysplasia.keys() if trigger in dysplasia_mention]
		grades = set(itertools.chain.from_iterable(grades))
		if grades:  # at least one dysplasia grade identified
			return [[dysplasia_mention, grade] for grade in grades]
		else:  # no dysplasia grades identified - map to simple Colon Dysplasia
			return [[dysplasia_mention, 'colon dysplasia']]

	@staticmethod
	def link_colon_adenocarcinoma(mention):  # @smarchesin TODO: needs to be improved to handle a larger pool of 'metastatic' cases
		"""
		Link (adeno)carcinoma mentions to the correct concepts

		Params:
			mention (spacy.tokens.span.Span): (hyperplastic polyp) entity mention extracted from text

		Returns: matched ontology concept label
		"""

		carcinoma_mention = mention.text
		if 'metasta' in carcinoma_mention:  # metastatic adenocarcinoma found -- 'metasta' handles both metasta-tic and metasta-sis/ses
			return [[carcinoma_mention, 'metastatic adenocarcinoma']]
		else:  # colon adenocarcinoma found
			return [[carcinoma_mention, 'colon adenocarcinoma']]

	@staticmethod
	def link_colon_hyperplastic_polyp(mention):
		"""
		Link hyperplastic polyp mentions to the correct concepts

		Params:
			mention (spacy.tokens.span.Span): (hyperplastic polyp) entity mention extracted from text

		Returns: matched ontology concept label(s)
		"""

		hyperplastic_mention = mention.text
		# idenfity presence of hyperplastic adenomatous polyps
		if 'adenomatous' in hyperplastic_mention:  # hyperplastic adenomatous polyp found
			return [[hyperplastic_mention, 'colon hyperplastic polyp'], [hyperplastic_mention, 'adenoma']]
		elif 'inflammatory' in hyperplastic_mention:  # hyperplastic inflammatory polyp found
			return [[hyperplastic_mention, 'colon hyperplastic polyp'], [hyperplastic_mention, 'colon inflammatory polyp']]
		elif 'glands' in hyperplastic_mention:  # hyperplastic glands found - skip it
			return [[hyperplastic_mention, None]]
		else:  # hyperplastic polyp found
			return [[hyperplastic_mention, 'colon hyperplastic polyp']]

	@staticmethod
	def link_colon_nos(mention):
		"""
		Link colon mentions to the colon concept

		Params:
			mention (spacy.tokens.span.Span): (colon) entity mention extracted from text

		Returns: matched ontology concept label
		"""

		colon_mention = mention.text
		assert colon_mention == 'colon'
		return [[colon_mention, 'colon, nos']]

	@staticmethod
	def link_colon_polyp(mention):  # @smarchesin TODO: what about plural nouns?
		"""
		Link polyp mentions to the polyp concept

		Params:
			mention (spacy.tokens.span.Span): (polyp) entity mention extracted from text

		Returns: matched ontology concept label
		"""

		polyp_mention = mention.text
		assert polyp_mention == 'polyp'
		return [[polyp_mention, 'polyp of colon']]

	def link_colon_biopsy(self, mention, labels, sim_thr=0.7):  # @smarchesin TODO: too hardcoded? too simplistic? what about plurals?
		"""
		Link colon biopsy mentions and the associated anatomical locations (if any)

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text
			labels (list(spacy.token.span.Span)): list of concept labels from reference ontology
			sim_thr (float): keep candidates with sim score greater than or equal to sim_thr

		Returns: colon biopsy concept and matched anatomical locations (if any)
		"""

		if mention.text == 'biopsy':  # mention contains 'biopsy' only
			return [[mention.text, 'biopsy of colon']]
		elif mention.text == 'colon biopsy':  # mention contains 'colon biopsy'
			return [[mention.text, 'biopsy of colon'], [mention.text, 'colon, nos']]
		elif mention[:2].text == 'colon biopsy':  # 'colon biopsy' as first term - match rest of mention w/ similarity-based linking
			anatomical_location = self.associate_mention2candidate(mention[2:], labels, sim_thr)
			if anatomical_location[0][1]:  # anatomical location found within mention
				return [[mention.text, 'biopsy of colon']] + anatomical_location
			else:  # return 'colon, nos' because of 'colon' within biopsy mention
				return [[mention.text, 'biopsy of colon'], [mention.text, 'colon, nos']]
		elif mention[-2:].text == 'colon biopsy':  # 'colon biopsy' as last term - match rest of mention w/ similarity-based linking
			anatomical_location = self.associate_mention2candidate(mention[:-2], labels, sim_thr)
			if anatomical_location[0][1]:  # anatomical location found within mention
				return [[mention.text, 'biopsy of colon']] + anatomical_location
			else:  # return 'colon, nos' because of 'colon' within biopsy mention
				return [[mention.text, 'biopsy of colon'], [mention.text, 'colon, nos']]
		elif mention[0].text == 'biopsy':  # 'biopsy' as first term - match rest of mention w/ similarity-based linking
			if 'colon' not in mention.text:  # 'colon' not in mention
				return [[mention.text, 'biopsy of colon']] + self.associate_mention2candidate(mention[1:], labels, sim_thr)
			else:  # 'colon' in mention -- hard to handle appropriately, keep biopsy and colon as concepts
				return [[mention.text, 'biopsy of colon'], [mention.text, 'colon, nos']]
		elif mention[-1].text == 'biopsy':  # 'biopsy' as last term - match rest of mention w/ similarity-based linking
			if 'colon' not in mention.text:  # 'colon' not in mention
				return [[mention.text, 'biopsy of colon']] + self.associate_mention2candidate(mention[:-1], labels, sim_thr)
			else:  # 'colon' in mention -- hard to handle appropriately, keep biopsy and colon as concepts
				return [[mention.text, 'biopsy of colon'], [mention.text, 'colon, nos']]
		else:  # biopsy not BOS or EOS
			if 'colon' not in mention.text:  # 'colon' not in mention
				biopsy_idx = [idx for idx, term in enumerate(mention) if 'biopsy' in term.text][0]  # get 'biopsy' mention index
				pre_anatomical_location = [['', '']]
				post_anatomical_location = [['', '']]
				if mention[:biopsy_idx]:  # link mention before 'biopsy'
					pre_anatomical_location = self.associate_mention2candidate(mention[:biopsy_idx], labels, sim_thr)
				if mention[biopsy_idx+1:]:  # link mention after 'biopsy'
					post_anatomical_location = self.associate_mention2candidate(mention[biopsy_idx+1:], labels, sim_thr)
				if pre_anatomical_location[0][1] and post_anatomical_location[0][1]:  # both mentions matched
					return [[mention.text, 'biopsy of colon']] + pre_anatomical_location + post_anatomical_location
				elif pre_anatomical_location[0][1]:  # only pre mention matched
					return [[mention.text, 'biopsy of colon']] + pre_anatomical_location
				elif post_anatomical_location[0][1]:  # only post mention matched
					return [[mention.text, 'biopsy of colon']] + post_anatomical_location
				else:  # no mention matched - return only 'biopsy of colon' concept
					return [[mention.text, 'biopsy of colon']]
			else:  # 'colon' in mention -- hard to handle appropriately, keep biopsy and colon as concepts
				return [[mention.text, 'biopsy of colon'], [mention.text, 'colon, nos']]

	# COLON SPECIFIC POST PROCESSING OPERATIONS

	@staticmethod
	def ad_hoc_colon_post_processing(mentions_and_concepts):  # @TODO: use it if post processing operations are required for colon
		"""
		Perform set of post processing operations

		Params:
			mentions_and_concepts (list(list(str))): list of mentions and concepts extracted from report

		Returns: mentions and concepts after post processing operations
		"""

		return mentions_and_concepts

	# CERVIX SPECIFIC LINKING FUNCTIONS

	def ad_hoc_cervix_linking(self, mention, labels, sim_thr=0.7):
		"""
		Perform set of cervix ad hoc linking functions 

		Params: 
			mention (spacy.tokens.span.Span): entity mention extracted from text
			labels (list(spacy.token.span.Span)): list of concept labels from reference ontology
			sim_thr (float): keep candidates with sim score greater than or equal to sim_thr

		Returns: matched ontology concept label(s)
		"""

		if 'dysplasia' in mention.text or 'squamous intraepithelial lesion' in mention.text:  # mention contains 'dysplasia' or 'squamous intraepithelial lesion'
			return self.link_cervix_dysplasia(mention)
		elif re.search(r'\bcin\d*', mention.text) or re.search(r'sil\b', mention.text):  # mention contains 'cin' or 'sil'
			return self.link_cervix_cin(mention)
		elif 'hpv' in mention.text:  # mention contains 'hpv'
			return self.link_cervix_hpv(mention, labels, sim_thr)
		elif 'infection' in mention.text:  # mention contains 'infection'
			return self.skip_cervix_infection(mention)
		elif 'koilocyt' in mention.text:  # mention contains 'koilocyte'
			return self.link_cervix_koilocytes(mention)
		elif 'epithelium' in mention.text or 'junction' in mention.text:  # mention contains 'epithelium' or 'junction'
			return self.link_cervix_epithelium(mention)
		elif 'leep' in mention.text:  # mention contains 'leep'
			return self.link_cervix_leep(mention)
		elif 'biopsy' in mention.text and 'portio' in mention.text:  # mention contains 'biopsy portio'
			return self.link_cervix_conization(mention)
		else:  # none of the ad hoc functions was required -- perform similarity-based linking
			return self.associate_mention2candidate(mention, labels, sim_thr)

	def link_cervix_dysplasia(self, mention):
		"""
		Identify (when possible) the cervix dysplasia grade and link the dysplasia mention to the correct concept 
		
		Params:
			mention (spacy.tokens.span.Span): (dysplasia) entity mention extracted from text

		Returns: matched ontology concept label(s)
		"""
		
		dysplasia_mention = mention.text
		# identify dysplasia grades within mention
		grades = [self.use_case_dysplasia[trigger] for trigger in self.use_case_dysplasia.keys() if trigger in dysplasia_mention]
		grades = set(itertools.chain.from_iterable(grades))
		if grades:  # at least one dysplasia grade identified
			return [[dysplasia_mention, grade] for grade in grades]
		else:  # no dysplasia grades identified - map to simple CIN
			return [[dysplasia_mention, 'cervical intraepithelial neoplasia']]

	def link_cervix_cin(self, mention):
		"""
		Identify (when possible) the cervix cin/sil grade and link the cin/sil mention to the correct concept 
		
		Params:
			mention (spacy.tokens.span.Span): (cin) entity mention extracted from text
		
		Returns: matched ontology concept label(s)
		"""
		
		cin_mention = mention.text
		# identify cin/sil grades within mention
		grades = [self.cin[trigger] for trigger in self.cin.keys() if trigger in cin_mention]
		if grades:  # at least one cin/sil grade identified
			return [[cin_mention, grade] for grade in grades]
		else:  # no cin/sil grades identified - map to simple cin/sil
			return [[cin_mention, 'cervical intraepithelial neoplasia']]

	def link_cervix_hpv(self, mention, labels, sim_thr=0.7):  # @smarchesin TODO: too hardcoded? too simplistic?
		"""
		Link cervix hpv mentions and the associated anatomical locations (if any)

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text
			labels (list(spacy.token.span.Span)): list of concept labels from reference ontology
			sim_thr (float): keep candidates with sim score greater than or equal to sim_thr

		Returns: cervix hpv concept and matched anatomical locations (if any)
		"""

		if mention.text == 'hpv':  # mention contains 'hpv' only
			return [[mention.text, 'human papilloma virus infection']]
		elif mention.text == 'hpv infection':  # mention contains 'hpv infection'
			return [[mention.text, 'human papilloma virus infection']]
		elif mention[:2].text == 'hpv infection':  # 'hpv infection' as first term - match rest of mention w/ similarity-based linking
			return [[mention[:2].text, 'human papilloma virus infection']] + self.associate_mention2candidate(mention[2:], labels, sim_thr)
		elif mention[-2:].text == 'hpv infection':  # 'hpv infection' as last term - match rest of mention w/ similarity-based linking
			return [[mention[-2:].text, 'human papilloma virus infection']] + self.associate_mention2candidate(mention[:-2], labels, sim_thr)
		elif mention[0].text == 'hpv':  # 'hpv' as first term - match rest of mention w/ similarity-based linking
			return [[mention[0].text, 'human papilloma virus infection']] + self.associate_mention2candidate(mention[1:], labels, sim_thr)
		elif mention[-1].text == 'hpv':  # 'hpv' as last term - match rest of mention w/ similarity-based linking
			return [[mention[-1].text, 'human papilloma virus infection']] + self.associate_mention2candidate(mention[:-1], labels, sim_thr)
		else:  # biopsy not BOS or EOS
			hpv_idx = [idx for idx, term in enumerate(mention) if 'hpv' in term.text][0]  # get 'hpv' mention index 
			pre_anatomical_location = [['', '']]
			post_anatomical_location = [['', '']]
			if mention[:hpv_idx]:  # link mention before 'hpv'
				pre_anatomical_location = self.associate_mention2candidate(mention[:hpv_idx], labels, sim_thr)
			if mention[hpv_idx+1:]:  # link mention after 'hpv'
				post_anatomical_location = self.associate_mention2candidate(mention[hpv_idx+1:], labels, sim_thr)
			if pre_anatomical_location[0][1] and post_anatomical_location[0][1]:  # both mentions matched
				return [[mention[hpv_idx].text, 'human papilloma virus infection']] + pre_anatomical_location + post_anatomical_location
			elif pre_anatomical_location[0][1]:  # only pre mention matched
				return [[mention[hpv_idx].text, 'human papilloma virus infection']] + pre_anatomical_location
			elif post_anatomical_location[0][1]:  # only post mention matched
				return [[mention[hpv_idx].text, 'human papilloma virus infection']] + post_anatomical_location
			else:  # no mention matched - return only 'human papilloma virus infection' concept
				return [[mention[hpv_idx].text, 'human papilloma virus infection']]

	@staticmethod
	def link_cervix_epithelium(mention):
		"""
		Identify (when possible) the cervix epithelium type and link it to the correct concept 

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text

		Returns: cervix epithelium concept
		"""

		epithelium_mention = mention.text
		# identify epithelium types within mention
		if 'simple' in epithelium_mention:
			return [[epithelium_mention, 'simple epithelium']]
		elif 'pavement' in epithelium_mention:
			return [[epithelium_mention, 'pavement epithelium']]
		elif 'junction' in epithelium_mention:
			return [[epithelium_mention, 'cervical squamo-columnar junction']]
		elif 'ectocervical' in epithelium_mention:
			return [[epithelium_mention, 'exocervical epithelium']]
		elif 'exocervical' in epithelium_mention:
			return [[epithelium_mention, 'exocervical epithelium']]
		elif 'glandular' in epithelium_mention:
			return [[epithelium_mention, 'cervix glandular epithelium']]
		elif 'squamous' in epithelium_mention:
			return [[epithelium_mention, 'cervix squamous epithelium']]
		else:  # epithelium type not found
			return [[epithelium_mention, 'cervix epithelium']]

	@staticmethod
	def link_cervix_leep(mention):
		"""
		Link cervical leep mention to the correct concept 

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text

		Returns: cervical leep concept
		"""

		leep_mention = mention.text
		assert 'leep' in leep_mention
		return [[leep_mention, 'loop electrosurgical excision']]

	@staticmethod
	def link_cervix_conization(mention):
		"""
		Link biopsy portio mention to conization concept

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text

		Returns: conization concept
		"""

		conization_mention = mention.text
		assert 'biopsy' in conization_mention and 'portio' in conization_mention
		return [[conization_mention, 'conization']]

	@staticmethod
	def link_cervix_koilocytes(mention):
		"""
		Link koilocyte mention to koilocytotic squamous cell concept

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text

		Returns: koilocytotic squamous cell concept
		"""

		koilocyte_mention = mention.text
		assert 'koilocyt' in koilocyte_mention
		return [[koilocyte_mention, 'koilocytotic squamous cell']]

	@staticmethod
	def skip_cervix_infection(mention):  # @smarchesin TODO: remove the third case (else condition) after testing
		"""
		Skip 'infection' mentions that are associated to 'hpv'

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text

		Returns: cervix concept if 'infection' is not associated to 'hpv' or None otherwise
		"""

		if mention.text == 'infection':  # mention contains 'infection' only -- skip it
			return [[mention.text, None]]
		elif mention.text == 'viral infection':  # mention contains 'viral infection' only -- skip it
			return [[mention.text, None]]
		else:  # mention contains other terms other than 'infection' -- unhandled
			print('mention contains unhandled "infection" mention -- set temp to None')
			print(mention.text)
			return [[mention.text, None]]

	# CERVIX SPECIFIC POST PROCESSING OPERATIONS

	def ad_hoc_cervix_post_processing(self, mentions_and_concepts):
		"""
		Perform set of post processing operations

		Params:
			mentions_and_concepts (list(list(str))): list of mentions and concepts extracted from report

		Returns: mentions and concepts after post processing operations
		"""

		return self.associate_cervix_in_situ_invasive_concepts(mentions_and_concepts)

	@staticmethod
	def associate_cervix_in_situ_invasive_concepts(mentions_and_concepts):
		"""
		Associate in situ/invasive cervical adenocarcinoma to the corresponding concepts

		Params:
			mentions_and_concepts (list(list(str))): list of mentions and concepts extracted from report

		Returns: mentions and concepts after in situ/invasive association
		"""

		# set invasive flag
		invasive = any([m_and_c for m_and_c in mentions_and_concepts if 'invasive' in m_and_c[0]])
		# remove 'invasive' or 'invasive growth' from mentions_and_concepts
		ixs = [
			ix for ix, m_and_c in enumerate(mentions_and_concepts)
			if 'invasive' == m_and_c[0] or 'invasive growth' == m_and_c[0] or 'growth invasive' == m_and_c[0]]
		new_mentions_and_concepts = []
		for ix, m_and_c in enumerate(mentions_and_concepts):
			if ix in ixs:  # skip m_and_c because it refers to invasive mentions
				continue
			new_mentions_and_concepts.append(m_and_c)

		# set squamous cell carcinoma/adenocarcinoma in situ and invasive concepts
		sqcc = ['squamous carcinoma in situ', 'cervical squamous cell carcinoma']
		adeno = ['cervical adenocarcinoma in situ', 'cervical adenocarcinoma']

		# get squamous cell carcinoma/adenocarcinoma mentions and concepts
		sqcc_mcs = [(ix, m_and_c) for ix, m_and_c in enumerate(new_mentions_and_concepts) if m_and_c[1] in sqcc]
		adeno_mcs = [(ix, m_and_c) for ix, m_and_c in enumerate(new_mentions_and_concepts) if m_and_c[1] in adeno]

		# replace wrong in situ/invasive concepts when found
		if invasive:  # invasive squamous cell carcinoma or adenocarcinoma
			for (ix, m_and_c) in sqcc_mcs:  # loop over squamous cell carcinoma mentions
				if m_and_c[1] == sqcc[0]:  # sqcc in situ found
					if 'in situ' in m_and_c[0]:  # keep in situ as contained within mention
						continue
					else:  # replace sqcc in situ with sqcc invasive
						new_mentions_and_concepts[ix] = [m_and_c[0], sqcc[1]]
			for (ix, m_and_c) in adeno_mcs:  # loop over cervical adenocarcinoma mentions
				if m_and_c[1] == adeno[0]:  # adeno in situ found
					if 'in situ' in m_and_c[0]:  # keep in situ as contained within mention
						continue
					else:  # replace adeno in situ with adeno invasive
						new_mentions_and_concepts[ix] = [m_and_c[0], adeno[1]]
		else:  # in situ squamous cell carcinoma or adenocarcinoma
			for (ix, m_and_c) in sqcc_mcs:  # loop over squamous cell carcinoma mentions
				if m_and_c[1] == sqcc[1]:  # sqcc invasive found
					# replace sqcc invasive with sqcc in situ
					new_mentions_and_concepts[ix] = [m_and_c[0], sqcc[0]]
			for (ix, m_and_c) in adeno_mcs:  # loop over cervical adenocarcinoma mentions
				if m_and_c[1] == adeno[1]:  # adeno invasive found
					# replace adeno invasive with adeno in situ
					new_mentions_and_concepts[ix] = [m_and_c[0], adeno[0]]
		# return post processed mentions and concepts
		return new_mentions_and_concepts

	# LUNG SPECIFIC LINKING FUNCTIONS

	def ad_hoc_lung_linking(self, mention, labels, sim_thr=0.7):
		"""
		Perform set of lung ad hoc linking functions

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text
			labels (list(spacy.token.span.Span)): list of concept labels from reference ontology
			sim_thr (float): keep candidates with sim score greater than or equal to sim_thr

		Returns: matched ontology concept label(s)
		"""

		return self.associate_mention2candidate(mention, labels, sim_thr)

	# CERVIX SPECIFIC POST PROCESSING OPERATIONS

	@staticmethod
	def ad_hoc_lung_post_processing(mentions_and_concepts):  # TODO: use this if post processing operations are required for lung
		"""
		Perform set of post processing operations

		Params:
			mentions_and_concepts (list(list(str))): list of mentions and concepts extracted from report

		Returns: mentions and concepts after post processing operations
		"""

		return mentions_and_concepts

	# ONTOLOGY-RELATED FUNCTIONS

	def process_ontology_concepts(self, labels):
		"""
		Process ontology labels using scispaCy

		Params:
			labels (list): list of concept labels

		Returns: a list/dict of processed concept labels
		"""

		proc_labels = []
		if self.biow2v or self.gpm or self.biofast_model:  # process onto concepts for biow2v, gpm, and biofast
			proc_labels.append([self.nlp(label) for label in labels])

		if self.bert_model:  # process onto concepts for BERT
			tokens = utils.assign_gpu(self.bert_tokenizer(labels, return_tensors="pt", padding=True), self.gpu)  # get tokens w/ padding
			embs = self.bert_model(**tokens)[0]  # get BERT last layer hidden states
			exp_attention_mask = tokens['attention_mask'].unsqueeze(-1).expand(embs.size())  # broadcast attention mask to embs.size
			pooled_embs = torch.sum(embs * exp_attention_mask, 1) / exp_attention_mask.sum(1)  # compute pooling to obtain label embeddings -- exp_attention_mask compute proper average (no [PAD])
			proc_labels.append([pooled_embs[ix].cpu().detach().numpy() for ix, label in enumerate(labels)])

		if len(proc_labels) == 2:
			return {label: [proc_labels[0][i], proc_labels[1][i]] for i, label in enumerate(labels)}
		elif len(proc_labels) == 1:
			return {label: [proc_labels[0][i]] for i, label in enumerate(labels)}
		else:  # raise exception
			print('No semantic matching method selected.\nPlease select any combination of: "biow2v", "str_match", "biofast", and "biobert"')
			raise Exception

	@staticmethod
	def lookup_snomed_codes(snomed_codes, use_case_ontology):
		"""
		Lookup for ontology concepts associated to target SNOMED codes

		Params:
			snomed_codes (list(str)/str): target SNOMED codes
			use_case_ontology (pandas DataFrame): reference ontology restricted to the use case considered

		Returns: a dict of identified ontology concepts {semantic_area: [iri, label], ...}
		"""
		
		lookups = {area: [] for area in set(use_case_ontology['semantic_area_label'].tolist()) if area is not None}
		if type(snomed_codes) == list:  # search for list of snomed codes
			snomed_codes = [code for code in snomed_codes if code]
			if snomed_codes:
				linked_data = use_case_ontology.loc[use_case_ontology['SNOMED'].isin(snomed_codes)][['iri', 'label', 'semantic_area_label']]
				if not linked_data.empty:  # matches found within ontology
					for linked_datum in linked_data.values.tolist():
						lookups[str(linked_datum[2])].append([linked_datum[0], linked_datum[1]])
			return lookups
		else:  # search for single snomed code
			if snomed_codes:
				linked_data = use_case_ontology.loc[use_case_ontology['SNOMED'] == snomed_codes][['iri', 'label', 'semantic_area_label']]
				if not linked_data.empty:  # match found within ontology
					linked_datum = linked_data.values[0].tolist()
					lookups[str(linked_datum[2])].append([linked_datum[0], linked_datum[1]])
			return lookups

	# AOEC SPECIFIC FUNCTIONS

	def aoec_entity_linking(self, reports, onto_proc, use_case_ontology, labels, use_case, sim_thr=0.7):
		"""
		Perform entity linking over translated AOEC reports
		
		Params:
			reports (dict): target reports
			onto_proc (OntologyProc): instance of OntologyProc class
			use_case_ontology (pandas.core.frame.DataFrame): ontology data restricted to given use case
			labels (list(spacy.tokens.doc.Doc)): list of processed ontology concepts
			use_case (str): the use_case considered - i.e. colon, lung, cervix, or celiac
			sim_thr (float): keep candidates with sim score greater than or equal to sim_thr
			
		Returns: a dict containing the linked concepts for each report w/ distinction between 'nlp' and 'struct' concepts
		"""
		
		concepts = dict()
		# loop over AOEC reports and perform linking
		for rid, rdata in tqdm(reports.items()):
			concepts[rid] = dict()

			# sanitize diagnosis
			diagnosis = utils.en_sanitize_record(rdata['diagnosis_nlp'], use_case)
			# extract entity mentions from diagnosis
			diagnosis = self.extract_entity_mentions(diagnosis)

			# sanitize materials
			materials = utils.en_sanitize_record(rdata['materials'], use_case)
			if use_case == 'colon':  # consider 'polyp' as a stopwords in materials @smarchesin TODO: what about the other use cases?
				materials = re.sub('polyp[s]?(\s|$)+', ' ', materials)
			# extract entity mentions from materials
			materials = self.extract_entity_mentions(materials)

			# combine diagnosis and materials mentions
			mentions = diagnosis + materials
			# link and store 'nlp' concepts
			nlp_concepts = self.link_mentions_to_concepts(mentions, labels, use_case_ontology, sim_thr)
			# link and store 'struct' concepts
			struct_concepts = self.lookup_snomed_codes(
				utils.sanitize_codes(rdata['diagnosis_struct']) +
				utils.sanitize_codes(rdata['procedure']) +
				utils.sanitize_codes(rdata['topography']), use_case_ontology)
			# merge 'nlp' and 'struct' concepts 
			concepts[rid] = onto_proc.merge_nlp_and_struct(nlp_concepts, struct_concepts)
		# return concepts divided into 'nlp' and 'struct' sections (used for debugging, evaluation, and other applications)
		return concepts

	# RADBOUD SPECIFIC FUNCTIONS

	def radboud_entity_linking(self, reports, use_case_ontology, labels, use_case, sim_thr=0.7):
		"""
		Perform entity linking over translated and processed Radboud reports

		Params:
			reports (dict): target reports 
			use_case_ontology (pandas.core.frame.DataFrame): ontology data restricted to given use case
			labels (list(spacy.tokens.doc.Doc)): list of processed ontology concepts
			use_case (str): the use_case considered - i.e. colon, lung, cervix, or celiac
			sim_thr (float): keep candidates with sim score greater than or equal to sim_thr

		Returns: a dict containing the linked concepts for each report w/ list of associated slides
		"""
		
		concepts = dict()
		# loop over Radboud processed reports and perform linking
		for rid, rdata in tqdm(reports.items()):
			concepts[rid] = dict()
			# extract entity mentions from conclusions
			mentions = self.extract_entity_mentions(utils.en_sanitize_record(rdata['diagnosis'], use_case))
			# link and store concepts from conclusions
			nlp_concepts = self.link_mentions_to_concepts(mentions, labels, use_case_ontology, sim_thr)
			# assign conclusion concepts to concepts dict
			concepts[rid]['concepts'] = nlp_concepts
			# assign slide ids to concepts dict if present
			if 'slide_ids' in rdata:
				concepts[rid]['slide_ids'] = rdata['slide_ids']
		# return linked concepts divided per diagnosis
		return concepts

	# GENERAL-PURPOSE FUNCTIONS

	def entity_linking(self, reports, use_case_ontology, labels, use_case, sim_thr=0.7, raw=False):
		"""
		Perform entity linking over translated and processed reports

		Params:
			reports (dict): target reports
			use_case_ontology (pandas.core.frame.DataFrame): ontology data restricted to given use case
			labels (list(spacy.tokens.doc.Doc)): list of processed ontology concepts
			use_case (str): the use_case considered - i.e. colon, lung, cervix, or celiac
			sim_thr (float): keep candidates with sim score greater than or equal to sim_thr
			raw (bool): whether to return concepts within semantic areas or mentions+concepts

		Returns: a dict containing the linked concepts for each report
		"""

		concepts = dict()
		# loop over translated and processed reports and perform linking
		for rid, rdata in tqdm(reports.items()):
			concepts[rid] = dict()
			# extract entity mentions from text
			mentions = self.extract_entity_mentions(utils.en_sanitize_record(rdata['text'], use_case))
			# link and store concepts from text
			concepts[rid] = self.link_mentions_to_concepts(mentions, labels, use_case_ontology, sim_thr, raw)

		# return concepts divided per diagnosis
		return concepts
