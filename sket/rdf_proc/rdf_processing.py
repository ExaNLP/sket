import os
import uuid
import hashlib
import datetime
import itertools

from rdflib import URIRef, Literal, Graph


class RDFProc(object):

	def __init__(self):  # @smarchesin TODO: add Test outcome to predicate2literal list (?why?)
		"""
		Set namespaces and properties w/ Literals as objects

		Params:
		
		Returns: None
		"""

		# set namespaces
		self.namespace = {'exa': 'https://w3id.org/examode/', 'dc': 'http://purl.org/dc/elements/1.1/'}
		self.gender = {'M': 'http://purl.obolibrary.org/obo/NCIT_C46109', 'F': 'http://purl.obolibrary.org/obo/NCIT_C46110'}
		self.age_set = {
			'young': 'https://hpo.jax.org/app/browse/term/HP:0011462',
			'middle': 'https://hpo.jax.org/app/browse/term/HP:0003596',
			'late': 'https://hpo.jax.org/app/browse/term/HP:0003584'
		}
		self.predicate2literal = [
			self.namespace['dc'] + 'identifier',
			self.namespace['exa'] + 'ontology/#hasDiagnosisText',
			self.namespace['exa'] + 'ontology/#hasAge',
			self.namespace['exa'] + 'ontology/#hasGender',
			self.namespace['exa'] + 'ontology/#hasImage',
			self.namespace['exa'] + 'ontology/#hasBlockNumber',
			self.namespace['exa'] + 'ontology/#hasSlideId',
			self.namespace['exa'] + 'ontology/#detectedHumanPapillomaVirus',
			self.namespace['exa'] + 'ontology/#koylociteDetected'
		]

	# COMMON FUNCTIONS

	@staticmethod
	def compute_age(birth_date, visit_date):
		"""
		Compute patient's age at the time of visit

		Params:
			birth_date (str): patient's birth date
			visit_date (str): visit date

		Returns: patient's age
		"""

		try:
			bd = datetime.datetime.strptime(birth_date[:8], '%Y%m%d')
			vd = datetime.datetime.strptime(visit_date[:8], '%Y%m%d')

			age = vd.year - bd.year - ((vd.month, vd.day) < (bd.month, bd.day))
		except Exception:
			age = None
		return age

	@staticmethod
	def associate_polyp2dysplasia(outcomes, mask_outcomes, debug=False):  # @smarchesin TODO: needs to be part of a dict of functions that are use-case dependants
		"""
		Associate polyp-type outcomes w/ dysplasia mentions (colon-related function)
		
		Params:
			outcomes (list(pair(str))): the list of report outcomes
			mask_outcomes (str): the string representing masked outcomes - 0 for dysplasia and 1 for polyp (sub)classes
			debug (bool): whether to keep flags for debugging
		
		Returns: a list of associated polyp-dysplasia pairs, when possible, or single polyp mentions
		"""
		
		pairs = list()
		if len(mask_outcomes) == 1:  # there is one outcome only 
			if mask_outcomes == '1':  # outcome is subclass of polyp class
				pairs.append([outcomes[0][0]])
				return pairs
			else:  # outcome is a dysplasia mention - append it w/ the general 'Polyp of Colon' concept 
				if debug:
					print('decoupled dysplasia mention')
				pairs.append(['http://purl.obolibrary.org/obo/MONDO_0021400', outcomes[0][0]])
				return pairs
		# group masked_outcomes by '0' -- e.g., 000111001000 --> ['000', '1', '1', '1', '00', '1', '000']
		mask_outcomes = [["".join(g)] if k == '0' else list(g) for k, g in itertools.groupby(mask_outcomes)]
		mask_outcomes = [item for sublist in mask_outcomes for item in sublist]
		for ix, out in enumerate(mask_outcomes):  # multiple outcomes available
			if ix == 0:  # first outcome
				if out[0] == '0':  # first outcome is a dysplasia mention
					if debug:
						print('decoupled dysplasia mention(s)')
					dysplasia_outcomes = [outcomes[ix+oix][0] for oix in range(len(out))]
					pairs.append(['http://purl.obolibrary.org/obo/MONDO_0021400'] + dysplasia_outcomes)
				else:  # first outcome is subclass of polyp class
					if mask_outcomes[ix+1][0] == '0':  # the second outcome is a dysplasia mention
						dysplasia_outcomes = [outcomes[ix+1+oix][0] for oix in range(len(mask_outcomes[ix+1]))]
						pairs.append([outcomes[ix][0]] + dysplasia_outcomes)
					else:  # the second outcome is another subclass of polyp class
						pairs.append([outcomes[ix][0]])
			else:
				if out[0] == '0':  # outcome is a dysplasia mention - skip it
					continue
				else:  # outcome is subclass of polyp class
					if ix+1 < len(mask_outcomes):  # ix != last
						if mask_outcomes[ix+1][0] == '0':  # succeeding outcome is a dysplasia mention
							dysplasia_outcomes = [outcomes[ix+1+oix][0] for oix in range(len(mask_outcomes[ix+1]))]
							pairs.append([outcomes[ix][0]] + dysplasia_outcomes)
						else:  # the succeeding outcome is another subclass of polyp class
							pairs.append([outcomes[ix][0]])
					else:  # ix == last
						pairs.append([outcomes[ix][0]])
		return pairs

	def serialize_report_graphs(self, graphs, output='stream', rdf_format='turtle'):
		"""
		Serialize report graphs into rdf w/ specified format

		Params:
			graphs (list(list(tuple))/list(tuple)): report graphs to be serialized
			output (str): output path of the serialized graph - if output == 'stream' --> return streamed output
			rdf_format (str): rdf serialization format

		Returns: the serialized rdf graph
		"""

		g = Graph()
		# bind namespaces to given prefix
		for px, ns in self.namespace.items():
			g.bind(px, ns, override=True)
		# loop over graphs and convert them into rdflib classes
		for graph in graphs:
			for triple in graph:
				s = URIRef(triple[0])
				p = URIRef(triple[1])
				if triple[1] in self.predicate2literal:
					o = Literal(triple[2])
				else:
					o = URIRef(triple[2])
				g.add((s, p, o))
		if output == 'stream':  # stream rdf graph to output 
			# serialize graphs into predefined rdf format
			return g.serialize(format=rdf_format)
		else:  # store rdf graph to file
			os.makedirs(os.path.dirname(output), exist_ok=True)
			# serialize graphs into predefined rdf format
			g.serialize(destination=output, format=rdf_format)
			print('rdf graph serialized to {} with {} format'.format(output, rdf_format))
			return True

	# AOEC SPECIFIC FUNCTIONS

	def aoec_create_graph(self, rid, report_data, report_concepts, onto_proc, use_case, debug=False):
		"""
		Create the AOEC rdf report graph out of extracted concepts
		
		Params:
			rid (str): the report id
			report_data (dict): the target report data
			report_concepts (dict): the concepts extracted from the target report
			onto_proc (OntologyProc): instance of OntologyProc class
			use_case (str): the use_case considered - i.e. colon, lung, cervix, or celiac
			debug (bool): whether to keep flags for debugging
		
		Returns: a list of (s, p, o) triples representing report data in rdf and a dict containing structured info
		"""
		
		rdf_graph = []
		struct_graph = {}

		# generate report id hashing 'diagnosis' field
		hrid = hashlib.md5(rid.encode()).hexdigest()
		struct_graph['ReportID'] = hrid
		
		# create report data-related triples

		# build the IRI for the resource
		resource = self.namespace['exa'] + 'resource/'

		# build the IRI for the given report
		report = resource + 'report/' + hrid
		struct_graph['ReportURL'] = report

		# build the IRI for the use-case ClinicalCaseReport
		use_case_ccreport = self.namespace['exa'] + 'ontology/#' + use_case.capitalize() + 'ClinicalCaseReport'
		struct_graph['ClinicalCase'] = use_case_ccreport

		# generate report data-related triples
		rdf_graph.append((report, 'a', use_case_ccreport))
		rdf_graph.append((report, self.namespace['dc'] + 'identifier', hrid))
		if report_data['diagnosis_nlp']:  # textual diagnosis is present within report data
			rdf_graph.append((report, self.namespace['exa'] + 'ontology/#hasDiagnosisText', report_data['diagnosis_nlp']))
			struct_graph['hasDiagnosisText'] = report_data['diagnosis_nlp']
		if 'image' in report_data:  # report belongs to v2
			rdf_graph.append((report, self.namespace['exa'] + 'ontology/#hasImage', report_data['image']))
		if 'internalid' in report_data:  # report belongs to v2
			rdf_graph.append((report, self.namespace['exa'] + 'ontology/#hasBlockNumber', report_data['internalid']))
			struct_graph['hasBlockNumber'] = report_data['internalid']
		else:  # report belongs to v1
			if len(rid.split('_')) == 1:  # no internalid specified - set to 1
				rdf_graph.append((report, self.namespace['exa'] + 'ontology/#hasBlockNumber', '1'))
				struct_graph['hasBlockNumber'] = '1'
			else:
				rdf_graph.append((report, self.namespace['exa'] + 'ontology/#hasBlockNumber', rid.split('_')[1]))
				struct_graph['hasBlockNumber'] = rid.split('_')[1]

		# create patient-related triples

		struct_graph['patient'] = {}

		# generate patient id hashing the first part of the 'rid' field
		pid = hashlib.md5(rid.split('_')[0].encode()).hexdigest()
		# build the IRI for the patient
		patient = self.namespace['exa'] + 'resource/patient/' + pid
		struct_graph['patient']['PatientURL'] = patient

		# generate patient-related triples
		rdf_graph.append((patient, 'a', 'http://purl.obolibrary.org/obo/IDOMAL_0000603'))
		struct_graph['patient']['a'] = 'http://purl.obolibrary.org/obo/IDOMAL_0000603'

		# associate report to patient
		rdf_graph.append((patient, self.namespace['exa'] + 'ontology/#hasClinicalCaseReport', report))
		# associate age to patient
		age = None
		if 'age' in report_data:  # age data is present within report_data
			age = report_data['age']
		elif report_data['birth_date'] and report_data['visit_date']:  # birth date and visit date are present - compute age
			age = self.compute_age(report_data['birth_date'], report_data['visit_date'])
		if age:  # age found within current report
			# associate age to report
			rdf_graph.append((patient, self.namespace['exa'] + 'ontology/#hasAge', age))
			struct_graph['patient']['hasAge'] = age

			# convert age to age set and associate to report
			if age < 40:  # young 
				rdf_graph.append((patient, self.namespace['exa'] + 'ontology/#hasAgeOnset', self.age_set['young']))
				struct_graph['patient']['hasAgeOnset'] = self.age_set['young']
			elif 40 <= age < 60:  # middle
				rdf_graph.append((patient, self.namespace['exa'] + 'ontology/#hasAgeOnset', self.age_set['middle']))
				struct_graph['patient']['hasAgeOnset'] = self.age_set['middle']
			else:  # late
				rdf_graph.append((patient, self.namespace['exa'] + 'ontology/#hasAgeOnset', self.age_set['late']))
				struct_graph['patient']['hasAgeOnset'] = self.age_set['late']
		# associate gender to patient
		if report_data['gender']:  # gender data is present within report_data
			rdf_graph.append((patient, self.namespace['exa'] + 'ontology/#hasGender', self.gender[report_data['gender']]))
			struct_graph['patient']['hasGender'] = self.gender[report_data['gender']]
			struct_graph['patient']['hasGenderLiteral'] = report_data['gender']

		# create report concept-related triples

		# set ontology 'Outcome' IRI to identify its descendants within the 'Diagnosis' section
		ontology_outcome = 'http://purl.obolibrary.org/obo/NCIT_C20200'
		if use_case == 'colon':
			# set ontology 'Polyp' IRI to identify its descendants within the 'Diagnosis' section
			ontology_polyp = 'http://purl.obolibrary.org/obo/MONDO_0021400'
		elif use_case == 'cervix':
			# set ontology 'Human Papilloma Virus Infection' and 'Koilocytotic Squamous Cell' IRIs to identify such outcomes within 'Diagnosis'
			ontology_hpv = 'http://purl.obolibrary.org/obo/MONDO_0005161'
			ontology_koilocyte = 'http://purl.obolibrary.org/obo/NCIT_C36808'

		# identify report procedures
		report_procedures = [procedure[0] for procedure in report_concepts['Procedure']]

		# identify report anatomical locations
		report_locations = [location[0] for location in report_concepts['Anatomical Location']]
		if not report_locations:  # @smarchesin TODO: decide how to handle cervix when location is absent
			if use_case == 'colon':
				# add 'Colon, NOS' IRI as default
				report_locations += ['http://purl.obolibrary.org/obo/UBERON_0001155']
		if use_case != 'cervix':  # @smarchesin TODO: this might be updated depending on the other use cases
			# identify report tests
			report_tests = [test[0] for test in report_concepts['Test']]

		# identify report outcomes
		report_outcomes = [
			(diagnosis[0], onto_proc.get_higher_concept(iri1=diagnosis[0], iri2=ontology_outcome)) for diagnosis in report_concepts['Diagnosis']]
		if use_case == 'colon':
			# identify report polyps
			report_polyps = [
				(diagnosis[0], onto_proc.get_higher_concept(iri1=diagnosis[0], iri2=ontology_polyp, include_self=True))
				for diagnosis in report_concepts['Diagnosis']]
			# restrict report_polyps/report_outcomes to those polyp-related/non polyp-related and mask concepts that are sublcass of Polyp w/ 1 and 0 otherwise (dysplasia-related concepts)
			for ix, (outcome, polyp) in enumerate(zip(report_outcomes, report_polyps)):
				if (outcome[1] is not None) and (polyp[1] is None):  # non-polyp outcome
					report_polyps.pop(ix)
				elif (outcome[1] is not None) and (polyp[1] is not None):  # polyp outcome
					report_outcomes.pop(ix)
				else:  # dysplasia outcome
					report_outcomes.pop(ix)
			# mask report_polyps w/ 1 for concepts that are subclass of Polyp and 0 for other 
			masked_outcomes = ''.join(['0' if report_outcome[1] is None else '1' for report_outcome in report_polyps])
			# associate polyp-related mentions w/ dysplasia ones - restricted to colon disease only
			paired_outcomes = self.associate_polyp2dysplasia(report_polyps, masked_outcomes, debug=debug)
			# concatenate the non-polyp outcomes to paired_outcomes
			paired_outcomes += [[outcome[0]] for outcome in report_outcomes]
		else:  # @smarchesin TODO: the 'else' will probably be replaced with use-case dependants if conditions
			paired_outcomes = [[outcome[0]] for outcome in report_outcomes]

		# set the counter for outcomes identified from report
		report_outcome_n = 0

		struct_graph['hasOutcome'] = []
		# loop over outcomes and build 'Outcome'-related triples
		for pair in paired_outcomes:  # @smarchesin TODO: handle negative results -- i.e., when len(paired_outcomes) == 0

			outcomes_struct = {}
			# increase outcomes counter
			report_outcome_n += 1

			# 'Diagnosis'-related triples

			# build the IRI for the identified outcome
			resource_outcome = resource + hrid + '/' + str(report_outcome_n)
			outcomes_struct['OutcomeURL'] = resource_outcome

			# attach outcome instance to graph
			rdf_graph.append((report, self.namespace['exa'] + 'ontology/#hasOutcome', resource_outcome))
			# specify what the resource_outcome is
			if use_case == 'cervix':
				if pair[0] == ontology_hpv:  # outcome is hpv infection
					rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#detectedHumanPapillomaVirus', pair[0]))
					outcomes_struct['detectedHumanPapillomaVirus'] = pair[0]
				if pair[0] == ontology_koilocyte:  # outcome is koilocyte
					rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#koylociteDetected', pair[0]))
					outcomes_struct['koylociteDetected'] = pair[0]
			else:  # regular outcome
				rdf_graph.append((resource_outcome, 'a', pair[0]))
				outcomes_struct['a'] = pair[0]

			if use_case == 'colon' and len(pair) > 1:  # target outcome has associated dysplasia
				outcomes_struct['hasDysplasia'] = []

				for dysplasia_outcome in pair[1:]:
					# specify the associated dysplasia
					rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#hasDysplasia', dysplasia_outcome))
					outcomes_struct['hasDysplasia'].append(dysplasia_outcome)

			# 'Anatomical'-related triples

			outcomes_struct['hasLocation'] = []
			# specify the anatomical location associated to target outcome
			for location in report_locations:  # @smarchesin TODO: correct? we might link multiple locations to the same outcome
				rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#hasLocation', location))
				outcomes_struct['hasLocation'].append(location)

			# 'Procedure'-related triples

			outcomes_struct['hasIntervention'] = []
			# loop over procedures and build 'Procedure'-related triples
			for ix, procedure in enumerate(report_procedures):  # @smarchesin TODO: correct? we might link the same procedure to multiple outcomes
				intervention_struct = {}

				# build the IRI for the identified procedure
				resource_procedure = resource + 'procedure/' + hrid + '/' + str(report_outcome_n) + '.' + str(ix+1)
				# attach procedure instance to graph
				rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#hasIntervention', resource_procedure))
				intervention_struct['InterventionURL'] = resource_procedure

				# specify what the resource_procedure is
				rdf_graph.append((resource_procedure, 'a', procedure))
				intervention_struct['a'] = procedure

				intervention_struct['hasTopography'] = []
				# specify the anatomical location associated to target procedure
				for location in report_locations:  # @smarchesin TODO: correct? we might associate multiple locations to the same procedure
					rdf_graph.append((resource_procedure, self.namespace['exa'] + 'ontology/#hasTopography', location))
					intervention_struct['hasTopography'].append(location)

				outcomes_struct['hasIntervention'].append(intervention_struct)

			# 'Test'-related triples - @smarchesin TODO: decide how to handle tests

			outcomes_struct['hasTest'] = []
			if use_case != 'cervix':
				for test in report_tests:  # @smarchesin TODO: is it correct? in this way we can associate multiple tests to the same outcome
					rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#hasTest', test))
					outcomes_struct['hasTest'].append(test)

			struct_graph['hasOutcome'].append(outcomes_struct)

		# return report rdf graph both as (s,p,o) list and dict
		return rdf_graph, struct_graph

	# RADBOUD SPECIFIC FUNCTIONS

	def radboud_create_graph(self, rid, report_data, report_concepts, onto_proc, use_case, debug=False):
		"""
		Create the Radboud rdf report graph out of extracted concepts

		Params:
			rid (str): the report id
			report_data (dict): the target report data
			report_concepts (dict): the concepts extracted from the target report
			onto_proc (OntologyProc): instance of OntologyProc class
			use_case (str): the use_case considered - i.e. colon, lung, cervix, or celiac
			debug (bool): whether to keep flags for debugging

		Returns: a list of (s, p, o) triples representing report data in rdf and a dict containing structured info
		"""

		rdf_graph = []
		struct_graph = {}

		# generate report id hashing 'diagnosis' field
		hrid = hashlib.md5(rid.encode()).hexdigest()
		struct_graph['ReportID'] = hrid

		# create report data-related triples

		# build the IRI for the resource
		resource = self.namespace['exa'] + 'resource/'

		# build the IRI for the given report
		report = resource + 'report/' + hrid
		struct_graph['ReportURL'] = report

		# build the IRI for the use-case ClinicalCaseReport
		use_case_ccreport = self.namespace['exa'] + 'ontology/#' + use_case.capitalize() + 'ClinicalCaseReport'
		struct_graph['ClinicalCase'] = use_case_ccreport

		# generate report data-related triples
		rdf_graph.append((report, 'a', use_case_ccreport))
		rdf_graph.append((report, self.namespace['dc'] + 'identifier', hrid))
		# store conclusion text from radboud report
		diagnosis = report_data['diagnosis']
		if diagnosis:  # textual diagnosis is present within conclusions
			rdf_graph.append((report, self.namespace['exa'] + 'ontology/#hasDiagnosisText', diagnosis))
			struct_graph['hasDiagnosisText'] = diagnosis

		if 'slide_ids' in report_data:
			struct_graph['slides'] = []

			# set ontology 'Slide Device'
			ontology_slide = 'http://purl.obolibrary.org/obo/NCIT_C50178'
			# generate report slide-related triples
			for slide_id in report_data['slide_ids']:
				slide = {}
				rdf_graph.append((report + '/slide/' + slide_id, 'a', ontology_slide))
				slide['a'] = ontology_slide
				rdf_graph.append((report + '/slide/' + slide_id, self.namespace['exa'] + 'ontology/#hasSlideId', slide_id))
				slide['hasSlideId'] = slide_id
				rdf_graph.append((report, self.namespace['exa'] + 'ontology/#hasSlide', report + '/slide/' + slide_id))
				slide['hasSlide'] = report + '/slide/' + slide_id
				struct_graph['slides'].append(slide)

		# create patient-related triples

		struct_graph['patient'] = {}

		# generate patient id hashing the first part of the 'rid' field (up to P00000###)
		pid = hashlib.md5('_'.join(rid.split('_')[0:3]).encode()).hexdigest()
		# build the IRI for the patient
		patient = self.namespace['exa'] + 'resource/patient/' + pid
		struct_graph['patient']['PatientURL'] = patient

		# generate patient-related triples
		rdf_graph.append((patient, 'a', 'http://purl.obolibrary.org/obo/IDOMAL_0000603'))
		struct_graph['patient']['a'] = 'http://purl.obolibrary.org/obo/IDOMAL_0000603'

		# associate report to patient
		rdf_graph.append((patient, self.namespace['exa'] + 'ontology/#hasClinicalCaseReport', report))

		# create report concept-related triples

		# set ontology 'Outcome' IRI to identify its descendants within the 'Diagnosis' section
		ontology_outcome = 'http://purl.obolibrary.org/obo/NCIT_C20200'
		if use_case == 'colon':
			# set ontology 'Polyp' IRI to identify its descendants within the 'Diagnosis' section
			ontology_polyp = 'http://purl.obolibrary.org/obo/MONDO_0021400'
		elif use_case == 'cervix':
			# set ontology 'Human Papilloma Virus Infection' and 'Koilocytotic Squamous Cell' IRIs to identify such outcomes within 'Diagnosis'
			ontology_hpv = 'http://purl.obolibrary.org/obo/MONDO_0005161'
			ontology_koilocyte = 'http://purl.obolibrary.org/obo/NCIT_C36808'

		# identify report procedures
		report_procedures = [procedure[0] for procedure in report_concepts['concepts']['Procedure']]

		# identify report anatomical locations
		report_locations = [location[0] for location in report_concepts['concepts']['Anatomical Location']]
		if not report_locations:  # @smarchesin TODO: decide how to handle cervix when location is absent
			if use_case == 'colon':
				# add 'Colon, NOS' IRI as default
				report_locations += ['http://purl.obolibrary.org/obo/UBERON_0001155']
		if use_case != 'cervix':  # @smarchesin TODO: this might be updated depending on the other use cases
			# identify report tests
			report_tests = [test[0] for test in report_concepts['concepts']['Test']]

		# identify report outcomes
		report_outcomes = [
			(diagnosis[0], onto_proc.get_higher_concept(iri1=diagnosis[0], iri2=ontology_outcome))
			for diagnosis in report_concepts['concepts']['Diagnosis']]
		if use_case == 'colon':
			# identify report polyps
			report_polyps = [
				(diagnosis[0], onto_proc.get_higher_concept(iri1=diagnosis[0], iri2=ontology_polyp, include_self=True))
				for diagnosis in report_concepts['concepts']['Diagnosis']]
			# restrict report_outcomes to those polyp-related and mask concepts that are sublcass of Polyp w/ 1 and 0 otherwise (dysplasia-related)
			for ix, (outcome, polyp) in enumerate(zip(report_outcomes, report_polyps)):
				if (outcome[1] is not None) and (polyp[1] is None):  # non-polyp outcome
					report_polyps.pop(ix)
				elif (outcome[1] is not None) and (polyp[1] is not None):  # polyp outcome
					report_outcomes.pop(ix)
				else:  # dysplasia outcome
					report_outcomes.pop(ix)
			# mask report_polyps w/ 1 for concepts that are subclass of Polyp and 0 for other
			masked_outcomes = ''.join(['0' if report_outcome[1] is None else '1' for report_outcome in report_polyps])
			# associate polyp-related mentions w/ dysplasia ones - restricted to colon disease only
			paired_outcomes = self.associate_polyp2dysplasia(report_polyps, masked_outcomes, debug=debug)
			# concatenate the non-polyp outcomes to paired_outcomes
			paired_outcomes += [[outcome[0]] for outcome in report_outcomes]
		else:  # @smarchesin TODO: the 'else' will probably be replaced with use-case dependants if conditions
			paired_outcomes = [[outcome[0]] for outcome in report_outcomes]

		# set the counter for outcomes identified from report
		report_outcome_n = 0

		struct_graph['hasOutcome'] = []
		# loop over outcomes and build 'Outcome'-related triples
		for pair in paired_outcomes:
			outcomes_struct = {}

			# increase outcomes counter
			report_outcome_n += 1

			# 'Diagnosis'-related triples

			# build the IRI for the identified outcome
			resource_outcome = resource + hrid + '/' + str(report_outcome_n)
			outcomes_struct['OutcomeURL'] = resource_outcome

			# attach outcome instance to graph
			rdf_graph.append((report, self.namespace['exa'] + 'ontology/#hasOutcome', resource_outcome))
			# specify what the resource_outcome is
			if use_case == 'cervix':
				if pair[0] == ontology_hpv:  # outcome is hpv infection
					rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#detectedHumanPapillomaVirus', pair[0]))
					outcomes_struct['detectedHumanPapillomaVirus'] = pair[0]
				if pair[0] == ontology_koilocyte:  # outcome is koilocyte
					rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#koylociteDetected', pair[0]))
					outcomes_struct['koylociteDetected'] = pair[0]
			else:  # regular outcome
				rdf_graph.append((resource_outcome, 'a', pair[0]))
				outcomes_struct['a'] = pair[0]

			if use_case == 'colon' and len(pair) > 1:  # target outcome has associated dysplasia
				outcomes_struct['hasDysplasia'] = []

				for dysplasia_outcome in pair[1:]:
					# specify the associated dysplasia
					rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#hasDysplasia', dysplasia_outcome))
					outcomes_struct['hasDysplasia'].append(dysplasia_outcome)

			# 'Anatomical'-related triples

			outcomes_struct['hasLocation'] = []
			# specify the anatomical location associated to target outcome
			for location in report_locations:  # @smarchesin TODO: correct? we might link multiple locations to the same outcome
				rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#hasLocation', location))
				outcomes_struct['hasLocation'].append(location)

			# 'Procedure'-related triples

			outcomes_struct['hasIntervention'] = []
			# loop over procedures and build 'Procedure'-related triples
			for ix, procedure in enumerate(report_procedures):  # @smarchesin TODO: correct? we might link the same procedure to multiple outcomes
				intervention_struct = {}

				# build the IRI for the identified procedure
				resource_procedure = resource + 'procedure/' + hrid + '/' + str(report_outcome_n) + '.' + str(ix+1)
				# attach procedure instance to graph
				rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#hasIntervention', resource_procedure))
				intervention_struct['InterventionURL'] = resource_procedure

				# specify what the resource_procedure is
				rdf_graph.append((resource_procedure, 'a', procedure))
				intervention_struct['a'] = procedure

				intervention_struct['hasTopography'] = []
				# specify the anatomical location associated to target procedure
				for location in report_locations:  # @smarchesin TODO: correct? we might link multiple locations to the same procedure
					rdf_graph.append((resource_procedure, self.namespace['exa'] + 'ontology/#hasTopography', location))
					intervention_struct['hasTopography'].append(location)

				outcomes_struct['hasIntervention'].append(intervention_struct)

			# 'Test'-related triples - @smarchesin TODO: decide how to handle tests

			outcomes_struct['hasTest'] = []
			if use_case != 'cervix':
				for test in report_tests:  # @smarchesin TODO: correct? we might link multiple tests to the same outcome
					rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#hasTest', test))
					outcomes_struct['hasTest'].append(test)

			struct_graph['hasOutcome'].append(outcomes_struct)

		# return report rdf graph
		return rdf_graph, struct_graph

	# GENERAL-PURPOSE FUNCTIONS

	def create_graph(self, rid, report_data, report_concepts, onto_proc, use_case, debug=False):
		"""
		Create the rdf report graph out of extracted concepts

		Params:
			rid (str): the report id
			report_data (dict): the target report data
			report_concepts (dict): the concepts extracted from the target report
			onto_proc (OntologyProc): instance of OntologyProc class
			use_case (str): the use_case considered - i.e. colon, lung, cervix, or celiac
			debug (bool): whether to keep flags for debugging

		Returns: a list of (s, p, o) triples representing report data in rdf and a dict containing structured info
		"""

		rdf_graph = []
		struct_graph = {}

		# generate report id
		hrid = 'r_' + rid
		struct_graph['ReportID'] = hrid

		# create report data-related triples

		# build the IRI for the resource
		resource = self.namespace['exa'] + 'resource/'

		# build the IRI for the given report
		report = resource + 'report/' + hrid
		struct_graph['ReportURL'] = report

		# build the IRI for the use-case ClinicalCaseReport
		use_case_ccreport = self.namespace['exa'] + 'ontology/#' + use_case.capitalize() + 'ClinicalCaseReport'
		struct_graph['ClinicalCase'] = use_case_ccreport

		# generate report data-related triples
		rdf_graph.append((report, 'a', use_case_ccreport))
		rdf_graph.append((report, self.namespace['dc'] + 'identifier', hrid))
		# store text from report
		diagnosis = report_data['text']
		if diagnosis:  # textual diagnosis is present within conclusions
			rdf_graph.append((report, self.namespace['exa'] + 'ontology/#hasDiagnosisText', diagnosis))
			struct_graph['hasDiagnosisText'] = diagnosis

		# create patient-related triples

		struct_graph['patient'] = {}

		# generate patient id
		pid = 'p_' + rid
		# build the IRI for the patient
		patient = self.namespace['exa'] + 'resource/patient/' + pid
		struct_graph['patient']['PatientURL'] = patient

		# generate patient-related triples
		rdf_graph.append((patient, 'a', 'http://purl.obolibrary.org/obo/IDOMAL_0000603'))
		struct_graph['patient']['a'] = 'http://purl.obolibrary.org/obo/IDOMAL_0000603'

		# associate report to patient
		rdf_graph.append((patient, self.namespace['exa'] + 'ontology/#hasClinicalCaseReport', report))
		# associate age to patient
		age = report_data['age']
		if age:  # age found within current report
			# associate age to report
			rdf_graph.append((patient, self.namespace['exa'] + 'ontology/#hasAge', age))
			struct_graph['patient']['hasAge'] = age

			# convert age to age set and associate to report
			if age < 40:  # young
				rdf_graph.append((patient, self.namespace['exa'] + 'ontology/#hasAgeOnset', self.age_set['young']))
				struct_graph['patient']['hasAgeOnset'] = self.age_set['young']
			elif 40 <= age < 60:  # middle
				rdf_graph.append((patient, self.namespace['exa'] + 'ontology/#hasAgeOnset', self.age_set['middle']))
				struct_graph['patient']['hasAgeOnset'] = self.age_set['middle']
			else:  # late
				rdf_graph.append((patient, self.namespace['exa'] + 'ontology/#hasAgeOnset', self.age_set['late']))
				struct_graph['patient']['hasAgeOnset'] = self.age_set['late']
		# associate gender to patient
		if report_data['gender']:  # gender data is present within report_data
			rdf_graph.append((patient, self.namespace['exa'] + 'ontology/#hasGender', self.gender[report_data['gender']]))
			struct_graph['patient']['hasGender'] = self.gender[report_data['gender']]
			struct_graph['patient']['hasGenderLiteral'] = report_data['gender']

		# create report concept-related triples

		# set ontology 'Outcome' IRI to identify its descendants within the 'Diagnosis' section
		ontology_outcome = 'http://purl.obolibrary.org/obo/NCIT_C20200'
		if use_case == 'colon':
			# set ontology 'Polyp' IRI to identify its descendants within the 'Diagnosis' section
			ontology_polyp = 'http://purl.obolibrary.org/obo/MONDO_0021400'
		elif use_case == 'cervix':
			# set ontology 'Human Papilloma Virus Infection' and 'Koilocytotic Squamous Cell' IRIs to identify such outcomes within 'Diagnosis'
			ontology_hpv = 'http://purl.obolibrary.org/obo/MONDO_0005161'
			ontology_koilocyte = 'http://purl.obolibrary.org/obo/NCIT_C36808'

		# identify report procedures
		report_procedures = [procedure[0] for procedure in report_concepts['Procedure']]

		# identify report anatomical locations
		report_locations = [location[0] for location in report_concepts['Anatomical Location']]
		if not report_locations:  # @smarchesin TODO: decide how to handle cervix when location is absent
			if use_case == 'colon':
				# add 'Colon, NOS' IRI as default
				report_locations += ['http://purl.obolibrary.org/obo/UBERON_0001155']
		if use_case != 'cervix':  # @smarchesin TODO: this might be updated depending on the other use cases
			# identify report tests
			report_tests = [test[0] for test in report_concepts['Test']]

		# identify report outcomes
		report_outcomes = [
			(diagnosis[0], onto_proc.get_higher_concept(iri1=diagnosis[0], iri2=ontology_outcome))
			for diagnosis in report_concepts['Diagnosis']]
		if use_case == 'colon':
			# identify report polyps
			report_polyps = [
				(diagnosis[0], onto_proc.get_higher_concept(iri1=diagnosis[0], iri2=ontology_polyp, include_self=True))
				for diagnosis in report_concepts['Diagnosis']]
			# restrict report_outcomes to those polyp-related and mask concepts that are sublcass of Polyp w/ 1 and 0 otherwise (dysplasia-related)
			for ix, (outcome, polyp) in enumerate(zip(report_outcomes, report_polyps)):
				if (outcome[1] is not None) and (polyp[1] is None):  # non-polyp outcome
					report_polyps.pop(ix)
				elif (outcome[1] is not None) and (polyp[1] is not None):  # polyp outcome
					report_outcomes.pop(ix)
				else:  # dysplasia outcome
					report_outcomes.pop(ix)
			# mask report_polyps w/ 1 for concepts that are subclass of Polyp and 0 for other
			masked_outcomes = ''.join(['0' if report_outcome[1] is None else '1' for report_outcome in report_polyps])
			# associate polyp-related mentions w/ dysplasia ones - restricted to colon disease only
			paired_outcomes = self.associate_polyp2dysplasia(report_polyps, masked_outcomes, debug=debug)
			# concatenate the non-polyp outcomes to paired_outcomes
			paired_outcomes += [[outcome[0]] for outcome in report_outcomes]
		else:  # @smarchesin TODO: the 'else' will probably be replaced with use-case dependants if conditions
			paired_outcomes = [[outcome[0]] for outcome in report_outcomes]

		# set the counter for outcomes identified from report
		report_outcome_n = 0

		struct_graph['hasOutcome'] = []
		# loop over outcomes and build 'Outcome'-related triples
		for pair in paired_outcomes:
			outcomes_struct = {}

			# increase outcomes counter
			report_outcome_n += 1

			# 'Diagnosis'-related triples

			# build the IRI for the identified outcome
			resource_outcome = resource + hrid + '/' + str(report_outcome_n)
			outcomes_struct['OutcomeURL'] = resource_outcome

			# attach outcome instance to graph
			rdf_graph.append((report, self.namespace['exa'] + 'ontology/#hasOutcome', resource_outcome))
			# specify what the resource_outcome is
			if use_case == 'cervix':
				if pair[0] == ontology_hpv:  # outcome is hpv infection
					rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#detectedHumanPapillomaVirus', pair[0]))
					outcomes_struct['detectedHumanPapillomaVirus'] = pair[0]
				if pair[0] == ontology_koilocyte:  # outcome is koilocyte
					rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#koylociteDetected', pair[0]))
					outcomes_struct['koylociteDetected'] = pair[0]
			else:  # regular outcome
				rdf_graph.append((resource_outcome, 'a', pair[0]))
				outcomes_struct['a'] = pair[0]

			if use_case == 'colon' and len(pair) > 1:  # target outcome has associated dysplasia
				outcomes_struct['hasDysplasia'] = []

				for dysplasia_outcome in pair[1:]:
					# specify the associated dysplasia
					rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#hasDysplasia', dysplasia_outcome))
					outcomes_struct['hasDysplasia'].append(dysplasia_outcome)

			# 'Anatomical'-related triples

			outcomes_struct['hasLocation'] = []
			# specify the anatomical location associated to target outcome
			for location in report_locations:  # @smarchesin TODO: correct? we might link multiple locations to the same outcome
				rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#hasLocation', location))
				outcomes_struct['hasLocation'].append(location)

			# 'Procedure'-related triples

			outcomes_struct['hasIntervention'] = []
			# loop over procedures and build 'Procedure'-related triples
			for ix, procedure in enumerate(report_procedures):  # @smarchesin TODO: correct? we might link the same procedure to multiple outcomes
				intervention_struct = {}

				# build the IRI for the identified procedure
				resource_procedure = resource + 'procedure/' + hrid + '/' + str(report_outcome_n) + '.' + str(ix+1)
				# attach procedure instance to graph
				rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#hasIntervention', resource_procedure))
				intervention_struct['InterventionURL'] = resource_procedure

				# specify what the resource_procedure is
				rdf_graph.append((resource_procedure, 'a', procedure))
				intervention_struct['a'] = procedure

				intervention_struct['hasTopography'] = []
				# specify the anatomical location associated to target procedure
				for location in report_locations:  # @smarchesin TODO: correct? we might link multiple locations to the same procedure
					rdf_graph.append((resource_procedure, self.namespace['exa'] + 'ontology/#hasTopography', location))
					intervention_struct['hasTopography'].append(location)

				outcomes_struct['hasIntervention'].append(intervention_struct)

			# 'Test'-related triples - @smarchesin TODO: decide how to handle tests

			outcomes_struct['hasTest'] = []
			if use_case != 'cervix':
				for test in report_tests:  # @smarchesin TODO: correct? we might link multiple tests to the same outcome
					rdf_graph.append((resource_outcome, self.namespace['exa'] + 'ontology/#hasTest', test))
					outcomes_struct['hasTest'].append(test)

			struct_graph['hasOutcome'].append(outcomes_struct)

		# return report rdf graph
		return rdf_graph, struct_graph
