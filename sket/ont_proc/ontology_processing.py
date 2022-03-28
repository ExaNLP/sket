import owlready2
import itertools
import pandas as pd

from collections import defaultdict
from copy import deepcopy
from owlready2 import IRIS

from ..utils import utils


class OntoProc(object):

	def __init__(self, ontology_path=None, hierarchies_path=None):
		"""
		Load ontology and set use-case variable 

		Params:
			ontology_path (str): ontology.owl file path
			hierarchies_path (str): hierarchy relations file path

		Returns: None
		"""

		if ontology_path:  # custom ontology path
			self.ontology = owlready2.get_ontology(ontology_path).load()
		else:  # default ontology path
			self.ontology = owlready2.get_ontology('./sket/ont_proc/ontology/examode.owl').load()
		if hierarchies_path:  # custom hierarchy relations path
			self.hrels = utils.read_hierarchies(hierarchies_path)
		else:  # default hierarchy relations path
			self.hrels = utils.read_hierarchies('./sket/ont_proc/rules/hierarchy_relations.txt')
		self.disease = {'colon': 'colon carcinoma', 'lung': 'lung cancer', 'cervix': 'cervical cancer', 'celiac': 'celiac disease'}

	def restrict2use_case(self, use_case, limit=1000):
		"""
		Restrict ontology to the considered use-case and return DataFrame containing concepts from restricted ontology

		Params:
			use_case (str): use case considered (colon, lung, cervix, celiac)
			limit (int): max number of returned elements

		Returns: a pandas DataFrame containng concepts information
		"""

		disease = self.disease[use_case]
		sparql = "PREFIX exa: <https://w3id.org/examode/ontology/> " \
			"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
			"select ?iri ?iri_label ?iri_SNOMED_code ?iri_UMLS_code ?semantic_area ?semantic_area_label where { " \
			"?iri rdfs:label ?iri_label ; exa:AssociatedDisease ?disease . " \
			"filter (langMatches( lang(?iri_label), 'en')). " \
			"?disease rdfs:label '" + disease + "'@en . " \
			"OPTIONAL {?iri exa:hasSNOMEDCode ?iri_SNOMED_code .} " \
			"OPTIONAL {?iri exa:hasUMLS ?iri_UMLS_code .} " \
			"OPTIONAL {?iri exa:hasSemanticArea ?semantic_area . " \
			"?semantic_area rdfs:label ?semantic_area_label . " \
			"filter (langMatches( lang(?semantic_area_label), 'en')).} " \
			"} " \
			"limit " + str(limit)
		# get ontology graph as in rdflib 
		ontology_graph = self.ontology.world.as_rdflib_graph()
		# issue sparql query
		r = ontology_graph.query(query_object=sparql)
		# convert query output to DataFrame
		ontology_dict = defaultdict(list)
		for e in r:
			# store entity as IRI
			if self.ontology[e[0]]:  # entity belongs to the ExaMode ontology
				ontology_dict['iri'].append(self.ontology[e[0]].iri)
			else:  # entity belongs to external ontologies
				ontology_dict['iri'].append(e[0].toPython() if e[0] else None)
			# store additional information associated w/ entity
			ontology_dict['label'].append(e[1].toPython() if e[1] else None)
			ontology_dict['SNOMED'].append(e[2].toPython().replace('*', '') if e[2] else None)
			ontology_dict['UMLS'].append(e[3].toPython() if e[3] else None)
			ontology_dict['semantic_area'].append(e[4].toPython() if e[4] else None)
			ontology_dict['semantic_area_label'].append(e[5].toPython() if e[5] else None)			
		return pd.DataFrame(ontology_dict)	

	@staticmethod
	def lookup_semantic_areas(semantic_areas, use_case_ontology):
		"""
		Lookup for ontology concepts associated to target semantic areas
		
		Params:
			semantic_areas (list(str)/str): target semantic areas
			use_case_ontology (pandas DataFrame): reference ontology restricted to the use case considered
			
		Returns a list of rows matching semantic areas
		"""
		
		if type(semantic_areas) == list:  # search for list of semantic areas
			rows = use_case_ontology.loc[use_case_ontology['semantic_area_label'].isin(semantic_areas)][['iri', 'label', 'semantic_area_label']]
		else:  # search for single semantic area
			rows = use_case_ontology.loc[use_case_ontology['semantic_area_label'] == semantic_areas][['iri', 'label', 'semantic_area_label']]
		if rows.empty:  # no match found within ontology
			return []
		else:  # match found
			return rows.values.tolist()

	def get_ancestors(self, concepts, include_self=False):
		"""
		Returns the list of ancestor concepts given target concept and hierachical relations
		
		Params:
			concepts (list(owlready2.entity.ThingClass/examode.DiseaseAnnotation)): list of concepts from ontology
			include_self (bool): whether to include current concept in the list of ancestors
		
		Returns: the list of ancestors for target concept
		"""
		
		assert type(concepts) == list

		# get latest concept within concepts
		concept = concepts[-1]
		# the get_properties class depends on concept type
		if type(concept) == owlready2.entity.ThingClass:  # class level - rely on ancestors() property (recursively return all parent classes)
			return concept.ancestors(include_self=include_self)
		else:  # instance level - navigate through instance-instance relations
			rels = [rel for rel in concept.get_properties() if rel.name in self.hrels]
			if rels:  # concept hierarchically related w/ ancestor
				assert len(rels) == 1
				# append ancestor to list
				concepts.append(rels[0][concept][0])  
				# recursively search for ancestors 
				return self.get_ancestors(concepts)
			else:  # base case
				return set(concepts[1:])

	def get_higher_concept(self, iri1, iri2, include_self=False):
		"""
		Return the ontology concept that is more general (hierarchically higher)
		
		Params:
			iri1 (str): the first iri considered
			iri2 (str): the second iri considered
			include_self (bool): whether to include current concept in the list of ancestors
			
		Returns: the hierarchically higher concept's iri
		"""
		
		# convert iris into full ontology concepts
		concept1 = IRIS[iri1]
		concept2 = IRIS[iri2]
		# get ancestors for both concepts
		ancestors1 = self.get_ancestors([concept1], include_self)
		ancestors2 = self.get_ancestors([concept2], include_self)
		if concept2 in ancestors1:  # concept1 is a descendant of concept2
			return iri2
		elif concept1 in ancestors2:  # concept1 is an ancestor of concept2
			return iri1
		else:  # concept1 and concept2 are not hierarchically related
			return None

	def merge_nlp_and_struct(self, nlp_concepts, struct_concepts):
		"""
		Merge the information extracted from 'nlp' and 'struct' sections
		
		Params:
			nlp_concepts (dict): the dictionary of linked concepts from 'nlp' section
			struct_concepts (dict): the dictionary of linked concepts from 'struct' section
		
		Returns: a dict containing the linked concepts w/o distinction between 'nlp' and 'struct' concepts
		"""
		
		cconcepts = dict()
		# merge linked concepts from 'nlp' and 'struct' sections
		for sem_area in nlp_concepts.keys():
			if nlp_concepts[sem_area] and struct_concepts[sem_area]:  # semantic area is not empty for both 'nlp' and 'struct' sections
				# get all the possible combinations of 'nlp' and 'struct' concepts
				combinations = list(itertools.product(nlp_concepts[sem_area], struct_concepts[sem_area]))
				# return IRIs to be removed (hierarchically higher)
				IRIs = {self.get_higher_concept(combination[0][0], combination[1][0]) for combination in combinations} - {None}
				# remove under-specified concepts and store remaining concepts
				cconcepts[sem_area] = deepcopy(nlp_concepts[sem_area])
				cconcepts[sem_area].extend([concept for concept in struct_concepts[sem_area] if concept[0] not in [concept[0] for concept in nlp_concepts[sem_area]]])
				# remove IRIs from cconcepts
				cconcepts[sem_area] = [concept for concept in cconcepts[sem_area] if concept[0] not in IRIs]
			elif nlp_concepts[sem_area]:  # semantic area is not empty only for the 'nlp' section
				cconcepts[sem_area] = deepcopy(nlp_concepts[sem_area])
			elif struct_concepts[sem_area]:  # semantic area is not empty only for 'struct' section
				cconcepts[sem_area] = deepcopy(struct_concepts[sem_area])
			else:  # semantic area is empty for both sections
				cconcepts[sem_area] = list()
		# return combined concepts
		return cconcepts
