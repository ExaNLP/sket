import os
import json

from .rep_proc.report_processing import ReportProc
from .ont_proc.ontology_processing import OntoProc
from .nerd.nerd import NERD
from .rdf_proc.rdf_processing import RDFProc

from .utils import utils


class SKET(object):

    def __init__(
            self,
            use_case, src_lang,
            biospacy="en_core_sci_lg", biow2v=True, biofast=None, biobert=None, str_match=False, gpu=None, rules=None, dysplasia_mappings=None, cin_mappings=None,
            ontology_path=None, hierarchies_path=None):
        """
        Load SKET components

        Params:
            SKET:
                use_case (str): considered use case
                src_lang (str): considered language
            NERD:
                biospacy (str): full spaCy pipeline for biomedical data
                biow2v (bool): whether to use biospacy to perform semantic matching or not
                biofast (str): biomedical fasttext model
                biobert (str): biomedical bert model
                str_match (bool): string matching
                gpu (int): use gpu when using BERT
                rules (str): hand-crafted rules file path
                dysplasia_mappings (str): dysplasia mappings file path
                cin_mappings (str): cin mappings file path
            OntoProc:
                ontology_path (str): ontology.owl file path
                hierarchies_path (str): hierarchy relations file path

        Returns: None
        """

        # load Named Entity Recognition and Disambiguation (NERD)
        self.nerd = NERD(biospacy, biow2v, str_match, biofast, biobert, rules, dysplasia_mappings, cin_mappings, gpu)
        # load Ontology Processing (OntoProc)
        self.onto_proc = OntoProc(ontology_path, hierarchies_path)
        # load Report Processing (ReportProc)
        self.rep_proc = ReportProc(src_lang, use_case)
        # load RDF Processing (RDFProc)
        self.rdf_proc = RDFProc()

        # define set of ad hoc labeling operations @smarchesin TODO: add 'custom' to radboud lung too if required
        self.ad_hoc_labeling = {
            'aoec': {
                'colon': {
                    'original': utils.aoec_colon_concepts2labels,
                    'custom': utils.aoec_colon_labels2binary},
                'cervix': {
                    'original': utils.aoec_cervix_concepts2labels,
                    'custom': utils.aoec_cervix_labels2aggregates},
                'lung': {
                    'original': utils.aoec_lung_concepts2labels}
            },
            'radboud': {
                'colon': {
                    'original': utils.radboud_colon_concepts2labels,
                    'custom': utils.radboud_colon_labels2binary},
                'cervix': {
                    'original': utils.radboud_cervix_concepts2labels,
                    'custom': utils.radboud_cervix_labels2aggregates}
            }
        }

        # set use case
        self.use_case = use_case
        # restrict hand-crafted rules and mappings based on use case
        self.nerd.restrict2use_case(use_case)
        # restrict onto concepts to the given use case
        self.onto = self.onto_proc.restrict2use_case(use_case)
        # restrict concept preferred terms (i.e., labels) given the use case
        self.onto_terms = self.nerd.process_ontology_concepts([term.lower() for term in self.onto['label'].tolist()])

    def update_nerd(
            self,
            biospacy="en_core_sci_lg", biofast=None, biobert=None, str_match=False, rules=None, dysplasia_mappings=None, cin_mappings=None, gpu=None):
        """
        Update NERD model w/ input parameters

        Params:
            biospacy (str): full spaCy pipeline for biomedical data
            biofast (str): biomedical fasttext model
            biobert (str): biomedical bert model
            str_match (bool): string matching
            rules (str): hand-crafted rules file path
            dysplasia_mappings (str): dysplasia mappings file path
            cin_mappings (str): cin mappings file path
            gpu (int): use gpu when using BERT

        Returns: None
        """

        # update nerd model
        self.nerd = NERD(biospacy, biofast, biobert, str_match, rules, dysplasia_mappings, cin_mappings, gpu)
        # restrict hand-crafted rules and mappings based on current use case
        self.nerd.restrict2use_case(self.use_case)

    def update_usecase(self, use_case):
        """
        Update use case and dependent functions

        Params:
            use_case (str): considered use case

        Returns: None
        """

        if use_case not in ['colon', 'cervix', 'lung']:  # raise exception
            print('current supported use cases are: "colon", "cervix", and "lung"')
            raise Exception
        # set use case
        self.use_case = use_case
        # update report processing
        self.rep_proc.update_usecase(self.use_case)
        # restrict hand-crafted rules and mappings based on use case
        self.nerd.restrict2use_case(use_case)
        # restrict onto concepts to the given use case
        self.onto = self.onto_proc.restrict2use_case(use_case)
        # restrict concept preferred terms (i.e., labels) given the use case
        self.onto_terms = self.nerd.process_ontology_concepts([term.lower() for term in self.onto['label'].tolist()])

    def update_nmt(self, src_lang):
        """
        Update NMT model changing source language

        Params:
            src_lang (str): considered source language

        Returns: None
        """

        if src_lang not in ['it', 'nl']:  # raise exception
            print('current supported source languages are: "it" and "nl"')
            raise Exception
        # update NMT model
        self.rep_proc.update_nmt(src_lang)

    @staticmethod
    def store_reports(reports, r_path):
        """
        Store reports

        Params:
            reports (dict): reports
            r_path (str): reports file path

        Returns: None
        """

        with open(r_path, 'w') as out:
            json.dump(reports, out, indent=4)

    @staticmethod
    def load_reports(r_fpath):
        """
        Load reports

        Params:
            r_fpath (str): reports file path

        Returns: reports
        """

        with open(r_fpath, 'r') as rfp:
            reports = json.load(rfp)
        return reports

    @staticmethod
    def store_concepts(concepts, c_fpath):
        """
        Store extracted concepts as JSON dict

        Params:
            concepts (dict): dict containing concepts extracted from reports
            c_fpath (str): concepts file path

        Returns: None
        """

        utils.store_concepts(concepts, c_fpath)

    @staticmethod
    def store_labels(labels, l_fpath):
        """
        Store mapped labels as JSON dict

        Params:
            labels (dict): dict containing labels mapped from extracted concepts
            l_fpath (str): labels file path

        Returns: None
        """

        utils.store_labels(labels, l_fpath)

    def store_rdf_graphs(self, graphs, g_fpath):
        """
        Store RDF graphs w/ RDF serialization format

        Params:
            graphs (list): list containing (s,p,o) triples representing ExaMode report(s)
            g_fpath (str): graphs file path

        Returns: None
        """

        rdf_format = g_fpath.split('.')[-1]

        if rdf_format not in ['ttl', 'n3', 'trig']:  # raise exception
            print('provide correct format: "ttl", "n3", or "trig".')
            raise Exception

        rdf_format = 'turtle' if rdf_format == 'ttl' else rdf_format
        self.rdf_proc.serialize_report_graphs(graphs, output=g_fpath, rdf_format=rdf_format)

    @staticmethod
    def store_json_graphs(graphs, g_fpath):
        """
        Store RDF graphs w/ JSON serialization format

        Params:
            graphs (dict): dict containing (s,p,o) triples representing ExaMode report(s)
            g_fpath (str): graphs file path

        Returns: None
        """

        os.makedirs(os.path.dirname(g_fpath), exist_ok=True)

        with open(g_fpath, 'w') as out:
            json.dump(graphs, out, indent=4)

    # EXAMODE RELATED FUNCTIONS

    def prepare_exa_dataset(self, ds_fpath, sheet, header, hospital, ver, ds_name=None):
        """
        Prepare ExaMode batch data to perform NERD

        Params:
            ds_fpath (str): examode dataset file path
            sheet (str): name of the excel sheet to use
            header (int): row index used as header
            hospital (str): considered hospital
            ver (int): data format version
            ds_name (str): dataset name

        Returns: translated, split, and prepared dataset
        """

        # get dataset name from file path if not provided
        if not ds_name:
            ds_name = ds_fpath.split('/')[-1].split('.')[0]  # ./dataset/raw/aoec/####.csv
        # set output directories
        proc_out = './dataset/processed/' + hospital + '/' + self.use_case + '/'
        trans_out = './dataset/translated/' + hospital + '/' + self.use_case + '/'

        if os.path.isfile(trans_out + ds_name + '.json'):  # translated reports file already exists
            print('translated reports file already exist -- remove it before running "exa_pipeline" to reprocess it')
            trans_reports = self.load_reports(trans_out + ds_name + '.json')
            return trans_reports
        elif os.path.isfile(proc_out + ds_name + '.json'):  # processed reports file already exists
            print('processed reports file already exist -- remove it before running "exa_pipeline" to reprocess it')
            proc_reports = self.load_reports(proc_out + ds_name + '.json')
            if hospital == 'aoec':
                # translate reports
                trans_reports = self.rep_proc.aoec_translate_reports(proc_reports)
            elif hospital == 'radboud':
                # translate reports
                trans_reports = self.rep_proc.radboud_translate_reports(proc_reports)
            else:  # raise exception
                print('provide correct hospital info: "aoec" or "radboud"')
                raise Exception

            if not os.path.exists(trans_out):  # dir not exists -- make it
                os.makedirs(trans_out)
            # store translated reports
            self.store_reports(trans_reports, trans_out + ds_name + '.json')

            return trans_reports
        else:  # neither processed nor translated reports files exist
            # load dataset
            dataset = self.rep_proc.load_dataset(ds_fpath, sheet, header)

            if hospital == 'aoec':
                if ver == 1:  # process data using method v1
                    proc_reports = self.rep_proc.aoec_process_data(dataset)
                else:  # process data using method v2
                    proc_reports = self.rep_proc.aoec_process_data_v2(dataset)

                # translate reports
                trans_reports = self.rep_proc.aoec_translate_reports(proc_reports)
            elif hospital == 'radboud':
                if ver == 1:  # process data using method v1
                    proc_reports = self.rep_proc.radboud_process_data(dataset)
                else:  # process data using method v2
                    proc_reports = self.rep_proc.radboud_process_data_v2(dataset)

                # translate reports
                trans_reports = self.rep_proc.radboud_translate_reports(proc_reports)
            else:  # raise exception
                print('provide correct hospital info: "aoec" or "radboud"')
                raise Exception

            if not os.path.exists(proc_out):  # dir not exists -- make it
                os.makedirs(proc_out)
            # store processed reports
            self.store_reports(proc_reports, proc_out + ds_name + '.json')
            if not os.path.exists(trans_out):  # dir not exists -- make it
                os.makedirs(trans_out)
            # store translated reports
            self.store_reports(trans_reports, trans_out + ds_name + '.json')

            return trans_reports

    def exa_entity_linking(self, reports, hospital, sim_thr=0.7):
        """
        Perform entity linking based on ExaMode reports structure and data

        Params:
            reports (dict): dict containing reports -- can be either one or many
            hospital (str): considered hospital
            sim_thr (float): keep candidates with sim score greater than or equal to sim_thr

        Returns: a dict containing concepts from input reports
        """

        # perform entity linking
        if hospital == 'aoec':  # AOEC data
            concepts = self.nerd.aoec_entity_linking(reports, self.onto_proc, self.onto, self.onto_terms, self.use_case, sim_thr)
        elif hospital == 'radboud':  # Radboud data
            concepts = self.nerd.radboud_entity_linking(reports, self.onto, self.onto_terms, self.use_case, sim_thr)
        else:  # raise exception
            print('provide correct hospital info: "aoec" or "radboud"')
            raise Exception
        return concepts

    def exa_labeling(self, concepts, hospital):
        """
        Map extracted concepts to pre-defined labels

        Params:
            concepts (dict): dict containing concepts extracted from report(s)
            hospital (str): considered hospital

        Returns: a dict containing labels from input report(s)
        """

        if hospital not in ['aoec', 'radboud']:
            print('provide correct hospital info: "aoec" or "radboud"')
            raise Exception
        labels = self.ad_hoc_labeling[hospital][self.use_case]['original'](concepts)
        return labels

    def create_exa_graphs(self, reports, concepts, hospital, struct=False):
        """
        Create report graphs in RDF format

        Params:
            reports (dict): dict containing reports -- can be either one or many
            concepts (dict): dict containing concepts extracted from report(s)
            hospital (str): considered hospital
            struct (bool): whether to return graphs structured as dict

        Returns: list of (s,p,o) triples representing report graphs and dict structuring report graphs (if struct==True)
        """

        if hospital == 'aoec':  # AOEC data
            create_graph = self.rdf_proc.aoec_create_graph
        elif hospital == 'radboud':  # Radboud data
            create_graph = self.rdf_proc.radboud_create_graph
        else:  # raise exception
            print('provide correct hospital info: "aoec" or "radboud"')
            raise Exception

        rdf_graphs = []
        struct_graphs = []
        # convert report data into (s,p,o) triples
        for rid in reports.keys():
            rdf_graph, struct_graph = create_graph(rid, reports[rid], concepts[rid], self.onto_proc, self.use_case)
            rdf_graphs.append(rdf_graph)
            struct_graphs.append(struct_graph)
        if struct:  # return both rdf and dict graphs
            return rdf_graphs, struct_graphs
        else:
            return rdf_graphs

    def exa_pipeline(self, ds_fpath, sheet, header, ver, use_case=None, hosp=None, sim_thr=0.7):
        """
        Perform the complete SKET pipeline over ExaMode data:
            - (i) Load dataset
            - (ii) Process dataset
            - (iii) Translate dataset
            - (iv) Perform entity linking and store concepts
            - (v) Perform labeling and store labels
            - (vi) Create RDF graphs and store graphs

        Params:
            ds_fpath (str): dataset file path
            sheet (str): name of the excel sheet to use
            header (int): row index used as header
            ver (int): data format version
            use_case (str): considered use case
            hosp (str): considered hospital
            sim_thr (float): keep candidates with sim score greater than or equal to sim_thr

        Returns: None
        """

        if use_case:  # update to input use case
            self.update_usecase(use_case)

        # get dataset name
        ds_name = ds_fpath.split('/')[-1].split('.')[0]  # ./dataset/raw/aoec/####.csv

        if hosp:  # update to input hospital
            if hosp not in ['aoec', 'radboud']:
                print('provide correct hospital info: "aoec" or "radboud"')
                raise Exception
            else:
                hospital = hosp
        else:
            # get hospital name
            hospital = ds_fpath.split('/')[-2]  # ./dataset/raw/ --> aoec <-- /####.csv

        # set output directories
        concepts_out = './outputs/concepts/' + hospital + '/' + self.use_case + '/'
        labels_out = './outputs/labels/' + hospital + '/' + self.use_case + '/'
        rdf_graphs_out = './outputs/graphs/rdf/' + hospital + '/' + self.use_case + '/'
        struct_graphs_out = './outputs/graphs/json/' + hospital + '/' + self.use_case + '/'

        # prepare dataset
        reports = self.prepare_exa_dataset(ds_fpath, sheet, header, hospital, ver, ds_name)

        # perform entity linking
        concepts = self.exa_entity_linking(reports, hospital, sim_thr)
        # store concepts
        self.store_concepts(concepts, concepts_out + 'concepts_' + ds_name + '.json')
        # perform labeling
        labels = self.exa_labeling(concepts, hospital)
        # store labels
        self.store_labels(labels, labels_out + 'labels_' + ds_name + '.json')
        # create RDF graphs
        rdf_graphs, struct_graphs = self.create_exa_graphs(reports, concepts, hospital, struct=True)
        # store RDF graphs
        self.store_rdf_graphs(rdf_graphs, rdf_graphs_out + 'graphs_' + ds_name + '.n3')
        self.store_rdf_graphs(rdf_graphs, rdf_graphs_out + 'graphs_' + ds_name + '.trig')
        self.store_rdf_graphs(rdf_graphs, rdf_graphs_out + 'graphs_' + ds_name + '.ttl')
        # store JSON graphs
        self.store_json_graphs(struct_graphs, struct_graphs_out + 'graphs_' + ds_name + '.json')