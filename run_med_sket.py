import argparse
import warnings

from sket.sket import SKET

warnings.filterwarnings("ignore", message=r"\[W008\]", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--src_lang', default='it', type=str, help='Considered source language.')
parser.add_argument('--use_case', default='colon', choices=['colon', 'cervix', 'lung'], help='Considered use-case.')
parser.add_argument('--spacy_model', default='en_core_sci_sm', type=str, help='Considered NLP spacy model.')
parser.add_argument('--w2v_model', default=False, action='store_true', help='Considered word2vec model.')
parser.add_argument('--fasttext_model', default=None, type=str, help='File path for FastText model.')
parser.add_argument('--bert_model', default=None, type=str, help='Considered BERT model.')
parser.add_argument('--string_model', default=False, action='store_true', help='Considered string matching model.')
parser.add_argument('--gpu', default=None, type=int, help='Considered GPU device. If not specified (default to None), use CPU instead.')
parser.add_argument('--thr', default=0.9, type=float, help='Similarity threshold.')
parser.add_argument('--store', default=False, action='store_true', help='Whether to store concepts, labels, and graphs.')
parser.add_argument('--raw', default=False, action='store_true', help='Whether to consider full pipeline or not.')
parser.add_argument('--debug', default=False, action='store_true', help='Whether to use flags for debugging.')
parser.add_argument('--dataset', default='', type=str, help='Dataset file path.')
args = parser.parse_args()


def main():
    # set SKET
    sket = SKET(args.use_case, args.src_lang, args.spacy_model, args.w2v_model, args.fasttext_model, args.bert_model, args.string_model, args.gpu)

    if args.dataset:  # use dataset from file path
        dataset = args.dataset
    else:  # use sample "stream" dataset
        dataset = {
                    'text': 'adenocarcinoma con displasia lieve, focalmente severa. Risultati ottenuti con biopsia al colon.',
                    'gender': 'M',
                    'age': 56,
                    'id': 'test'
        }

    # use SKET pipeline to extract concepts, labels, and graphs from dataset
    sket.med_pipeline(dataset, args.src_lang, args.use_case, args.thr, args.store, args.raw, args.debug)

    if args.raw:
        print('processed data up to concepts.')
    else:
        print('full pipeline.')


if __name__ == "__main__":
    main()
