import argparse
import warnings

from sket.sket import SKET

warnings.filterwarnings("ignore", message=r"\[W008\]", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='./dataset/original/colon/aoec/ExaMode_2ndDS_AOEC_Colon_SubjectsAnonymized.xlsx', type=str, help='Dataset file.')
parser.add_argument('--sheet', default='list2', type=str, help='Considered dataset sheet.')
parser.add_argument('--header', default=0, type=str, help='Header row within dataset.')
parser.add_argument('--ver', default=2, type=str, help='Considered versioning for operations.')
parser.add_argument('--use_case', default='colon', choices=['colon', 'cervix', 'lung'], help='Considered use-case.')
parser.add_argument('--hospital', default='aoec', choices=['aoec', 'radboud'], help='Considered hospital.')
parser.add_argument('--spacy_model', default='en_core_sci_lg', type=str, help='Considered NLP spacy model.')
parser.add_argument('--w2v_model', default=True, type=bool, help='Considered word2vec model.')
parser.add_argument('--fasttext_model', default=None, type=str, help='File path for FastText model.')
parser.add_argument('--bert_model', default=None, type=str, help='Considered BERT model.')
parser.add_argument('--string_model', default=True, type=bool, help='Considered string matching model.')
parser.add_argument('--gpu', default=0, type=int, help='Considered GPU device. If None, use CPU instead.')
parser.add_argument('--thr', default=1.8, type=float, help='Similarity threshold.')
args = parser.parse_args()


def main():
    # set source language based on hospital
    if args.hospital == 'aoec':
        src_lang = 'it'
    elif args.hospital == 'radboud':
        src_lang = 'nl'
    else:  # raise exception
        print('Input hospital does not belong to available ones.\nPlease consider either "aoec" or "radboud" as hospital.')
        raise Exception
    # set SKET
    sket = SKET(args.use_case, src_lang, args.spacy_model, args.w2v_model, args.fasttext_model, args.bert_model, args.string_model, args.gpu)

    # use SKET pipeline to extract concepts, labels, and graphs from args.dataset
    sket.exa_pipeline(args.dataset, args.sheet, args.header, args.ver, args.use_case, args.hospital, args.thr)


if __name__ == "__main__":
    main()
