import numpy as np
import json
import argparse

from sklearn.metrics import hamming_loss, accuracy_score, classification_report


parser = argparse.ArgumentParser()
parser.add_argument('--gt', default='./ground_truth/colon/aoec/colon_labels_2ndDS.json', type=str, help='Ground truth file.')
parser.add_argument('--outputs', default='./outputs/labels/aoec/colon/labels_ExaMode_2ndDS_AOEC_Colon_SubjectsAnonymized.json', type=str, help='SKET results file.')
parser.add_argument('--use_case', default='colon', choices=['colon', 'cervix', 'lung'], help='Considered use-case.')
parser.add_argument('--hospital', default='aoec', choices=['aoec', 'radboud'], help='Considered hospital.')
args = parser.parse_args()


label2class = {
    'cervix': {
        'Normal glands': 'glands_norm',
        'Normal squamous': 'squamous_norm',
        'Cancer - squamous cell carcinoma in situ': 'cancer_scc_insitu',
        'Low grade dysplasia': 'lgd',
        'Cancer - squamous cell carcinoma invasive': 'cancer_scc_inv',
        'High grade dysplasia': 'hgd',
        'Koilocytes': 'koilocytes',
        'Cancer - adenocarcinoma invasive': 'cancer_adeno_inv',
        'Cancer - adenocarcinoma in situ': 'cancer_adeno_insitu',
        'HPV infection present': 'hpv'
    },
    'colon': {
        'Hyperplastic polyp': 'hyperplastic',
        'Cancer': 'cancer',
        'Adenomatous polyp - high grade dysplasia': 'hgd',
        'Adenomatous polyp - low grade dysplasia': 'lgd',
        'Non-informative': 'ni'
    },
    'lung': {
        'No cancer': 'no_cancer',
        'Cancer - non-small cell cancer, adenocarcinoma': 'cancer_nscc_adeno',
        'Cancer - non-small cell cancer, squamous cell carcinoma': 'cancer_nscc_squamous',
        'Cancer - small cell cancer': 'cancer_scc',
        'Cancer - non-small cell cancer, large cell carcinoma': 'cancer_nscc_large'
    }
}


def main():
    # read ground-truth
    with open(args.gt, 'r') as gtf:
        ground_truth = json.load(gtf)

    gt = {}
    # prepare ground-truth for evaluation
    if args.use_case == 'colon' and args.hospital == 'aoec':
        gt = ground_truth
    else:
        ground_truth = ground_truth['ground_truth']
        for data in ground_truth:
            rid = data['report_id_not_hashed']
            gt[rid] = {label2class[args.use_case][label]: 0 for label in label2class[args.use_case].keys()}
            for datum in data['labels']:
                label = label2class[args.use_case][datum['label']]
                if label in gt[rid]:
                    gt[rid][label] = 1

    # read SKET results
    with open(args.outputs, 'r') as rsf:
        rs = json.load(rsf)

    sket = {}
    # prepare SKET results for evaluation
    for rid, rdata in rs.items():
        if args.use_case == 'colon' and args.hospital == 'aoec':
            rid = rid.split('_')[0]
        if args.hospital == 'radboud' and args.use_case == 'colon':
            sket[rid] = rdata['labels']
        else:
            sket[rid] = rdata

    # fix class order to avoid inconsistencies
    rids = list(sket.keys())
    classes = list(sket[rids[0]].keys())

    # obtain ground-truth and SKET scores
    gt_scores = []
    sket_scores = []

    for rid in gt.keys():
        gt_rscores = []
        sket_rscores = []
        for c in classes:
            gt_rscores.append(gt[rid][c])
            sket_rscores.append(sket[rid][c])
        gt_scores.append(gt_rscores)
        sket_scores.append(sket_rscores)
    # convert to numpy
    gt_scores = np.array(gt_scores)
    sket_scores = np.array(sket_scores)

    # compute evaluation measures
    print('Compute evaluation measures')

    # exact match accuracy & hamming loss
    print("Accuracy (exact match): {}".format(accuracy_score(gt_scores, sket_scores)))
    print("Hamming loss: {}\n".format(hamming_loss(gt_scores, sket_scores)))

    # compute classification report
    print("Classification report:")
    print(classification_report(y_true=gt_scores, y_pred=sket_scores, target_names=classes))


if __name__ == "__main__":
    main()
