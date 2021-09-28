import time
import json
import os
workpath = os.path.dirname(os.path.abspath(__file__))  # Returns the Path your .py file is in
pp = os.path.join(workpath,'../../')
import sys
sys.path.insert(0,pp)
from sket.sket import SKET
print('start sket initialization')

output_concepts_dir = os.path.join(workpath, './config.json')
f = open(output_concepts_dir,'r')
data = json.load(f)
st = time.time()
# sket_pipe = SKET('colon', 'en', 'en_core_sci_sm', True, None, None, False, 0)
sket_pipe = SKET('colon', 'en', 'en_core_sci_sm', data['w2v_model'], data['fasttext_model'], data['bert_model'], data['string_model'],data['gpu'])
end = time.time()
print('sket initialization completed in: ',str(end-st), ' seconds')
