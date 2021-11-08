import json
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sket_server.sket_rest_config import sket_pipe
from django.core.files.storage import FileSystemStorage
import shutil
import os


@api_view(['GET', 'POST'])
def annotate(request, use_case=None, language=None, obj=None, rdf_format=None):
    json_resp_single = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}

    if request.method == 'GET':
        return Response(json_resp_single)

    elif request.method == 'POST':
        workpath = os.path.dirname(os.path.abspath(__file__))
        output_concepts_dir = os.path.join(workpath, '../sket_rest_config/config.json')
        f = open(output_concepts_dir, 'r')
        data = json.load(f)
        thr = data['thr']
        files = []
        labels = {}
        concepts = {}
        rdf_graphs = {}

        if use_case is None or language is None:
            response = {'ERROR': 'Your request is invalid!'}
            return Response(response, status=status.HTTP_400_BAD_REQUEST)
        if obj not in ['concepts','graphs','labels','n3','turtle','all','trig',None]:
            response = {'ERROR': 'Your request is invalid!'}
            return Response(response, status=status.HTTP_400_BAD_REQUEST)
        if obj == 'graphs' and (
                rdf_format is None or rdf_format == 'all' or rdf_format not in ['n3', 'turtle', 'trig']):
            response = {'ERROR': 'Your request is invalid: the allowed rdf_formats are: turtle, n3, trig'}
            return Response(response, status=status.HTTP_400_BAD_REQUEST)

        store = True

        if (use_case is not None and language is not None and obj in ['n3', 'turtle', 'trig', 'all']) or (
                use_case is not None and language is not None):
            store = True
        if obj in ['concepts', 'graphs', 'labels']:
            store = False
            if obj in ['concepts','labels']:
            	rdf_format = 'turtle'
        if store == True and obj is None:
            rdf_format = 'all'
        if store == True and obj in ['n3', 'turtle', 'trig', 'all']:
            rdf_format = obj


        if len(request.FILES) > 0:
            for file in request.FILES.items():
                files.append(file[1])

        if len(files) == 0:
            print('json')
            if type(request.data) == dict:
                request_body = request.data
                concepts, labels, rdf_graphs = sket_pipe.med_pipeline(request_body, language, use_case, thr, store,
                                                                      rdf_format,
                                                                      False, False)
        elif len(files) > 0:
            for file in files:
                workpath = os.path.dirname(os.path.abspath(__file__))
                fs = FileSystemStorage(os.path.join(workpath, './tmp'))
                print(file.name)
                file_up = fs.save(file.name, file)
                uploaded_file_path = os.path.join(workpath, './tmp/' + file_up)
                print(rdf_format)
                try:
                    concepts, labels, rdf_graphs = sket_pipe.med_pipeline(uploaded_file_path, language, use_case, thr,
                                                                          store, rdf_format,
                                                                          False, False)
                except Exception as e:
                    print(e)
                    js_resp = {'error': 'an error occurred: ' + str(e) + '.'}
                    return Response(js_resp)
                finally:
                    for root, dirs, files in os.walk(os.path.join(workpath, './tmp')):
                        for f in files:
                            os.unlink(os.path.join(root, f))
                        for d in dirs:
                            shutil.rmtree(os.path.join(root, d))
        if not store:
            if obj == 'graphs':
                return Response(rdf_graphs, status=status.HTTP_201_CREATED)
            elif obj == 'labels':
                return Response(labels, status=status.HTTP_201_CREATED)
            elif obj == 'concepts':
                return Response(concepts, status=status.HTTP_201_CREATED)
        elif store:
            json_resp = {"response": "request handled with success."}
            return Response(json_resp, status=status.HTTP_201_CREATED)

    else:
        return Response(json_resp_single, status=status.HTTP_201_CREATED)

