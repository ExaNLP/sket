import json
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sket_server.sket_rest_config import sket_pipe
from django.core.files.storage import FileSystemStorage
import shutil
import os

@api_view(['GET','POST'])
def annotate(request,use_case = None,language = None,obj = None):

    json_resp_single = {'key1':'value1','key2':'value2', 'key3':'value3'}

    if request.method == 'GET':
        return Response(json_resp_single)

    elif request.method == 'POST':

        files = []
        labels = {}
        concepts = {}
        rdf_graphs = {}
        
        store = True
        if obj in ['concepts','graph','labels']:
            store = False

        if len(request.FILES) > 0:
            for file in request.FILES.items():
                files.append(file[1])

        if len(files) == 0:
            print('json')
            if type(request.data) == dict:
                request_body = request.data
                concepts, labels, rdf_graphs = sket_pipe.med_pipeline(request_body, language, use_case, 0.9, store, False, False)
        elif len(files) > 0:
            for file in files:
                workpath = os.path.dirname(os.path.abspath(__file__))
                fs = FileSystemStorage(os.path.join(workpath,'./tmp'))
                print(file.name)
                file_up = fs.save(file.name, file)
                uploaded_file_path = os.path.join(workpath,'./tmp/'+ file_up)
                print(uploaded_file_path)
                try:
                    concepts, labels, rdf_graphs = sket_pipe.med_pipeline(uploaded_file_path, language, use_case, 0.9, store, False, False)
                except Exception as e:
                    print(e)
                    js_resp = {'error':'an error occurred: ' + str(e) + '.'}
                    return Response(js_resp)
                finally:
                    for root, dirs, files in os.walk(os.path.join(workpath,'./tmp')):
                        for f in files:
                            os.unlink(os.path.join(root, f))
                        for d in dirs:
                            shutil.rmtree(os.path.join(root, d))
        if not store:
            if obj == 'graph':
                return Response(rdf_graphs, status=status.HTTP_201_CREATED)
            elif obj == 'labels':
                return Response(labels, status=status.HTTP_201_CREATED)
            elif obj == 'concepts':
                return Response(concepts, status=status.HTTP_201_CREATED)
        elif store:
            json_resp = {"response":"request handled with success."}
            return Response(json_resp, status=status.HTTP_201_CREATED)

    else:
        return Response(json_resp_single, status=status.HTTP_201_CREATED)
