from django.urls import path
from . import views
from django.views.decorators.csrf import csrf_exempt

from django.contrib.auth import views as auth_views

app_name='sket_server.sket_rest_app'
urlpatterns = [

    # path('annotate/<use_case>/<language>/<object>', views.annotate, name='annotate'),
    path('', views.annotate, name='annotate'),
    # path('annotate/<reports>', views.annotate, name='annotate'),
    path('annotate/<use_case>/<language>/<obj>/<rdf_format>', views.annotate, name='annotate'),    
    path('annotate/<use_case>/<language>/<obj>', views.annotate, name='annotate'), 
    path('annotate/<use_case>/<language>', views.annotate, name='annotate'),

]
