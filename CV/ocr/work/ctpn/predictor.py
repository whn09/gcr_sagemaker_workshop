# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback

import flask
from flask import request
from ctpn.ctpnmodel import MyService

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    #health = MyService.get_model() is not None  
    health = True
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['GET','POST'])
def transformation():
    if request.method == 'GET':
        fn=request.args.get('fn')
    else:
        fn=request.form.get('fn')
    print(fn)
    result=MyService.predict(fn)
    return flask.Response(response=result, status=200, mimetype='text/plain')
