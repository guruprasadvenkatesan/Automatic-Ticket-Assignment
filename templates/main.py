# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:07:51 2020
@author: win10
"""
#pip install fastapi uvicorn

# Library Imports
import uvicorn
from fastapi import FastAPI

#Create App Object
app = FastAPI()

# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

#  Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/Welcome')
def get_name(name: str):
    return {'Welcome To my home page': f'{name}'}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn main:app --reload
