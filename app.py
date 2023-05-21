# run application in browser  http://localhost:5000
# https://github.com/mainadennis/An-AI-Chatbot-in-Python-and-Flask          #origin of the application
from decouple import config     # pip install python-decouple
import openai
import uuid
#import time
from datetime import datetime
import socket
from flask import Flask, render_template, request, session
#from flask_ngrok import run_with_ngrok
from Processing.qna_mod import QnA


# chat initialization
openai.api_type = config('OPENAI_API_TYPE')
openai.api_base = config('OPENAI_API_BASE')
openai.api_version = config('OPENAI_API_VERSION')
openai.api_key = config('OPENAI_API_KEY')

project     = "www.multima.cz"      # project name
max_tokens  = 500  # maximum tokens in chunk of text (not modify)
q = QnA(project, max_tokens)

app = Flask(__name__)
app.secret_key = 'X12X87B'
#run_with_ngrok(app) 

@app.route("/")
def home():

    hostname = socket.gethostname()

    session['session_id'] = {
        "id" :      str(uuid.uuid4()),              # session id
        "chat" :    [],                             # chat information
        "time" :    datetime.utcnow(),              # time of last activity
        "ip" :      socket.gethostbyname(hostname), # IP adress
        "hostname": hostname                        # name of remote computer
        }

    hostname = session['session_id']["hostname"]
    ip_address = session['session_id']["ip"]
    start_time = session['session_id']["time"]
    print(f"Started Hostname: {hostname} IP Address: {ip_address} time: {start_time}")
 
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    id = session['session_id']["id"]
    hostname = session['session_id']["hostname"]
    start_time = session['session_id']["time"]
    print(f"get id: {id} pc: {hostname} start time: {start_time}")

    res = getResponse(msg)
    return res



def getResponse(question):
    result = q.answer_question(question)
    return result


if __name__ == "__main__":
    app.run()

