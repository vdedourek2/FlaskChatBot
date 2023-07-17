# run application in browser  http://localhost:5000
# https://github.com/mainadennis/An-AI-Chatbot-in-Python-and-Flask          #origin of the application
from dotenv import load_dotenv      # python-dotenv
import os
from datetime import datetime
import socket
from flask import Flask, render_template, request
#from flask_ngrok import run_with_ngrok
from Processing.qna_mod import QnA

#project     = "www.multima.cz"      # project name
project     = "www.portalvs.sk"
max_tokens  = 500  # maximum tokens in chunk of text (not modify)
q = QnA(project = project, maxs = max_tokens, is_qa = False)

app = Flask(__name__)

load_dotenv()

app.secret_key = os.environ["FLASK_SECRET_KEY"]
#run_with_ngrok(app) 

@app.route("/")
def home():

    hostname = socket.gethostname()
 
    print(f"Started Hostname: {hostname} IP Address: {socket.gethostbyname(hostname)} time: {datetime.utcnow()}")
 
    q.user_history_cleaning()   # cleaning user history

    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    id  = request.form["session_id"]
    hostname = socket.gethostname()
    req_time = datetime.utcnow()
    print(f"get id: {id} pc: {hostname} time: {req_time}")

    res = getResponse(msg, id)
    return res

def getResponse(question, id):
    result = q.answer_question(question = question, user_id = id)
    return result

if __name__ == "__main__":
    app.run()

