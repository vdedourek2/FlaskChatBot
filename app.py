# run application in browser  http://localhost:5000
# https://github.com/mainadennis/An-AI-Chatbot-in-Python-and-Flask          #origin of the application
from dotenv import load_dotenv      # python-dotenv
import os
import time
from datetime import datetime
import socket
from flask import Flask, render_template, request
#from flask_ngrok import run_with_ngrok
from Processing.qna_sk_un_mod import QnA, DebugFlag

#project     = "www.multima.cz"      # project name
project     = "www.portalvs.sk"
max_tokens  = 500  # maximum tokens in chunk of text (not modify)
debug_flag = [DebugFlag.Question, DebugFlag.Answer, DebugFlag.Time, DebugFlag.Context, DebugFlag.Params, DebugFlag.Headings]

q = QnA(project = project, original_language = "sk", maxs = max_tokens, is_qa = True, debug_flag = debug_flag,
        maxa=500, maxc=3500)

app = Flask(__name__)

load_dotenv()

app.secret_key = os.getenv("FLASK_SECRET_KEY")
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
    print(f"\nId: {id} pc: {hostname} time: {req_time}\nDotaz: {msg}")

    res = getResponse(msg, id)
    print(f"Odpověď: {res}")
    return res

def getResponse(question, id):
    st = time.time()
    result = q.answer_question(question = question, user_id = id)
    et = time.time()
    return result + f" ({round(et - st, 1)} s)"

if __name__ == "__main__":
    app.run()

