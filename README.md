An AI Chatbot-Multima using Python and Flask

![Chatbot](Chatbot.png)

## An AI Chatbot Multima
Chatbot uses data from www.multima.cz. The primary data is in Czech, but the chatbot communicates in all languages. It only supports simple query/response communication. It recognizes the language of the question and tries to answer in the same language.


![Chatbot](ChatbotFlow.png)


## Requirements (libraries)
1. Flask - framework for web application
2. OpenAI - ChatBot GPT API
3. python-decouple - processing environment variable
4. uuid - work with uuid
5. socket - to get web user information
6. langdetect - language detection from a text
7. deep-translator - language translator
8. transformers - GPT2 tokenizer
9. Qdrant - Qdrant vector database
10. pandas - Dataframe framework
11. gunicorn - LINUX server for web application running


![Structure](structure.png)

## Visual Studio SetUp
1. Create solution PythonApplication
2. Create Project PythonFlask
3. Create a python virtual environment 

Run ```pip install -U Flask``` to install ```flask```

Run ```pip install -U openai``` to install ```openai```

Run ```pip install -U python-decouple``` to install ```decouple```

Run ```pip install -U uuid``` to install ```uuid```

Run ```pip install -U langdetect``` to install ```langdetect```

Run ```pip install -U deep-translator``` to install ```deep-translator```

Run ```pip install -U transformers``` to install ```transformers```

Run ```pip install -U qdrant-client``` to install ```Qdrant```

Run ```pip install -U pandas``` to install ```pandas```

Run ```pip install -U gunicorn``` to install ```gunicorn```

4. Create requirements.txt from virtual environment.

5. Add solution to source Control (GitHub. Repository = PythonApplication)
6. To expose your bot via Ngrok, run ```pip install flask-ngrok``` to install ```flask-ngrok``` Then you'll need to configure your ngrok credentials(login: email + password) Then uncomment this line ```run_with_ngrok(app) ``` and comment the last two lines ```if __name__ == "__main__": app.run() ``` Notice that ngrok is not used by default.
7. To access your bot on localhost, go to ```http://127.0.0.1:5000/ ``` If you're on Ngrok your url will be ```some-text.ngrok.io```


## GitHub SetUp
Repository PythonApplication is created from VisualStudio.

https://github.com/vdedourek2

Create PythonFlask repository and copy from PythonFlask project.
Delete setup.ini file in FlaskRepository. Delete pywin32 from requirements.txt.



## Qdrant setup
https://cloud.qdrant.io/

Create cluster with vector database 1536 dimension in Qdrant cloud. Get URL and API_KEY.



## Render setup
https://dashboard.render.com/

Create Web service from GitHub repository PythonFlask.

SetUp environment variables:

QDRANT_URL=

QDRANT_API_KEY=

OPENAI_API_TYPE=azure

OPENAI_API_BASE=

OPENAI_API_VERSION=

OPENAI_API_KEY=

PYTHON_VERSION=3.11.2

![File](file.png)

Push button Manual Deploy.

Application is accesible on https://chatbot-multima.onrender.com


## Access on web

https://chatbot-multima.onrender.com

![Chatbot Talking](ChatbotTalking.png)

## Regards,
 > [Multima a.s.](https://www.multima.cz/).
