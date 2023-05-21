# Building a question answer system with your embeddings - using ChatCompletion
# https://platform.openai.com/docs/tutorials/web-qa-embeddings

from decouple import config     # pip install python-decouple
from langdetect import detect
from deep_translator import GoogleTranslator    # pip install -U deep-translator     https://deep-translator.readthedocs.io/en/latest/README.html
import openai
import time
import math
import uuid
from datetime import datetime
from transformers import GPT2TokenizerFast
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pandas as pd

class QnA(object):
    '''
    ### Class for question answer system
    ------------------------------------
    #   project - project name
    #   max_tokens - maximum tokens in chunk
    #   log_db - loging query/answer to Qdrant database(True = yes, False = No)
    '''
    def __init__(self, project:str, max_tokens:int, log_db:bool = False):
        print("*** Question and Answer started ***")
        st = time.time()
        
        self.collection = project       # name of collection in Qdrant database
        self.max_tokens = max_tokens    # maximum tokens in chunk
        self.log_db = log_db            # loging query/answer to Qdrant database(True = yes, False = No)

        self.question_embeddings = []   # question embeddings are set in create_context_from_db() method

        #setup tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # open Qdrant database
        self.qdrant_client = QdrantClient(
            url=config('QDRANT_URL'), 
            api_key=config('QDRANT_API_KEY'),
        ) 


        et = time.time()
        print(f"Elapsed starting time {round(et - st, 3)} seconds") 

   
    def get_prompt_parts(self, 
                         question:str):
        '''
        Get prompt parts in language of the question
        ------------------------------------------
        question - Question
        return tuple of the question prompt (part1, part2, part3, language)
        '''

        # detection of the question language
        # there are assumption that is cs. On other side is made language detection
        language = "cs"
        cs_question = GoogleTranslator(source='auto', target=language).translate(question) 
        if cs_question != question:
            language = detect(question)

 
        # system message
        text_part = ["Jsi AI asistent. Odpovídáš na základě kontextu uživatele. Když nevíš nebo nejsi jistý, odpověz \"Nevím\".  Odpověz stručně a výstižně.",
                     "Kontext:",
                     "Otázka:"
            ]
 
        #text_part_1 = "Odpověz na otázku na základě kontextu níže. Pokud si nejsi jistý odpovědí, odpověz \"Nevím\". Odpověz stručně a výstižně."
        #text_part_1 += "\n\nKontext:"
        #text_part_2 = ""
        #text_part_3 = ""
    
        match language:
            case "cs":
                no = 1  # no operation

            case "en":
                text_part = ["Answer the question based on the user's context. If you don't know then respond \"I don't know\". Keep the answer short and concise.",
                             "Context:",
                             "Question:"
                    ]

                #text_part_1 = "Answer the question based on the context below. If you don't know then respond \"I don't know\". Keep the answer short and concise."
                #text_part_1 += "\n\nContext:"
                #text_part_2 = ""
                #text_part_3 = ""

            case _:
                # in different language is needed translation to language of question
                text_part = GoogleTranslator(source='cs', target=language).translate_batch(text_part)  
  
        return (text_part[0], text_part[1], text_part[2], language)

    def agregate_context(self,
        db_list,            # record list form database
        max_len:int=1800    # maximum length of context
                     ):
        '''
        Agregate list of chunks for the same title. Insert title to text when it's missing.
        -----------------------------------------------------------------------------------
        db_list - record list form database
        max_len - maximum length of context
        '''

        # 1 - creating list of selected parts of context - chunk_list
        df_chunk = pd.DataFrame(columns = ['dbid', 'title', 'text', 'row'])
        cur_len = 0     # actual length all parts of context
        df_index = 0
        for record in db_list:
            # Add the length of the text to the current length
            cur_len += record.payload["n_tokens"] + 1
        
            # If the context is too long, break
            if cur_len > max_len:
                break

            # Else add it to the text that is being returned
            df_chunk.loc[df_index] = [record.id, record.payload["title"], record.payload["text"], 0]
            df_index += 1

        # 2 - creating list of titles - title_dir
        title_dir = {}     # directory of titles (title : row)
        row = 0

        for index, record in df_chunk.iterrows():
           if title_dir.get(record.title) == None:
                title_dir[record.title] = row
                row += 1

        # 3 setup row item
        df_chunk['row'] = df_chunk.title.apply(lambda x: title_dir[x])

        # 4 - sort df_chunk (row, dbid)
        df_chunk.sort_values(["row", "dbid"], ascending=[True, True])

        # 5 - creating agregate chunks for the same title - agregate_list
        agregate_list = []
        title_last = ""
        text_agr = ""
        for index, row in df_chunk.iterrows():
            if title_last == row['title']:
                text_agr += " " + row['text']
            else:
                if title_last != "":
                    agregate_list.append(text_agr)
                title_last = row['title']
                text_agr = row['text']
                if not text_agr.startswith(title_last):
                    text_agr = title_last + "\n" + text_agr

        agregate_list.append(text_agr)
        return agregate_list

    def create_context_from_db(self,
        question:str,       # question
        language:str,       # question language
        max_len:int=1800,   # maximum length of context
        debug:bool=False    # output debug information about context of question
                      ):
        """
        Create a context for a question by finding the most similar context from the vector database Qadrant
        ----------------------------------------------------------------------------------------------------
        question - question
        language - questioin language
        max_len - maximum length of context
        debug - output debug information about context of question
        return context
        """

        # Get the embeddings for the question
        if openai.api_type == "azure":
            engine = "ada"
        else:
            engine = "text-embedding-ada-002"

        # create question for embedding in original language
        emb_question = question
        if language != "cs":
            emb_question = GoogleTranslator(source=language, target="cs").translate(question)  

        # get embeddings vector
        self.question_embeddings = openai.Embedding.create(input=emb_question, engine=engine)['data'][0]['embedding']

        # setup limit for search operation in Qdrant
        search_limit = math.ceil(max_len / self.max_tokens) + 1

        # search vectors in Qdrant with the best distance
        try:
            result = self.qdrant_client.search(
                collection_name = self.collection,
                search_params=models.SearchParams(
                    hnsw_ef=128,
                    exact=True
                ),
                query_vector = self.question_embeddings,
                limit = search_limit,
            )
        except Exception as e:
            print(f"Qdrant search exception: {e}") 

        returns = self.agregate_context(result, max_len)
        context = "\n\n###\n\n".join(returns)

        # If debug, print the raw model response
        if debug:
            print("Context:\n" + context)
            print("\n\n")



        # Return the context
        return context


    def answer_question(self,
        question:str    ="Kolik má Multima zaměstnanců?",
        max_len:int     =1800,      # maximum tokens of prompt information
        max_tokens:int  =150,       # maximum tokens of answer
        debug:bool      =False,     # output debug information about context of question
        price:bool      =False      # output debug information about price of question
     ):
        """
        Answer a question based on the most similar context from the Qdrant texts
        Maximum tokens for text-davinci-003 is 4097
        there is done loging to Qdrant DB collection project-LOG
        Structure:
        id - UUID
        vector - embedding vector
        date - date and time in form yyyy-mm-dd HH:MM:SS
        q - question
        a - answer
        -------------------------------------------------------------------------
        question - Question
        max_len - Maximum tokens of prompt information
        max_tokens - Maximum tokens of the answer
        debug - Output debug information about context of question
        price - Output debug information about price of question
        returns answer
        """

        st = time.time()

        if question.strip() == "":
            return "Zadejte prosím otázku."

        parts = self.get_prompt_parts(question)
  

        #test_user_content = parts[1] + "\nX\n\n" + parts[2] + "\n" + question + " "+ parts[0]
        test_user_content = parts[0] + "\nX\n\n" + question

        # calculating # tokens of context result
        # Load the gpt2 tokenizer which is designed to work with the ada-002 model
        token_list = self.tokenizer.encode(test_user_content)
        context_len = max_len - len(token_list) + 1

        context = self.create_context_from_db(
            question = question,
            language = parts[3],     #question language
            max_len=context_len,
            debug = debug,
        )

        # creating user content
        system_content = parts[0]
        user_content = parts[1] + "\n" + context + "\n\n" + parts[2] + "\n" + question
        #system_content = parts[0] + "\n" + context
        #user_content = question

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
 
        try:
 
            if openai.api_type == "azure":
                response = openai.ChatCompletion.create(
                    engine = "chat",
                    messages=messages,
                    temperature=0,
                    max_tokens=max_tokens,
                    top_p = 1,
                    n=1,
                    stop=None,
                    presence_penalty=0,
                    frequency_penalty=0,
                )
            else:
                response = openai.ChatCompletion.create(
                    model= "gpt-3.5-turbo",
                    messages=messages,
                    temperature=0,
                    max_tokens=max_tokens,
                    top_p = 1,
                    n=1,
                    stop=None,
                    presence_penalty=0,
                    frequency_penalty=0,
                )

            answer = response['choices'][0]['message']['content']


            # price calculation of the question
            if price:
                question_tokens = len(self.tokenizer.encode(question))

                model_price = 0.002
                price = (question_tokens * 0.00004 + response.usage.total_tokens * model_price) / 1000

                message = f"Tokens: Prompt={response.usage.prompt_tokens} Completion={response.usage.completion_tokens} Total={response.usage.total_tokens} "
                message += f"Price: ({question_tokens} x 0.00004 + {response.usage.total_tokens} x {model_price})/1000 = {price:8.6} $ ... {price * 21.5:8.6} Kč"    
                print(message)


            # loging question/answer
            if self.log_db:

                log_point = models.PointStruct(
                    id = str(uuid.uuid1()),
                    payload={
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "q": question,
                        "a": answer
                    },
                    vector=self.question_embeddings,
                )

                self.qdrant_client.upsert(
                    collection_name=self.collection + "-LOG",
                    points=[log_point]
                )

            et = time.time()

            if price:
                print(f"Elapsed time {round(et - st, 3)} seconds") 


            return answer
        except Exception as e:
            print(e)
            return ""


    def answer_question_print(self,
        question:str    ="Kolik má Multima zaměstnanců?",
        max_len:int     =1800,      # maximum tokens of prompt information
        max_tokens:int  =245,       # maximum tokens of answer
        debug:bool      =False,     # output debug information about context of question
        price:bool      =False      # output debug information about price of question
         ):
        """
        Print Answer/question to system output
        -------------------------------------------------------------------------
        question - Question
        max_len - Maximum tokens of prompt information
        max_tokens - Maximum tokens of the answer
        debug - Output debug information about context of question
        price - Output debug information about price of question
        """
        print("\n" + "Q: " + question)
        print("A:" + self.answer_question(question=question, max_len = max_len, max_tokens = max_tokens, debug=debug, price=price) + "\n")     


          