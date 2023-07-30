# encoding: utf-8
# Building a question answer system with your embeddings - using ChatCompletion
# https://platform.openai.com/docs/tutorials/web-qa-embeddings

import os
from langdetect import detect
from deep_translator import GoogleTranslator    # pip install -U deep-translator     https://deep-translator.readthedocs.io/en/latest/README.html
import openai
import time
import math
from datetime import datetime
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pandas as pd
import uuid
from enum import Enum
from langchain import PromptTemplate
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv      # python-dotenv
import colorama
from termcolor import colored

# result of conversation
class ResultQA(Enum):
    FirstQuestion = 1   # first question
    NextQuestion = 2    # next question in conversation
    Overflow = 3        # # of tokens of context < minc
    Timeout = 4         # time between last answer and question > timeout
    EndConversation = 5 # question doesn't have join to last conversation

# conversation chain
class ConverChain(Enum):
    Chain = 1           # the question depends on last communication
    NoChain = 2         # the question doesn't depend on last communication 
    Overflow = 3        # # of tokens of context < minc
    Timeout = 4         # time between last answer and question > timeout
    Error = 5           # error 

class QnA(object):
    '''
    ### Class for question answer system
    ------------------------------------
    #   project - project name
    #   original_language - original language qdrant text data
    #   maxs - maximum tokens in segment (chunk)
    #   minc - minimum tokens in context
    #   maxc - maximum tokens in context
    #   maxa - maximum tokens of answer
    #   timeout - timeout for conversation in seconds (chain communication)
    #   is_qa - True - query/answer, False - chain communication
    #   log_db - loging query/answer to Qdrant database(True = yes, False = No)
    '''
    def __init__(self,
        project:str,
        original_language:str = "cs",
        maxs:int = 500,
        minc:int = 1000,
        maxc:int = 2000,
        maxa:int = 245,
        timeout:int = 10,
        is_qa:bool = True,
        log_db:bool = False
        ):
        print(colored("*** Question and Answer started ***", "cyan"))
        st = time.time()
        
        self.collection = project       # name of collection in Qdrant database
        self.original_language = original_language
        self.maxs = maxs                # maximum tokens in chunk
        self.minc = minc                # minimum tokens in context
        self.maxc = maxc                # maximum tokens in context
        self.maxa = maxa                # maximum tokens of answer
        self.timeout = timeout          # timeout for conversation (chain communication)
        self.is_qa = is_qa              # is_qa - True - query/answer, False - chain communication
        self.log_db = log_db            # loging query/answer to Qdrant database(True = yes, False = No)

        #self.model = "gpt-3.5-turbo-0613"    # OpenAI model (gpt-3.5-turbo)
        self.maxmod = 4096              # maximum tokens of the model in ChatCompletion
        self.model_price_inp = 0.0015   # price of 1000 input tokens
        self.model_price_out = 0.002    # price of 1000 output tokens

        self.user_history = {}          # communication history with users. Structure   "user_id":last_result ...
        self.question_embeddings = []   # question embeddings are set in create_context_from_db() method

        # setup tokenizer for gpt-4, gpt-3.5-turbo, text-embedding-ada-002
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        self.encoding = tiktoken.get_encoding('cl100k_base')
   
        load_dotenv()
        colorama.init()

        openai.api_type = os.getenv("OPENAI_API_TYPE")
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_version = os.getenv("OPENAI_API_VERSION")
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # creating chat object
        if os.getenv("OPENAI_API_TYPE") == "azure":
            self.chat = AzureChatOpenAI(
                                deployment_name="chat",
                                model_name="gpt-3.5-turbo",
                                temperature=0)
        else:
            self.chat = ChatOpenAI(
                                model_name="gpt-3.5-turbo",
                                temperature=0,
                                openai_api_key=os.getenv("OPENAI_API_KEY"))


        template_depends = """
Co je pták? Odpověď: Kompletní
Co je? Odpověď: Nekompletní
Kolik má obyvatel? Odpověď: Nekompletní
Proč? Odpověď: Nekompletní
Kde to je? Odpověď: Nekompletní
Kolik to bylo? Odpověď: Nekompletní
Jaké je IČO <název_firmy>? Odpověď: Kompletní
{question} Odpověď:"""

        self.prompt_depends = PromptTemplate.from_template(template_depends)

        # open Qdrant database
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"), 
            api_key=os.getenv("QDRANT_API_KEY"),
        ) 


        et = time.time()
        print(f"Elapsed starting time {round(et - st, 3)} seconds") 

   
    def get_prompt_parts(self, 
                         question:str):
        '''
        Get prompt parts in language of the question
        ------------------------------------------
        question - Question
        return tuple of the question prompt (part1, part2, part3, language, nf)
        '''

        # detection of the question language
        # there are assumption that is cs. On other side is made language detection
        language = self.original_language
        cs_question = GoogleTranslator(source='auto', target=language).translate(question) 
        if cs_question != question:
            language = detect(question)

 
        # system message
        text_part = ["Jsi AI asistent. Odpovídáš na základě kontextu uživatele. Když nevíš nebo nejsi jistý, odpověz \"Nevím\". Odpověz stručně a výstižně.",
                     "Kontext:"
            ]
 
        match language:
            case "cs":
                nf = 72

            case "en":
                text_part = ["You are AI Assistant. Answer the question based on the user's context. If you don't know then respond \"I don't know\". Keep the answer short and concise.",
                             "Context:"
                    ]
                nf = 37

            case "sk":
                text_part = ["Si AI asistent. Odpovedáš na základe kontextu užívateľa. Keď nevieš alebo nie si istý, odpovedz \"Neviem\". Odpovedz stručne a výstižne.",
                             "Kontext:"
                    ]
                nf = 65

            case _:
                # in different language is needed translation to language of question
                text_part = GoogleTranslator(source='cs', target=language).translate_batch(text_part)
                nf = len(self.encoding.encode(text_part[0] + " " + text_part[1]))
          
        prompt_frame = text_part[0] + "\n\n" + text_part[1] + "\n"

        return (prompt_frame, language, nf)

    def agregate_context(self,
        db_list,            # record list from database
        max_len:int=1800    # maximum length of context
                     ):
        '''
        Agregate list of chunks for the same title. Insert title to text when it's missing.
        -----------------------------------------------------------------------------------
        db_list - record list from database
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
        df_chunk.sort_values(["row", "dbid"], ascending=[True, True], inplace=True)

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
        if os.getenv("OPENAI_API_TYPE") == "azure":
            engine = "ada"
        else:
            engine = "text-embedding-ada-002"

        # create question for embedding in original language
        emb_question = question
        if language != self.original_language:
            emb_question = GoogleTranslator(source=language, target=self.original_language).translate(question)  

        # get embeddings vector
        self.question_embeddings = openai.Embedding.create(input=emb_question, engine=engine)['data'][0]['embedding']

        # setup limit for search operation in Qdrant
        search_limit = math.ceil(max_len / self.maxs) + 1

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
            print(colored(f"Qdrant search exception: {e}", "red")) 


        returns = self.agregate_context(result, max_len)
        if len(returns) == 0:
            failure = "Pre zadanou otázku nemám k dispozici žiadnou informaci. Zkuste změnit formulaci."
            if language!= "cs":
                failure = GoogleTranslator(source="cs", target=language).translate(failure)
            return (failure, "")


        context = "\n\n###\n\n".join(returns)

        # If debug, print the raw model response
        if debug:
            print("Context:\n" + context)
            print("\n\n")

        # Return the context
        return ("", context)

    def question_depends(self,
        question:str,
        last_result:dir ={},            # last result of answer_question
        nc:int = 0                      # length of context
    ):
        """
        It figure out the questuiion depends on last communication
        -------------------------------------------------------------------------
        return - ConverChain(Enum) -     Chain, NoChain, Overflow, Timeout, Error
        """
        st = time.time()

        # check timeout
        if len(last_result) == 0:
            return ConverChain.NoChain

        if st - last_result["ta"] > self.timeout:
            return ConverChain.Timeout

        if nc < self.minc:
            return ConverChain.Overflow

        prompt_user = self.prompt_depends.format(question = question)

        messages = [
            SystemMessage(content="Jsi AI asistent. Určuješ, zda otázka je kompletní pro vytvoření odpovědi."),
            HumanMessage(content=prompt_user),
        ]

        answer = self.chat(messages).content

        if answer.lower()[0:9] == "kompletní":
            return ConverChain.NoChain
        else:
            return ConverChain.Chain
  

    def answer_question(self,
        question:str        ="Kolik má Multima zaměstnanců?",
        user_id:str         ="",        # user id
        debug:bool          =False,     # output debug information about context of question
        price:bool          =False      # output debug information about price of question
     ):
        """
        Answer a question based on the most similar context from the Qdrant texts
        Maximum tokens for text-davinci-003 is 4097
        there is done loging to Qdrant DB collection project-LOG
        -------------------------------------------------------------------------
        question - question
        user_id - unique user id
        debug - output debug information about context of question
        price - output debug information about price of question

        Log Record Structure:
        id - UUID
        vector - embedding vector
        date - date and time in form yyyy-mm-dd HH:MM:SS
        q - question
        a - answer

        returns answer
        """

        st = time.time()

        question = question.strip()

        # get last result from user communication history
        if user_id in self.user_history.keys():
            last_result = self.user_history[user_id]
        else:
            last_result = {}


        conv_flag = None           # conversation flag ResultQA(Enum)
        qa_next = []

        if self.is_qa or len(last_result) == 0:
            nqa = 0                    # # of qa tokens
        else:
            nqa = last_result["nqa"]
        
        # check nonsense question
        if len(question) < 2:
            if nqa == 0:
                qa_next = []
                conv_flag = ResultQA.FirstQuestion
            else:
                conv_flag = ResultQA.NextQuestion
                qa_next = last_result["qa"]

            answer = "Zadejte prosím otázku."
            if not self.is_qa:
                self.user_history[user_id] = {"ta":time.time(), "a": answer, "qa":qa_next, "nqa":nqa, "status":"F", "conv":conv_flag}

            return answer

        parts = self.get_prompt_parts(question)
        prompt_frame, language, nf = parts

        nq = len(self.encoding.encode(question))    # tokens of question
 
        # calculating # tokens of context result
        nc = min(self.maxc, self.maxmod - nf - nqa - nq - self.maxa)

        # depends and check for conversation only
        if not self.is_qa:
            chain = self.question_depends(question, last_result, nc)
            match chain:
                case ConverChain.NoChain | ConverChain.Overflow | ConverChain.Timeout:
                    last_result["qa"]  = []
                    last_result["nqa"] = 0
                    last_result["ta"]  =time.time()
                    nqa = 0
                    nc = min(self.maxc, self.maxmod - nf - nqa - nq - self.maxa)    # recalcilating nc

                case ConverChain.Error:
                    raise Exception("question_depends error")
 
            if chain == ConverChain.Overflow:
                conv_flag = ResultQA.Overflow
            if chain == ConverChain.Timeout:
                conv_flag = ResultQA.Timeout

        if nc < self.minc:
            raise Exception("Context length < " + str(self.minc))

        # creating context from embeddings
        context_question = ""
        if len(last_result) > 0:
            for qa_record in last_result["qa"]:
                context_question += qa_record[0] + "\n"
        context_question += question

        failure, context = self.create_context_from_db(
            question = context_question,
            language = language,     #question language
            max_len=nc,
            debug = debug,
        )

        if failure != "":
            return failure

        # creating prompt messages
        system_content = prompt_frame + context
 
        messages_lch = [SystemMessage(content=system_content)]
        if len(last_result) > 0:
            for qa_record in last_result["qa"]:
                messages_lch.append(HumanMessage(content=qa_record[0]))
                messages_lch.append(AIMessage(content=qa_record[1]))
        
        messages_lch.append(HumanMessage(content=question))

        # ask chat for answer
        try:
            with get_openai_callback() as cb:
                answer = self.chat(messages_lch).content
                total_tokens = cb.total_tokens
                prompt_tokens = cb.prompt_tokens
                completion_tokens = cb.completion_tokens

            na = completion_tokens

            # price calculation of the question
            if price:
                question_tokens = len(self.encoding.encode(question))

                price = (question_tokens * 0.0001 + (total_tokens - na) * self.model_price_inp + na * self.model_price_out) / 1000

                message = f"Tokens: Prompt={prompt_tokens} Completion={na} Total={total_tokens} "
                message += f"Price: ({question_tokens} x 0.0001 + {total_tokens  - na} x {self.model_price_inp} + {na} * {self.model_price_out})/1000 = {price:8.6} $ ... {price * 21.5:8.6} Kč"    
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

            if conv_flag == None:
                if nqa == 0:
                    conv_flag = ResultQA.FirstQuestion
                else:
                    conv_flag = ResultQA.NextQuestion

            status = "Y"
            
            # conversation mode
            if not self.is_qa:
                if nqa == 0:
                    qa_next = [(question, answer)]
                else:
                    qa_next = last_result["qa"]
                    qa_next.append((question, answer))

            nqa_next = nqa + nq + na

            et = time.time()

            if price:
                print(f"Elapsed time {round(et - st, 3)} seconds") 

            if not self.is_qa:
                self.user_history[user_id] = {"ta":et, "a": answer, "qa":qa_next, "nqa":nqa_next, "status":status, "conv":conv_flag}

            return answer

        except Exception as e:
            if not self.is_qa:
                self.user_history[user_id] = {"ta":time.time(), "a": e, "qa":[], "nqa":0, "status":"E", "conv":ResultQA.EndConversation}

            print(colored(e, "red"))
            return e

    def answer_question_print(self,
        question:str    ="Kolik má Multima zaměstnanců?",
        debug:bool      =False,     # output debug information about context of question
        price:bool      =False      # output debug information about price of question
         ):
        """
        Print Answer/question to system output
        -------------------------------------------------------------------------
        """
        print("\n" + "Q: " + question)
        print("A:" + self.answer_question(question=question, debug=debug, price=price))     

    def get_status(self,
        user_id:str =""        # user id          
                 ):
        """
        Get status from last conversation
        -------------------------------------------------------------------------
        """
        if user_id in self.user_history.keys():
            last_result = self.user_history[user_id]
            status = last_result["status"]
        else:
            status = ""

        return status

    def get_conv(self,
        user_id:str =""        # user id          
                 ):
        """
        Get result of last conversation
        -------------------------------------------------------------------------
        """
        if user_id in self.user_history.keys():
            last_result = self.user_history[user_id]
            conv = last_result["conv"]
        else:
            conv = None

        return conv

    def user_history_cleaning(self):
        """
        Remove records from self.user_history, where is timeout since last activity
        -------------------------------------------------------------------------
        """
        actual_time = time.time()
      
        remove = [k for k in self.user_history if actual_time - self.user_history[k]["ta"] > self.timeout]
        for k in remove: 
            del self.user_history[k]