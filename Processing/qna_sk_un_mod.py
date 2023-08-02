# encoding: utf-8
# Building a question answer system for Slovakia Universities
# https://platform.openai.com/docs/tutorials/web-qa-embeddings

import os
import sys
from langdetect import detect
from deep_translator import GoogleTranslator    # pip install -U deep-translator     https://deep-translator.readthedocs.io/en/latest/README.html
import openai
import time
import math
import json
from datetime import datetime
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.http import models
#import pandas as pd
import uuid
from enum import Enum
from langchain import PromptTemplate
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
#from langchain.embeddings.openai import OpenAIEmbeddings
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

class DebugFlag(Enum):
    Context = 1,    # protocol conteq of question
    Time = 2,       # protocol elapsed time
    Price = 3,      # protocol price paid
    Params = 4,     # protocol question params
    Headings = 5,   # protocol question headings
    Question = 6,   # protocol question
    Answer = 7,     # protocol answer
    History = 8,    # protocol question/answer history

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
    #   log_db - loging query/answer to Qdrant database (True = yes, False = No)
    #   debug_flag - loging debug information (DebugFlag)
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
        log_db:bool = False,
        debug_flag:list(DebugFlag) = [],
        ):
        print(colored("*** Question and Answer started ***", "cyan"))
        st = time.time()
        
        self.collection = project       # name of collection in Qdrant database
        self.original_language = original_language  #
        self.maxs = maxs                # maximum tokens in chunk
        self.minc = minc                # minimum tokens in context
        self.maxc = maxc                # maximum tokens in context
        self.maxa = maxa                # maximum tokens of answer
        self.timeout = timeout          # timeout for conversation (chain communication)
        self.is_qa = is_qa              # is_qa - True - query/answer, False - chain communication
        self.log_db = log_db            # loging query/answer to Qdrant database(True = yes, False = No)
        self.debug_flag = debug_flag    # debugging flags

        #self.model = "gpt-3.5-turbo-0613"    # OpenAI model (gpt-3.5-turbo)
        self.maxmod = 4096              # maximum tokens of the model in ChatCompletion
        self.model_price_inp = 0.0015   # price of 1000 input tokens
        self.model_price_out = 0.002    # price of 1000 output tokens

        self.user_history = {}          # communication history with users. Dictionary   ("user_id1":{last_result}, ...)
                                        # last_result
                                        # {"ta":et, "a": answer, "qa":qa_next, "nqa":nqa_next, "status":status, "conv":conv_flag}
                                        # et - time last answer
                                        # answer - last answer
                                        # qa_next- list conversation [(question, answer), ...]
                                        # nqa_next - number tokens of qa_next
                                        # status - status of answer
                                        #   Y – vygenerovaná odpověď
                                        #   N – neznámá odpověď
                                        #   F – nesmyslný dotaz. Odpověď nemohla být vygenerována.
                                        #   E – neočekávaná chyba při generování

                                        # conv_flag - conversatin flag (see ConverChain(Enum))
        self.history_limit = 5          # maximální počet záznamů v self.user_history pro každého uživatele
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
                                #model_name="gpt-3.5-turbo",
                                model_name="gpt-4",
                                temperature=0)

            self.ada_engine = "ada"                         # engine for embeddings

        else:
            self.chat = ChatOpenAI(
                                #model_name="gpt-3.5-turbo",
                                model_name="gpt-4",
                                temperature=0,
                                openai_api_key=os.getenv("OPENAI_API_KEY"))

            self.ada_engine = "text-embedding-ada-002"      # engine for embeddings

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
        print(f"Elapsed starting time {round(et - st, 3)} s\n") 

   
    def get_un_dimension(self,
        dim_type:str,
        text:str
        )->str:
        '''
        Get dimension for type
        ------------------------------------------
        dim_type - dimension type (Univerzita, Fakulta, Program)
        text - checking dimension name text
        return correct dimension name

        dim_type        text                                return
        Univerzita      přibližný název univerzity          přesný název univerzity
        Fakulta         přibližný název fakulty             přesný název fakulty
        Program         přibližný název stud. programu      přesný název stud. programu
        '''

        st = time.time()

        dimension = ""

        # Get the embeddings for the question
        text_embeddings = openai.Embedding.create(
            input=text,
            engine=self.ada_engine)['data'][0]['embedding']

         # search vectors in Qdrant with the best distance
        try:
            result = self.qdrant_client.search(
                collection_name = self.collection,
                search_params=models.SearchParams(
                    hnsw_ef=128,
                    exact=True
                ),
                query_vector = text_embeddings,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="heading",
                            match=models.MatchValue(
                                value=dim_type,
                            ),
                        )
                    ]
                ),
                limit = 2,
            )

            if len(result) > 0:
                match dim_type:
                    case "Univerzita":
                        dimension = result[0].payload["university"]
                    case "Fakulta":
                        dimension = result[0].payload["faculty"]
                    case "Program":
                        dimension = result[0].payload["program"]
 
        except Exception as e:
            print(colored(f"Qdrant search exception: {e}", "red")) 


        if DebugFlag.Time in self.debug_flag:
            et = time.time()
            print(f"Dimension '{dim_type}' time: {round(et - st, 3)} s") 

        return dimension


    def get_question_parameters(self,
        question:str,    
        )->dict:
        '''
        Get question parameters:
            question_type - Detail, Vyber
            question_subject  - Univerzita, Fakulta, Program
            university - název univerzity
            faculty - název fakulty
            form - forma studia
            program - název studijního programu
            code - kód studijního programu
            content - v případě, že se otázka týkala univerzit obsahuje None. Jestliže je úplně mimo toto téma, je zde naplněna odpověď.
        '''
  
        def get_parameters(question_type, question_subject, university, faculty, form, program, code):
            """Get question parameters by Chat GPT ChatCompletion"""
            parameters = {
                "question_type":    question_type,
                "question_subject": question_subject,
                "university":       university,
                "faculty":          faculty,
                "form":             form,
                "program":          program,
                "code":             code,
            }
            return json.dumps(parameters)

        st = time.time()

        # parameters JSON schema
        schema = {
            "type": "object",
            "properties": {
                "question_type": {
                    "type": "string",
                    "enum": ["Detail", "Výber"],
                    "description": "Druh otázky:\n" \
                                   "Detail - Či ide o získanie detailných informácií o subjektu dotazu, ktorý je v otázke už uvedený (univerzita, fakulta, študijný program).\n" \
                                   "Výber - Či ide o nalezenie predmetu dotazu (univerzita, fakulta, študijný program).",
                },
                "question_subject": {
                    "type": "string",
                    "enum": ["Študijný program", "Fakulta", "Univerzita"],
                    "description": "Akého subjektu sa otázka týka.",
                },
                "university": {
                    "type": "string",
                    "description": "Názov university, ktorý je v otázke uveden.",
                }, 
                "faculty": {
                    "type": "string",
                    "description": "Názov fakulty, ktorá je v otázke uvedena.",
                },
                "form": {
                    "type": "string",
                    "enum": ["Bakalárske štúdium", "Inžinierske a Magisterské štúdium", "Doktorandské štúdium"],
                    "description": "Forma štúdia, ktorá je v otázke uvedena",
                },
                "program": {
                    "type": "string",
                    "description": "Študijný program (obor), ktorý je v otázke uveden.",
                },
                "code": {
                    "type": "string",
                    "description": "Kód študijného programu (oboru), ktorý je v otázke uveden.",
                },
            },
            "required": ["question_type", "question_subject"],
        }


        # ChatCompletion functions definition
        functions = [
            {
                "name": "get_parameters",
                "description": "Získajte dôležité parametre otázky.",
                "parameters": schema,
            },
        ]

        messages = [
            {"role": "system", "content": "Ste užitočný univerzitný poradca." },
            {"role": "user", "content": question},
        ]

        response = openai.ChatCompletion.create(
            #model="gpt-3.5-turbo",
            model="gpt-4",
            temperature=0,
            messages=messages,
            functions=functions,
            function_call="auto",  # auto is default, but we'll be explicit
        )

        response_message = response["choices"][0]["message"]

        if response_message.get("function_call"):
            function_args = json.loads(response_message["function_call"]["arguments"])
            function_args["content"] = None
        else:
            function_args = {"content":response_message.content}
            if DebugFlag.Params in self.debug_flag:
                print(f"Params: {function_args}")

            return function_args
 
        # correction of university, faculty, program
        question_type = function_args["question_type"]        # Detail, Výber
        question_subject = function_args["question_subject"]  # "Študijný program", "Fakulta", "Univerzita"

        if not (question_type == "Výber" and question_subject == "Univerzita"):
            if "university" in function_args:
                function_args["university"] = self.get_un_dimension("Univerzita", function_args["university"])

        if  (question_type == "Výber" and question_subject == "Študijný program") or (question_type == "Detail" and question_subject in ["Fakulta", "Študijný program"]):
            if "faculty" in function_args:
                function_args["faculty"] = self.get_un_dimension("Fakulta", function_args["faculty"])

        if  question_type == "Detail" and question_subject == "Študijný program":
            if "code" not in function_args and "program" in function_args:   # if program code is presented then program name is not used
                function_args["program"] = self.get_un_dimension("Program", function_args["program"]) 

        if DebugFlag.Params in self.debug_flag:
            print(f"Params: {function_args}")


        if DebugFlag.Time in self.debug_flag:
            et = time.time()
            print(f"Params time: {round(et - st, 3)} s") 

        return function_args


    def get_question_headings(self,
        question_type:str,
        question_subject:str,
        )->list:
        '''
        Get headings list
        ------------------------------------------------
        question_type - Detail, Výber
        question_subject - Univerzita, Fakulta, Program
        Výstup:
        Seznam názvů heading dle tabulky výše
        '''

        if question_type == "Detail":
            match question_subject:
                case "Univerzita":
                    head_list = ["Profil univerzity"]
                case "Fakulta":
                    head_list = ["Profil univerzity", "Profil fakulty", "Detail fakulty", "Ubytovanie a stravovanie"]
                case "Študijný program":
                    head_list = ["Predmet", "Uplatnenie absolventov", "Podmienky prijatia", "Podmienky pro zahraničných študentov", "Podmienky prijatia bez prijímacej skúšky",
                                 "Všeobecné informácie k prijímacej skúške", "Školné, poplatky a fakturačné údaje", "Detail študijného programu"]

        else:           # Výber
            match question_subject:
                case "Univerzita":
                    head_list = ["Profil univerzity"]
                case "Fakulta":
                    head_list = ["Profil univerzity", "Profil fakulty", "Detail fakulty"]
                case "Študijný program":
                    head_list = ["Predmet", "Uplatnenie absolventov", "Zoznam študijných programov"]


        if DebugFlag.Headings in self.debug_flag:
            print(f"Headings: {head_list}")

        return head_list

    def get_qdrant_condition(self,
        question:str,
        ):      # ->Tuple[str, list]
        '''
        Construction filter condition for search operation in Qdrant database
        ------------------------------------------------
        question_parameters - question parameters from get_question_parameters
        question_headings - list selected headings from get_question_headings
        Return:
        (failure answer, List of FieldCondition for query_filter of search operation)
        '''
        # get question parameters
        question_parameters = self.get_question_parameters(question)
        if "question_type" not in question_parameters:
            return (question_parameters["content"], [])

        st = time.time()

        # get question headings
        question_type = question_parameters["question_type"]
        question_subject = question_parameters["question_subject"]
        question_headings = self.get_question_headings(question_type, question_subject)
 
        # Construction filter condition
        condition_list = []
        param_list = []

        question_type = question_parameters["question_type"]        # Detail, Výber
        question_subject = question_parameters["question_subject"]  # "Študijný program", "Fakulta", "Univerzita"

        # correction of university, faculty, program
        if not (question_type == "Výber" and question_subject == "Univerzita"):
            if "university" in question_parameters:
                param_list.append(("university", question_parameters["university"]))

        if  (question_type == "Výber" and question_subject == "Študijný program") or (question_type == "Detail" and question_subject in ["Fakulta", "Študijný program"]):
            if "faculty" in question_parameters:
                param_list.append(("faculty", question_parameters["faculty"]))

        if  (question_type == "Výber" and question_subject == "Študijný program") or (question_type == "Detail" and question_subject == "Študijný program"):
            if "form" in question_parameters:
                param_list.append(("form", question_parameters["form"]))


        if  question_type == "Detail" and question_subject == "Študijný program":
            if "code" in question_parameters:   # if program code is presented then program name is not used
                param_list.append(("code", question_parameters["code"]))
            else:
                if "program" in question_parameters:
                    param_list.append(("program", question_parameters["program"]))

        # conditions for question parameters
        for heading, value in param_list:
            condition_list.append(
                models.FieldCondition(
                    key=heading,
                    match=models.MatchValue(
                        value=value,
                    ),
                )
            )

        # conditions for question headings
        if len(question_headings) > 0:
            condition_list.append(
                models.FieldCondition(
                    key="heading",
                    match=models.MatchAny(
                        any=question_headings,
                    ),
                )
            )

        if DebugFlag.Time in self.debug_flag:
            et = time.time()
            print(f"Qdrant condition time: {round(et - st, 3)} s") 



        return ("", condition_list)




    def get_prompt_parts(self, 
        question:str):
        '''
        Get prompt parts in language of the question
        ------------------------------------------
        question - Question
        return tuple of the question prompt (part1, part2, part3, language, nf)
        '''

        # detection of the question language
        # there are assumption that is original_language. On other side is made language detection
        language = self.original_language
        cs_question = GoogleTranslator(source='auto', target=language).translate(question) 
        if cs_question != question:
            language = detect(question)

 
        # system message
        text_part = ["Jsi chytrý AI bot na portále vysokých škol Slovenska a radíš studentům. Odpověz na základě kontextu uživatele. Když nevíš nebo nejsi jistý, odpověz \"Nevím\". Odpověz výstižně.",
                     "Kontext:"
            ]
 
        match language:
            case "cs":
                nf = 94

            case "en":
                text_part = ["You are a smart AI bot on the Slovak university portal and you advise students. Answer the question based on the user's context. If you don't know then respond \"I don't know\". Keep the answer concise.",
                             "Context:"
                    ]
                nf = 48

            case "sk":
                text_part = ["Si šikovný AI bot na portáli vysokých škôl Slovenska a radíš študentom. Odpovedz na základe kontextu užívateľa. Keď nevieš alebo nie si istý, odpovedz \"Neviem\". Odpovedz výstižne.",
                             "Kontext:"
                    ]
                nf = 91

            case _:
                # in different language is needed translation to language of question
                text_part = GoogleTranslator(source="cs", target=language).translate_batch(text_part)
                nf = len(self.encoding.encode(text_part[0] + " " + text_part[1]))
          
        prompt_frame = text_part[0] + "\n\n" + text_part[1] + "\n"

        return (prompt_frame, language, nf)

    def agregate_context(self,
        db_list,            # record list from database
        title_list:list=["title"],
        max_len:int=1800    # maximum length of context
                     )->list:
        '''
        Agregate list of chunks for the same title. Insert title to text when it's missing.
        -----------------------------------------------------------------------------------
        db_list - record list from database
        title_list - list of key field names from vector DB
        max_len - maximum length of context
        '''

 
        # 1 - creating list of selected parts of context <= max_len with add prefix from payload of title_list
        agregate_list = []
        if len(db_list) == 0:
            return agregate_list
 
        prefix_last = ""
        cur_len = 0     # current length of aggregate chunks

        for record in db_list:
            prefix = ""
            append_chunk = False    # if agreagete chunk will be add to list
            for title in title_list:
                if title != "heading":  # heading is included in text
                    prefix += record.payload[title] + "\n"

            chunk_len = record.payload["n_tokens"] + 1
            if prefix != prefix_last:
                chunk = prefix + record.payload["text"]
                chunk_len += len(self.encoding.encode(prefix))
                prefix_last = prefix
                if prefix_last != "":
                    append_chunk = True
            else:
                chunk += "\n\n" + record.payload["text"]

            # Add the length of the text to the current length
            cur_len += chunk_len
 
            # If the context is too long, break
            if cur_len > max_len:
                break

            if append_chunk:
                agregate_list.append(chunk)

        agregate_list.append(chunk) # append last agregate chunk

        return agregate_list

    def create_context_from_db(self,
        question:str,       # question
        language:str,       # question language
        max_len:int=1800,   # maximum length of context
        ):
        """
        Create a context for a question by finding the most similar context from the vector database Qadrant
        ----------------------------------------------------------------------------------------------------
        question - question
        language - questioin language
        max_len - maximum length of context
        return (failure_text, context)
        """

        failure_answer, qdrant_condition = self.get_qdrant_condition(question)
        if failure_answer != "":
            if language != self.original_language:
                failure_answer = GoogleTranslator(source=self.original_language, target=language).translate(failure_answer)  
            return (failure_answer, "")

        st = time.time()

        # Get the embeddings for the question
        emb_question = question
        if language != self.original_language:
            emb_question = GoogleTranslator(source=language, target=self.original_language).translate(question)  

        self.question_embeddings = openai.Embedding.create(
            input=emb_question,
            engine=self.ada_engine)['data'][0]['embedding']


        # setup limit for search operation in Qdrant
        search_limit = math.ceil(max_len / self.maxs) + 1


        # search vectors in Qdrant with the best distance
        try:
            result = self.qdrant_client.search(
                collection_name = self.collection,
                query_filter=models.Filter(
                    must=qdrant_condition
                ),

                search_params=models.SearchParams(
                    hnsw_ef=128,
                    exact=True
                ),
                query_vector = self.question_embeddings,
                limit = search_limit,
            )
        except Exception as e:
            print(colored(f"Qdrant search exception: {e}", "red")) 
            sys.exit()


        title_list = ["university", "faculty", "form", "code", "program", "heading"]
        returns = self.agregate_context(result, title_list, max_len)
        if len(returns) == 0:
            failure = "Pre zadanú otázku nemám k dispozícii žiadne informácie. Skúste zmeniť formuláciu."
            if language!= "sk":
                failure = GoogleTranslator(source="sk", target=language).translate(failure)
            return (failure, "")

        context = "\n\n###\n\n".join(returns)

        # If debug, print the raw model response
        if DebugFlag.Context in self.debug_flag:
            print("Context:\n" + context)
            print("\n\n")

        if DebugFlag.Time in self.debug_flag:
            et = time.time()
            print(f"Qdrant context time: {round(et - st, 3)} s") 

        # Return the context
        return ("", context)

    def question_depends(self,
        question:str,
        last_result:dir ={},            # last result of answer_question
        nc:int = 0                      # length of context
    )->ConverChain:
        """
        It figure out the question depends on last communication
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


        ''' dočasně vyřazeno - předpokládá se vždy závislost na předchozích otázkách


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
        ''' 
        return ConverChain.Chain

    def answer_question(self,
        question:str        ="Kolik má Multima zaměstnanců?",
        user_id:str         ="",        # user id
     ):
        """
        Answer a question based on the most similar context from the Qdrant texts
        Maximum tokens for text-davinci-003 is 4097
        there is done loging to Qdrant DB collection project-LOG
        -------------------------------------------------------------------------
        question - question
        user_id - unique user id

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

        if DebugFlag.Question in self.debug_flag:
            print (f"Q: {question}")

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

        prompt_frame, language, nf = self.get_prompt_parts(question)

        nq = len(self.encoding.encode(question))    # tokens of question
 
        # calculating # tokens of context result
        nc = min(self.maxc, self.maxmod - nf - nqa - nq - self.maxa)

        # depends and check for conversation only
        if not self.is_qa:
            st2 = time.time()

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

            if DebugFlag.Time in self.debug_flag:
                et = time.time()
                print(f"Chain conversation time: {round(et - st2, 3)} s") 

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
        )

        if failure != "":
            if DebugFlag.Answer in self.debug_flag:
                print (f"A: {failure}")
            
            return failure

        st3 = time.time()

        # creating prompt messages
        system_content = prompt_frame + context
 
        messages_lch = [SystemMessage(content=system_content)]
        if len(last_result) > 0:
            for qa_record in last_result["qa"]:
                messages_lch.append(HumanMessage(content=qa_record[0]))
                messages_lch.append(AIMessage(content=qa_record[1]))

            if DebugFlag.History in self.debug_flag:
                print ("Conversation history:")
                for qa_record in last_result["qa"]:
                    print(f"   Q: {qa_record[0]}\n   A: {qa_record[1]}")

        
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
            if DebugFlag.Price in self.debug_flag:
                question_tokens = len(self.encoding.encode(question))

                price = (question_tokens * 0.0001 + (total_tokens - na) * self.model_price_inp + na * self.model_price_out) / 1000

                message = f"Tokens: Prompt={prompt_tokens} Completion={na} Total={total_tokens} "
                message += f"Price: ({question_tokens} x 0.0001 + {total_tokens  - na} x {self.model_price_inp} + {na} * {self.model_price_out})/1000 = {price:8.6} $ ... {price * 21.5:8.6} Kč"    
                print(message)

            if DebugFlag.Time in self.debug_flag:
                et = time.time()
                print(f"Answer time: {round(et - st3, 3)} s") 


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
                    
                    # shortening qa_next to history_limit
                    if len(qa_next) > self.history_limit:
                        del qa_next[0]

            nqa_next = nqa + nq + na

            et = time.time()

            if not self.is_qa:
                self.user_history[user_id] = {"ta":et, "a": answer, "qa":qa_next, "nqa":nqa_next, "status":status, "conv":conv_flag}

            if DebugFlag.Answer in self.debug_flag:
                print (f"A: {answer}")

            if DebugFlag.Time in self.debug_flag:
                print(f"Complet time: {round(et - st, 3)} s") 

            print("")
            return answer

        except Exception as e:
            if not self.is_qa:
                self.user_history[user_id] = {"ta":time.time(), "a": e, "qa":[], "nqa":0, "status":"E", "conv":ResultQA.EndConversation}

            print(colored(e, "red"))
            return e

    def answer_question_print(self,
        question:str    ="Kolik má Multima zaměstnanců?",
        ):
        """
        Print Answer/question to system output
        -------------------------------------------------------------------------
        """
        print("\n" + "Q: " + question)
        print("A:" + self.answer_question(question=question))     

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