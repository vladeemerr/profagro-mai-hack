import os
import nltk
import faiss
import torch
import PyPDF2
import chromadb
import typing as tp
from gc_api import GigaChatAPI
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from langchain.memory import ConversationBufferMemory
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PERSIST_DIRECTORY = "./chroma_db"  # Директория для сохранения базы данных
COLLECTION_NAME = "my_rag_collection"
DIMENSION = 1024  # Размерность ваших эмбеддингов

SYS_PROMPT = '''Ты — система для помощи сельхоз работнику, работающему с прицепным распределителем Amazone. \
                Используя информацию из документации по настройке и эксплуатации распределителя Amazone, твоя задача — автоматически генерировать ответ на вопрос пользователя. \
                Ответ должен быть максимально приближенным и подробным и основываться только на информацию из контекста.'''

CHECK_LIST_PROMPT = '''Ты — система для помощи сельхоз работнику, работающему с прицепным распределителем Amazone.  
                        Используя информацию из документации по настройке и эксплуатации распределителя Amazone, 
                        твоя задача — автоматически генерировать список шагов, которые необходимо выполнить пользователю
                        Чек-лист должен быть структурированным, каждый пункт кратким, но в нем много информации'''

reranker_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
reranker = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
reranker.eval()

api = GigaChatAPI(contour=None,
                  rquid=RQUID,
                  auth_data=AUTH_DATA,
                  oauth_url=total_auth['mlspace']['OAUTH_URL'],
                  api_url=total_auth['mlspace']['API_URL'],
                  token=total_auth['mlspace']['TOKEN'])

def get_text_from_pdf(pdf_docs: tp.List[str]) -> str:
    texts = []
    for pdf in pdf_docs:
        text = ''
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        texts.append(text)
    return texts


def get_text_chunk(raw_text: str,
                   chunk_size: int=1000,
                   chunk_overlap: int=200,
                   length_function: tp.Callable[[tp.Any], int]=len):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   length_function=length_function)
    chunks = text_splitter.split_text(raw_text)
    return chunks  


def get_text_chunks(texts: tp.List[str],
                    chunk_size: int=1000,
                    chunk_overlap: int=300,
                    length_function: tp.Callable[[tp.Any], int]=len):
    chunks_lst = [get_text_chunk(x,
                                 chunk_size,
                                 chunk_overlap,
                                 length_function) for x in texts]
    return chunks_lst


def make_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        embeddings_dict = api.get_embedding(corpus=chunk,
                                            profanity_check=False)
        embeddings_list = [x['embedding'] for x in embeddings_dict]
        embeddings += embeddings_list
    return embeddings
        

def create_vectorstorage(chunks_list: tp.List[str],
                         embed: tp.List[tp.Any],
                         collection_name: str):
    chunks = []
    for chunk in chunks_list:
        chunks += chunk
    client = chromadb.Client()
    collection = client.create_collection(name=collection_name)
    ids = [str(idx) for idx in range(1, len(embed)+1)]
    collection.add(ids=ids,
                   documents=chunks,
                   embeddings=embed)
    return collection


def get_result_from_vectorstorage(collection,
                                  query_text: str,
                                  n_results: int=5):
    result = collection.query(query_texts=[query_text],
                              query_embeddings=make_embeddings([query_text]),
                              n_results=n_results)
    return result


def get_retriever_result(query_text: str,
                         chunks_list: tp.List[str],
                         n_results: int=5):
    chunks = []
    for chunk in chunks_list:
        chunks += chunk
    tokenized_chunks = [word_tokenize(chunk) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = word_tokenize(query_text)
    scores = bm25.get_scores(tokenized_query)
    ranked_chunks = sorted(zip(scores, chunks), 
                           key=lambda x: x[0], 
                           reverse=True)
    top_n_results = ranked_chunks[:n_results]
    return [x[1] for x in top_n_results]

# TODO: Нужно возвращать не только строку с топом чанков, но и список используемых чанков.
# Нужно добавить вывод самих чанков здесь, в get_answer_llm, а также переделать run.py
def get_top_relevant(query_text: str,
                     context: tp.List[str],
                     top_n: int=3):
    pairs = [[query_text, x] for x in context]
    with torch.no_grad():
        inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
    ranked_chunks = sorted(zip(scores, context), 
                           key=lambda x: x[0], 
                           reverse=True)
    top_n_chunks = [chunk[1] for chunk in ranked_chunks[:top_n]]
    return ''.join(top_n_chunks), top_n_chunks


def get_answer_llm(query_text: str,
                   chunks_list: tp.List[str],
                   collection: tp.Any,
                   n_results: int=5,
                   temperature: float=0.):
    context_from_chroma = get_result_from_vectorstorage(collection=collection,
                                                        query_text=query_text,
                                                        n_results=n_results)
    context_from_retriever = get_retriever_result(query_text=query_text,
                                                  chunks_list=chunks_list,
                                                  n_results=n_results)
    context_list = context_from_chroma['documents'][0] + context_from_retriever
    top_chunk_txt, top_chunk_list = get_top_relevant(query_text=query_text,
                                                     context=context_list,
                                                     top_n=5)
    question = []
    sys_message = {'role': 'user',
                   'content': SYS_PROMPT}
    question.append(sys_message)
    usr_message = {'role': 'user',
                   'content':  f'''Используя инструкцию постарайся ответить на следующий вопрос. 
                                   Отвечай максимально приближенно и подробно. 
                                   Если в ответе на вопрос возможна вариативность, опиши все возможные варианты! 
                                   Рассуждай! Старайся цитировать инструкцию. 
                                   Инструкция: {top_chunk_txt}
                                   Вопрос: {query_text}'''}
    question.append(usr_message)
    answer = api.get_answer_gigachat(question,
                                     profanity_check=False,
                                     temperature=temperature)
    return answer[0]['message']['content'], top_chunk_list
    

def get_checklist_llm(query_text: str,
                      chunks_list: tp.List[str],
                      collection: tp.Any,
                      n_results: int=20,
                      temperature: float=0.):
    context_from_chroma = get_result_from_vectorstorage(collection=collection,
                                                        query_text=query_text,
                                                        n_results=n_results)
    context_from_retriever = get_retriever_result(query_text=query_text,
                                                  chunks_list=chunks_list,
                                                  n_results=n_results)
    context_list = context_from_chroma['documents'][0] + context_from_retriever
    top_chunk_txt, top_chunk_list = get_top_relevant(query_text=query_text,
                                                     context=context_list,
                                                     top_n=20)
    question = []
    sys_message = {'role': 'user',
                   'content': CHECK_LIST_PROMPT}
    question.append(sys_message)
    usr_message = {'role': 'user',
                   'content':  f'''Используя инструкцию постарайся ответить на следующий вопрос. 
                                   Отвечай максимально приближенно и подробно. 
                                   Если в ответе на вопрос возможна вариативность, опиши все возможные варианты! 
                                   Рассуждай! Старайся цитировать инструкцию. 
                                   Инструкция: {top_chunk_txt}
                                   Вопрос: {query_text}'''}
    question.append(usr_message)
    answer = api.get_answer_gigachat(question,
                                     profanity_check=False,
                                     temperature=temperature)
    return answer[0]['message']['content'], top_chunk_list
