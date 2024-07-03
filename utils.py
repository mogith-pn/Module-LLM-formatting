import requests 
import json
import streamlit as st     

from pydantic import BaseModel, validator
from typing import List, Union

from google.protobuf.json_format import MessageToDict, ParseDict
from concurrent.futures import ThreadPoolExecutor

from clarifai.client.app import App

def list_all_models(pat):
  llm_community_models = App(pat=pat).list_models(filter_by={"query": "LLM"},
                                           only_in_app=False)
  model_list={}
  for model in llm_community_models:
    user_id = model.user_id
    app_id = model.app_id
    model_id = model.id
    model_list[f"{user_id}:{(model_id.replace('_','.'))}"]=f"https://clarifai.com/{user_id}/{app_id}/models/{model_id}"
    sorted_dict = {k: model_list[k] for k in sorted(model_list)}
    
  return sorted_dict  

def pydantic_prompt_format(schema):

    return f"""The output should be formatted as a JSON instance that conforms to the JSON schema below.
    As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
    the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.
    Here is the output schema:
    ```
    {schema}
    ```
    Return only the json output, don't include "``` json ```" separator. 
    """ 

def prompt_template_rag(query, context_hits, format_instructions):

  prompt = f"""You are a polite support agent. A customer asks you: "{query}". Provide a helpful response by referring to the context. 
  Here are relevant <Context>: {context_hits} </context>. 
  IMPORTANT:
  1. Be polite and helpful.
  2. Generated response should adhere to below instruction standards.

  <Instruction> : {format_instructions} </Instruction>
  """
  return prompt

def prompt_template_chat(query, format_instructions):
  prompt=f"""You are a polite chat bot, your task is to answer user query in a concise and with proper formatting.
  This is the user query : {query}
  IMPORTANT:
  Please follow this output formatting instructions strictly.
  format instructions : {format_instructions}
  """
  return prompt
  
def search_hits_and_metadata(search_obj, query):
    res = list(search_obj.query(ranks=[{"text_raw": query}],
                                filters=[{'input_types': ['text']}]))
    hits = [hit.input.data.text.url for hit in res[0].hits]
    return hits
  
def retrieve_hits_with_metadata(search_obj, query, pat):
    hits_url = search_hits_and_metadata(search_obj, query)
    executor = ThreadPoolExecutor(max_workers=10)
    
    def hit_to_document(hit, pat):
      h= {"authorization": f"Bearer {pat}"}
      request = requests.get(hit, headers=h)
      request.encoding = request.apparent_encoding
      requested_text = request.text
      return (requested_text)

    # Iterate over hits and retrieve hit.score and text
    futures = [executor.submit(hit_to_document, hit, pat) for hit in hits_url]
    contexts = list(set([future.result() for future in futures])) 
    return contexts
    
def generate_chat_response(query, contexts, model_obj, format_instructions, is_rag):
    if is_rag:
      prompt = prompt_template_rag(query, contexts, format_instructions)
    else:
      prompt = prompt_template_chat(query, format_instructions)
    try:
      response = model_obj.predict_by_bytes(bytes(prompt,"utf-8"), input_type="text", inference_params={"max_tokens":4000})
      try:
        return json.dumps(response.outputs[0].data.text.raw, indent=4)
      except Exception as e:
        print(f"Response from model: {response.outputs[0].data.text.raw}")
        print(f"Error in json conversion:{e}")
    except Exception as e:
      print(f"Error in generating response: {e}, Please try again!")

    
def chat(search_obj, query, model_obj, format_instructions, pat, is_rag=True):
  context = (retrieve_hits_with_metadata(search_obj, query, pat)) if is_rag  else None
  resp = generate_chat_response(query, context, model_obj, format_instructions, is_rag)
  return json.loads(resp)