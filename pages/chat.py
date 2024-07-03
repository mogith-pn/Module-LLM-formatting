import streamlit as st

from clarifai.client.search import Search
from clarifai.client.model import Model
from clarifai.modules.css import ClarifaiStreamlitCSS

from pydantic import create_model
from typing import List, Union

from utils import *

ClarifaiStreamlitCSS.insert_default_css(st)
url_params = st.experimental_get_query_params()
search_obj = Search(app_id=url_params['app_id'][0], user_id=url_params['user_id'][0], pat=url_params['pat'][0], top_k=5)

@st.cache_data
def model_Select(pat):
  llm_params = list_all_models(pat)
  return llm_params

st.title("Structure LLM outputs")

if "start_chat" not in st.session_state.keys():
  st.session_state.start_chat = False

with st.sidebar:
  st.title('Enter your desired output format with data type')
  dynamic_fields = {}

  # Number of fields user wants to add
  num_fields = st.number_input('Enter number of fields', min_value=1, max_value=10)

  for i in range(num_fields):
      field_name = st.text_input(f'**Enter name for field {i+1}**', value='response', placeholder='Product_id, name, price, etc..')
      field_type = st.text_input(f'**Enter type for field {i+1}**', value='str', placeholder='List, int, str, Union[int, str], etc..')
      try:
        dynamic_fields[field_name] = (eval(field_type, {'Union': Union, 'List': List, 'int': int, 'str': str, 'float': float, 'bool': bool}), ...)
      except:
        st.error('Invalid data type entered. Please enter a valid data type')
      
  # Create the dynamic model
  if st.button('Create Model'):
      DynamicModel = create_model('DynamicProduct', **dynamic_fields) 
      dict_keys = {k:v for k,v in DynamicModel.schema().items()}
      
      if 'DynamicModel' not in st.session_state.keys():
          st.session_state.DynamicModel = dict_keys

      # Display the class structure
      st.code(DynamicModel.schema_json(indent=2))
      
if st.session_state.start_chat==False:
  if st.button("start_chat"):
      st.session_state.start_chat = True

if "DynamicModel" in st.session_state.keys() and st.session_state.start_chat:
  query_params = st.experimental_get_query_params()
  llm_params = model_Select(query_params['pat'][0])
  selected_llm = st.selectbox("**Select the LLM you want to use:**", list(llm_params.keys()), index=62)
  st.write(f"Selected LLM: {llm_params[selected_llm]}")
  model_obj = Model(url=llm_params[selected_llm], pat = query_params['pat'][0])


if st.session_state.start_chat:
  
  is_rag = st.toggle("Enable RAG")
  request =  st.text_area('Enter your query here', height=200)
  
  if request :
    with st.spinner("Generating response"):
      if is_rag:
        format_instructions = pydantic_prompt_format(st.session_state.DynamicModel)
        response = chat(search_obj, request, model_obj, format_instructions, query_params['pat'][0])
        st.code(response)
      
      else:
        format_instructions = pydantic_prompt_format(st.session_state.DynamicModel)
        response = chat(search_obj, request, model_obj, format_instructions, query_params['pat'][0], is_rag=False)
        st.json(response)