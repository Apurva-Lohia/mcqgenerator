import os
import re
import json
import pandas as pd
import traceback
import boto3
import PyPDF2
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import read_file,get_table_data

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

model_kwargs =  { 
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

model = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

messages = [ 
    ("system", "You are a helpful assistant"),
    ("human", "Text:{text} \
        You are an expert MCQ maker. Given the above text, it is your job to \
        create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. \
        Make sure the questions are not repeated and check all the questions to be conforming the text as well.\
        Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
        Ensure to make {number} MCQs\
        ### RESPONSE_JSON\
        {response_json}"
    ),
]

prompt = ChatPromptTemplate.from_messages(messages)

chain = prompt | model | StrOutputParser()
quiz = chain.invoke({"text": TEXT, "number": 4 , "subject": "Chemistry", "tone": "friendly", "response_json": json.dumps(RESPONSE_JSON)}).strip()

messages2 = [ 
    ("system", "You are a helpful assistant"),
    ("human",
        "You are an expert english grammarian and writer. Given a Multiple Choice Quiz for students.\
        You need to evaluate the complexity of teh question and give a complete analysis of the quiz if the students\
        will be able to unserstand the questions and answer them. Only use at max 50 words for complexity analysis.\
        if the quiz is not at par with the cognitive and analytical abilities of the students,\
        update tech quiz questions which needs to be changed  and change the tone such that it perfectly fits the student abilities\
        Quiz_MCQs:{quiz}\
        Check from an expert English Writer of the above quiz:"
    ),
]

analysis_prompt = ChatPromptTemplate.from_messages(messages2)

composed_chain = {"quiz" : chain}| analysis_prompt | model | StrOutputParser()

response = composed_chain.invoke({"text": TEXT, "number": 4 , "subject": "Chemistry", "tone": "friendly", "response_json": json.dumps(RESPONSE_JSON)})
print(response)


