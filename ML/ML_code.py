from __future__ import absolute_import, division
from langchain.llms import OpenAI
import re
import random
import PyPDF2
from flask import Flask, request, jsonify
import os
from io import BytesIO
import requests
from werkzeug.utils import secure_filename
import json
import base64
import warnings
import json
import tempfile
import flask_cors
from flask import Flask
from flask_cors import CORS

warnings.filterwarnings("ignore")

os.environ["OPENAI_API_KEY"] = "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"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# PDF => TEXT

question_list = []


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text


# TEXT => 자연스러운 데이터로 변경
# 입력의 자유도 설정
llm = OpenAI(temperature=0.9, openai_api_key=os.environ["OPENAI_API_KEY"])


def text_to_data(pdf_text):
    text = "This article is a translation of the resume file. These are the words I entered on my resume, so I feel awkward with spacing and changing the lines, so please change it to a natural sentence"
    combined_text = f"{text}\n\n{pdf_text}"
    soft_pdf_text = llm(combined_text)
    return soft_pdf_text

# 질문 생성


def tec_question_generate(soft_pdf_text):
    text = "여러분이 회사 면접관인데 한 사람의 학업 정보가 담긴 이 기사를 읽고 실제 면접을 위해 10가지 질문을 할 것이라고 가정해보자. 하지만 질문을 할 때는 구체적이고 직관적으로 말해주세요. 그리고 현장에서 핵심 개념이나 키워드나 헷갈리는 개념에 대해 물어보면 좋을 것 같다. 면접관이 잘 준비하지 못했을 것에 대해 몇 가지 질문을 하는 것이 좋을 것 같습니다. 그리고 면접자를 당황하게 하는 질문이 있으면 좋을 것 같아요."
    c_text = f"{soft_pdf_text},{text}"
    tec_question = llm(c_text)
    return tec_question

# input_features = ['분야', '나이', '경력 및 프로젝트 사항', '지원회사']


def input_question_generate(*input_features):
    Field = 'IT'
    Age = 24
    Career_and_Project = ['MS 애저톤', '논문', '해커톤']
    Prospective_company = '카카오'
    input_list = [Field, Age, Career_and_Project, Prospective_company]
    c_text = "여러분이 회사 면접관인데 한 사람의 학업 정보가 담긴 이 기사를 읽고 실제 면접을 위해 10가지 질문을 할 것이라고 가정해보자. 하지만 질문을 할 때는 구체적이고 직관적으로 말해주세요. 그리고 현장에서 핵심 개념이나 키워드나 헷갈리는 개념에 대해 물어보면 좋을 것 같다. 면접관이 잘 준비하지 못했을 것에 대해 몇 가지 질문을 하는 것이 좋을 것 같습니다. 그리고 면접자를 당황하게 하는 질문이 있으면 좋을 것 같아요."
    text = "This article is a translation of the resume file. These are the words I entered on my resume, so I feel awkward with spacing and changing the lines, so please change it to a natural sentence"
    input_text = f"{input_features[0]}는 {input_list[0]}이고 {input_features[1]}는 {input_list[1]}이고 {input_features[2]}에는 {input_list[2]}가 있고 {input_features[3]}는 {input_list[3]}이다."
    combined_text = f"{input_text},{text}"
    natural_text = llm(combined_text)
    combined_text = f"{natural_text},{c_text}"
    input_question = llm(combined_text)
    return input_question

# # Load GPT-4 tokenizer and model
# tokenizer = GPT4Tokenizer.from_pretrained("gpt-4")
# model = GPT4LMHeadModel.from_pretrained("gpt-4")

# 후속 질문 생성
# def generate_followup_question(prompt, model, tokenizer):
#     input_ids = tokenizer.encode(prompt, return_tensors="pt")
#     output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
#     decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
#     return decoded_output

# 상황별 질문 선택


def fir_qt(question_list):
    llm = OpenAI(temperature=0.9)
    prompt = f"Based on the question_list : '{question_list}' , Please select the best question to ask from the list of questions in the interview situation"
    first_question = llm(prompt)
    print(first_question)
    return first_question

# 면접 질문 선택


def AskTechnicalQuestions(question_list):
    if not question_list:
        print("The list of questions is empty.")
    else:
        random_question = random.choice(question_list)
    return random_question
    # return random_question
    # 기술 질문
    # 상황에 맞는 질문을 question_list 선택

#  후속 질문 생성


def AskFollowUpQuestions(question, answer):
    llm = OpenAI(temperature=0.9)
    prompt = f"Based on the interviewee's answer: '{answer}' to {question}, 후속 질문으로 적절한 질문 하나를 만들어 주십시오."
    followup_question = llm(prompt)
    print(followup_question)
    return followup_question

# 후속 질문 여부 판반


def should_ask_followup(question, answer):
    llm = OpenAI(temperature=0.9)
    prompt = f"Based on the interviewee's answer: '{answer}' to {question}, should I ask a follow-up question?"
    followup_question = llm(prompt)
    print(followup_question)
    return "Yes" in followup_question

# 피드백


def Feedback(question, answer):
    llm = OpenAI(temperature=0.9)
    prompt = f"Based on the interviewee's answer : '{answer}' to {question}' , 답변에 대한 피드백을 작성해 주시기 바랍니다."
    feedback = llm(prompt)
    print(feedback)
    return feedback

# AI 답변


def AI_answer(question):
    llm = OpenAI(temperature=0.9)
    prompt = f"Based on the interviewee's answer: '{question}', 인터뷰 대상자가 할 수 있는 최고의 답변을 짧고 명확한 답변으로 작성해 주시기 바랍니다."
    ai_answer = llm(prompt)
    print(ai_answer)
    return ai_answer

# 수정된 답변


def Mix_answer(feedback, answer):
    llm = OpenAI(temperature=0.9)
    prompt = f"ai가 답한 피드백 참고하여 user가 답한 답변을 수정해서 짧고 명료한 답변으로 수정해 주시기 바랍니다. user : '{answer}', ai: '{feedback}'"
    mix_answer = llm(prompt)
    print(mix_answer)
    return mix_answer


class ChatSetting:
    def __init__(self, fields, age, career, company, image):
        self.fields = fields
        self.age = age
        self.career = career
        self.company = company
        self.image = image


class ChatReceive:
    def __init__(self, gptQuestion, gptMessage, feedbackMessage, mixMessage):
        self.gptQuestion = gptQuestion
        self.gptMessage = gptMessage
        self.feedbackMessage = feedbackMessage
        self.mixMessage = mixMessage


class ChatFirstMessage:
    def __init__(self, first_message):
        self.firstMessage = first_message


class ChatSend:
    def __init__(self, gptQuestion, userMessage):
        self.gptQuestion = gptQuestion
        self.userMessage = userMessage


app = Flask(__name__)           # default setting
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app, origins='http://localhost:3000')


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/ML-server', methods=['POST'])
def gptMessage():
    global question_list

    request_data = request.get_data(as_text=True)
    chat_send_dict = json.loads(request_data)
    gptQuestion = chat_send_dict['gptQuestion']
    userMessage = chat_send_dict['userMessage']

    # chatSend = ChatSend(**request.json)
    # gptQuestion = chatSend.gptQuestion
    # userMessage = chatSend.userMessage

    feedback = Feedback(gptQuestion, userMessage)
    ai_answer = AI_answer(gptQuestion)
    mix_answer = Mix_answer(feedback, userMessage)

    ck = should_ask_followup(gptQuestion, userMessage)
    if ck == True:
        next_question = AskFollowUpQuestions(gptQuestion, userMessage)
        question_list.remove(next_question)
    else:
        next_question = AskTechnicalQuestions(question_list)

    chatReceive = ChatReceive(
        next_question, ai_answer, feedback, mix_answer)
    return jsonify(chatReceive.__dict__)


@app.route('/setting', methods=['POST'])
def first_message():
    global question_list
    chat_setting = request.files.get('chatSetting').read().decode('utf-8')
    file = request.files.get('image').read()
    chat_setting_dict = json.loads(chat_setting)
    chat_setting = ChatSetting(chat_setting_dict['fields'], chat_setting_dict['age'],
                               chat_setting_dict['career'], chat_setting_dict['company'], file)
    file = chat_setting.image

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file)
        tmp.flush()
        pdf_text = extract_text_from_pdf(tmp.name)
    soft_pdf_text = text_to_data(pdf_text)
    ques = tec_question_generate(soft_pdf_text)
    question_list = question_list + ques.split('\n')
    input_features = [chat_setting.fields, chat_setting.age,
                      chat_setting.career, chat_setting.company]
    input_question = input_question_generate(*input_features)
    question_list = question_list + input_question.split('\n')
    first_message = fir_qt(question_list)
    setReceive = ChatFirstMessage(first_message)
    return json.dumps(setReceive.__dict__)


if __name__ == "__main__":                              # default setting
    app.run(host="0.0.0.0", port="8000", debug=True)
