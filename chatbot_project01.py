import requests
import json
import numpy as np
from flask import Flask, request, jsonify
from neo4j import GraphDatabase
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from sentence_transformers import SentenceTransformer, util
import faiss

# ตั้งค่าโมเดล SentenceTransformer สำหรับความคล้ายคลึงของข้อความ
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# เชื่อมต่อ Neo4j
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "poramest")

def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]
    driver.close()

# ดึงข้อมูลข้อความจากฐานข้อมูล Neo4j
cypher_query = '''
MATCH (n:Greeting) RETURN n.name as name, n.msg_reply as reply;
'''
greeting_corpus = []
results = run_query(cypher_query)
for record in results:
    greeting_corpus.append(record['name'])

greeting_corpus = list(set(greeting_corpus))  # เอาข้อความมาใส่ใน corpus

# ฟังก์ชันคำนวณความคล้ายของข้อความ
def compute_similar(corpus, sentence):
    a_vec = model.encode([corpus], convert_to_tensor=True, normalize_embeddings=True)
    b_vec = model.encode([sentence], convert_to_tensor=True, normalize_embeddings=True)
    similarities = util.cos_sim(a_vec, b_vec)
    return similarities

# ค้นหาข้อความตอบกลับใน Neo4j
def neo4j_search(neo_query):
    results = run_query(neo_query)
    for record in results:
        response_msg = record['reply']
    return response_msg

# ฟังก์ชันคำนวณและหาข้อความตอบกลับจาก Neo4j
def compute_response(sentence):
    greeting_vec = model.encode(greeting_corpus, convert_to_tensor=True, normalize_embeddings=True)
    ask_vec = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)

    # Compute cosine similarities
    greeting_scores = util.cos_sim(greeting_vec, ask_vec)
    greeting_scores_list = greeting_scores.tolist()
    greeting_np = np.array(greeting_scores_list)

    max_greeting_score = np.argmax(greeting_np)
    Match_greeting = greeting_corpus[max_greeting_score]

    # ตรวจสอบคะแนนความเหมือน หากสูงกว่า 0.5 ให้ดึงข้อความตอบกลับจาก Neo4j
    if greeting_np[np.argmax(greeting_np)] > 0.5:
        My_cypher = f"MATCH (n:Greeting) WHERE n.name ='{Match_greeting}' RETURN n.msg_reply AS reply"
        my_msg = neo4j_search(My_cypher)
        return my_msg
    else:
        # ถ้าหาความเหมือนไม่เจอ ให้ใช้โมเดลจาก Ollama สร้างข้อความ
        return ollama_response(sentence)

# ฟังก์ชันเรียก Ollama API เพื่อตอบกลับ
def ollama_response(prompt):
    OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Adjust URL if necessary
    headers = {
        "Content-Type": "application/json"
    }

    # Prepare the request payload for the TinyLLaMA model
    payload = {
        "model": "supachai/llama-3-typhoon-v1.5",  # Assuming this model is available
        "prompt": prompt+ "ตอบสั้นๆไม่เกิน 20 คำ ตอบเป็นภาษาไทยเท่านั้น ผู้ตอบคือผู้เชี่ยวชาญการเลี้ยงโค" ,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            response_data = response.json()
            decoded_text = response_data.get("response", "No response key found")
            return "Ollama ช่วยตอบ"+decoded_text
        else:
            return f"Failed to get a response from Ollama API: {response.status_code}"
    except requests.RequestException as e:
        return f"Error occurred while contacting Ollama API: {e}"

# สร้าง Flask app
app = Flask(__name__)

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)  # รับข้อมูลจาก Line API
    try:
        json_data = json.loads(body)  # แปลงข้อมูลที่รับมาเป็น JSON
        access_token = 'NdsLwrTY89buAJL1Sjh+6977nq9pBkLZzI3XhqvIHP5JilfjhCG1gZjkNBhso4d7D7B9/XYvvrDkV/3vX5onOStkuj+ICKByGLIGcsHlyMGbIAal6r65J+4+G+b3G7SYZwx5oG2HrDZILpbvgprCxAdB04t89/1O/w1cDnyilFU='
        secret = '9b552a1d011668cc44d6159fda23a3a7'
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)

        # ข้อความที่ได้รับจากผู้ใช้
        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']

        # คำนวณและค้นหาข้อความตอบกลับ
        response_msg = compute_response(msg)

        # ส่งข้อความตอบกลับไปยัง Line
        line_bot_api.reply_message(tk, TextSendMessage(text=response_msg))
        print(msg, tk)
    except Exception as e:
        print(f"Error: {e}")
        print(body)  # ในกรณีที่เกิดข้อผิดพลาด

    return 'OK'

if __name__ == '__main__':
    app.run(port=5000)
