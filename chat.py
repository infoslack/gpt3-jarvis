import os
import re
import json
import config
import openai
import numpy as np
from numpy.linalg import norm
from time import time,sleep
from uuid import uuid4
import datetime
openai.api_key = config.OPENAI_API_KEY

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=2)

def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")

def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='utf-8', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    v = response['data'][0]['embedding']
    return v

def similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def get_memories(vector, logs, count):
    scores = []
    for log in logs:
        if vector == log['vector']:
            continue
        score = similarity(log['vector'], vector)
        log['score'] = score
        scores.append(log)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered

def load_conversation():
    files = [f for f in os.listdir('conversations') if '.json' in f]
    result = []
    for file in files:
        data = load_json('conversations/%s' % file)
        result.append(data)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)
    return ordered

def summarize_memories(memories):
    memories = sorted(memories, key=lambda d: d['time'], reverse=True)
    block = '\n\n'.join([mem['message'] for mem in memories])
    identifiers = [mem['uuid'] for mem in memories]
    timestamps = [mem['time'] for mem in memories]
    prompt = open_file('prompt_notes.txt').replace('<<INPUT>>', block)
    notes = gpt3_completion(prompt)
    vector = gpt3_embedding(block)
    info = {'notes': notes, 'uuids': identifiers, 'times': timestamps, 'uuid': str(uuid4()),
            'vector': vector, 'time': time()}
    filename = 'notes_%s.json' % time()
    save_json('local_notes/%s' % filename, info)
    return notes

def get_last_messages(conversation, limit):
    try:
        short = conversation[-limit:]
    except:
        short = conversation
    output = ''
    for i in short:
        output += '%s\n\n' % i['message']
    output = output.strip()
    return output

def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, top_p=1.0,
                    tokens=500, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'JARVIS:']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='utf-8', errors='ignore').decode()

    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            save_file('logs/%s' % filename, prompt + '\n\n----------\n\n' + text)
            return text
        except Exception as fail:
            retry += 1
            if retry >= max_retry:
                return "GPT-3 error: %s" % fail
            print('Error communicating with OpenAI:', fail)
            sleep(1)

if __name__ == '__main__':
    while True:
        a  = input('\n\nUSER: ')
        timestamp = time()
        vector = gpt3_embedding(a)
        timestring = timestamp_to_datetime(timestamp)
        message = '%s: %s - %s' % ('USER', timestring, a)
        info = {'speaker': 'USER', 'time': timestamp, 'vector': vector, 'message': message, 'uuid': str(uuid4()), 'timestring': timestring}
        filename = 'log_%s_USER.json' % timestamp
        save_json('conversations/%s' % filename, info)

        conversation = load_conversation()
        memories = get_memories(vector, conversation, 10)
        notes = summarize_memories(memories)
        recent = get_last_messages(conversation, 4)

        prompt = open_file('prompt_response.txt').replace('<<NOTES>>', notes).replace('<<CONVERSATION>>', recent)
        output = gpt3_completion(prompt)
        timestamp = time()
        vector = gpt3_embedding(output)
        timestring = timestamp_to_datetime(timestamp)
        message = '%s: %s - %s' % ('JARVIS', timestring, output)
        info = {'speaker': 'JARVIS', 'time': timestamp, 'vector': vector, 'message': message, 'uuid': str(uuid4()), 'timestring': timestring}
        filename = 'log_%s_JARVIS.json' % time()
        save_json('conversations/%s' % filename, info)

        print('\n\nJARVIS: %s' % output)
