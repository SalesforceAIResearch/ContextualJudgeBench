import time
import sys
import torch
from openai import OpenAI


MAX_CONNECTION_TRIES = 120 # number of times to try connecting judge endpoints
DELAY_TIME = 10 # sleep for 10s; 120 x 10 = 20 minutes before giving up

def wait_until_server_ready(base_url, max_connection_tries=None, delay_time=None):
    if max_connection_tries is None:
        max_connection_tries = MAX_CONNECTION_TRIES
    if delay_time is None:
        delay_time = DELAY_TIME
    for i in range(max_connection_tries):
        try:
            client = OpenAI(base_url=base_url)
            print(client.models.list())
            return
        except:
            print(f'Waiting for server to be ready... {i+1} / {max_connection_tries}')
            time.sleep(delay_time)
    raise ConnectionError(f'Cannot connect to server after {max_connection_tries} tries.')


class VllmEndpoint:
    def __init__(self, model_name, model_url, sampling_params, api_key='sample-api-key'):
        self.model_name = model_name
        self.model_url = model_url
        self.sampling_params = sampling_params
        self.api_key = api_key

        # Assume that together AI models are up
        if 'together.xyz' not in model_url:
            wait_until_server_ready(model_url)


    def generate(self, messages):

        client = OpenAI(base_url=self.model_url, api_key=self.api_key)
        received = False
        cnt_try = 0
        while not received and cnt_try < 3:
            try:
                sp = self.sampling_params
                response = client.chat.completions.create(
                    model=self.model_name, 
                    messages=messages, 
                    temperature=sp.temperature, 
                    max_tokens=sp.max_tokens, 
                    top_p=sp.top_p, 
                    n=sp.n,
                )
                texts = [r.message.content for r in response.choices]
                received = True
            except Exception as e:
                texts = ["None"]
                error = sys.exc_info()[0]

                print(e)
                print("API error:", error)
                cnt_try += 1
                time.sleep(10)

        return texts
