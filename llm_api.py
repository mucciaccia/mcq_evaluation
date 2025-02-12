import requests

class LlmApi:

    def __init__(self):
        pass

    def call_api(self, prompt):
        pass

class GptApi(LlmApi):

    def get_api_key():
        file = open("api_key.txt", "r")
        api_key = file.readline().rstrip('\n')
        file.close()
        return api_key

    def call_gpt_api(self, prompt, model="gpt-4o", max_tokens=1000):
        api_key = GptApi.get_api_key()

        url = "https://api.openai.com/v1/chat/completions".format(model=model)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": f"{prompt}"}], 
            "temperature": 0.1
        }

        response = requests.post(url, headers=headers, json=data)
        return response.json()
    
    def call_api(self, prompt, model):
        response = self.call_gpt_api(prompt=prompt, model=model)
        if 'choices' in response:
            text = response['choices'][0]['message']['content']
        else:
            print(response)
            raise Exception("API error")
        return text