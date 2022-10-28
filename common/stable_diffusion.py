import requests
from time import sleep
endpoint = "https://api.together.xyz"

def submit_job(model_owner, model, prompt, num_returns, args):
    data = {
        "num_returns": num_returns,
        "source": "discord_bot",
        "prompt": prompt,
        "model": model,
        "model_owner": model_owner,
        "owner": "together",
        "tags": [
            "discord_bot"
        ],
        "seed": 1,
        "args": args,
        "output": {},
    }
    res = requests.put(endpoint+"/jobs/jobs", json=data).json()
    return res

def fetch_results(job_id):
    returned_payload = None
    while True:
        sleep(1)
        print("waiting...")
        res = requests.get(endpoint+"/jobs/job/"+job_id).json()
        returned_payload = res['output']
        if res['status'] == "finished" or res['status'] == "failed":
            break
    print(returned_payload)

res = submit_job("together", "StableDiffusion", "a lion is lying on the grass", 1, {})
print(res)
print("fetching results...")
fetch_results(res['id'])