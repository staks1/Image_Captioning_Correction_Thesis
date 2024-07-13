import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    # cache_dir="/data/yash/base_models",
    device_map='auto'
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", 
                                          # cache_dir="/data/yash/base_models"
                                         )

def get_llama2_chat_reponse(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature= 0.00001)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response



# zero shot prompt 
prompt ="[INST] <<SYS>>\
You are an engineer assistant specialized in industry, factories, industry safety.You will be given 1 sentence. The sentence describes a problem in the industry domain. The sentence may not be complete or may be very brief and will not always be grammatically or syntactically accurate . Your goal is to understand the problem from the sentence and generate 3 sentences that could potentially solve the problem described. The sentences you provide must be solutions to the problem and be grammatically and syntactically accurate.Do not repeat the sentence i give you. Number each provided sentence.\
<</SYS>>\
{Suggest 3 solutions to correct the problem in the sentence : fire extinguisher blocking door.}[/INST]"

# 1 shot prompt 
prompt2 ="<<SYS>>\
You are an engineer assistant specialized in industry, factories, industry safety.You will be given 1 sentence. The sentence describes a problem in the industry domain. The sentence may not be complete or may be very brief and will not always be grammatically or syntactically accurate . Your goal is to understand the problem from the sentence and generate 3 sentences that could potentially solve the problem described. The sentences you provide must be solutions to the problem and be grammatically and syntactically accurate.Do not repeat the prompt i give you. Number each provided sentence.\
<</SYS>>\
[INST]{Suggest 3 solutions to correct the problem in the sentence : fire extinguisher blocking door.}[/INST]\
\Solution :\
1. Relocate the fire extinguisher: Consider repositioning the fire extinguisher to a location that does not obstruct the door, such as a nearby wall or corner. This would ensure that the fire extinguisher is easily accessible without blocking any doors.\
2. Install a sliding fire extinguisher: Design a fire extinguisher that can be easily slid out of the way when not in use, allowing the door to open freely. This could involve using a lightweight and compact design, or incorporating a sliding mechanism into the fire extinguisher's casing.\
3. Use a retractable fire extinguisher: Develop a fire extinguisher that can be retracted into a housing when not in use, allowing the door to open without obstruction. This could involve using a spring-loaded or motorized mechanism to retract the fire extinguisher into its housing.\
[INST]{Suggest 3 solutions to correct the problem in the sentence : person on ladder unstable , broke ladder.\
Solution :}[/INST]"

# few shot prompt 

# run model 
solutions = get_llama2_chat_reponse(prompt2, max_new_tokens=1000)
x = ''.join([x for x in solutions if x!=prompt2])

print(x)
