import json
import os
import re
from llm_api import GptApi
from llm_api import LlmApi

config_path = "./config.json"
question_list_path = "./question_list.json"

input_path = "./question_list.json"
temporary_path = "./temp.json"
output_path = "./question_evaluation_list.json"

with open(config_path, "r") as json_file:
    config = json.load(json_file)

# Checking the fields in the config.json file ------------------------------------------------------------------
if "format" not in config:
   raise Exception("Missing required 'format' field in config.json: expected a boolean (true or false).")
if "language" not in config:
   raise Exception("Missing required 'language' field in config.json: expected a boolean (true or false).")
if "grammar" not in config:
   raise Exception("Missing required 'grammar' field in config.json: expected a boolean (true or false).")
if "relevance" not in config:
   raise Exception("Missing required 'relevance' field in config.json: expected a boolean (true or false).")
if "multi-hop" not in config:
   raise Exception("Missing required 'multi-hop' field in config.json: expected a boolean (true or false).")
if type(config["format"]) is not bool:
    raise Exception(f"Invalid 'format' field in config.json: expected a boolean (true or false), but got {config['format']} instead.")
if type(config["language"]) is not bool:
    raise Exception(f"Invalid 'format' field in config.json: expected a boolean (true or false), but got {config['format']} instead.")
if type(config["grammar"]) is not bool:
    raise Exception(f"Invalid 'format' field in config.json: expected a boolean (true or false), but got {config['format']} instead.")
if type(config["relevance"]) is not bool:
    raise Exception(f"Invalid 'format' field in config.json: expected a boolean (true or false), but got {config['format']} instead.")
if type(config["options"]) is not bool:
    raise Exception(f"Invalid 'format' field in config.json: expected a boolean (true or false), but got {config['format']} instead.")
if type(config["multi-hop"]) is not bool:
    raise Exception(f"Invalid 'format' field in config.json: expected a boolean (true or false), but got {config['format']} instead.")

language_prompt = """\
Considering the following context and the question based on this context: 
<context>
{context}
</context>
<question>
{question}
</question>
Is the entire question written in English?
Respond only 'Yes' or 'No'.
"""

grammar_prompt = """\
Considering these specific terms and acronyms within the context: 
{glossary} 
Considering the following context and the question based on this context: 
<context>{context}</context>
<question>{question}</question>
Is there any grammatical error in the question? Answer only ’Yes’ or ’No’.
"""

relevance_prompt = """\
Considering the following context and the question
based on this context:
<context>{context}</context>
<question>{question}</question>
Please grade the relevance of this question relative to the context with a score from 0 to 10. 
A question is considered relevant if it pertains to the core themes and
concepts discussed in the text, engages with the
important ideas and content presented, and does
not focus on memorizing specific details such as
specific dates, or precise wording. Answer only
with the number corresponding to the score.
"""

options_prompt = """\
Considering the following context and the question based on this context:
<context>{context}</context>
<question>{question}</question>
<answer>{answer}</answer>
Evaluate the answer to this question with a score
from 0 to 10, with 0 being completely wrong and
10 being completely correct. Answer only with
the number corresponding to the score.
"""

llmApi = GptApi()
model="gpt-4o-mini"

with open(question_list_path, "r") as json_file:
    question_list = json.load(json_file)

output = {}
output["config"] = config
output["evaluated_questions"] = []
last_evaluation = {}

if os.path.exists(temporary_path):
    print("Temporary file detected. Previous completions have been restored, and evaluation will proceed.")
    with open(temporary_path, "r") as file:
        temp_json = file.read()
        temp_json = "[" + temp_json.rstrip(",\n") + "]"
        temp_question_list = json.loads(temp_json)
        for qe in temp_question_list:
            if qe["evaluation"]["completed"] == True:
                output["evaluated_questions"].append(qe)
        if temp_question_list[-1]["evaluation"]["completed"] == False:
            last_evaluation = temp_question_list[-1]["evaluation"]

print("Begginning evaluation")
print(f"A temporary file will be created to recover the results in case of any issues.")
for i, question_data in enumerate(question_list["questions"]):
    print(f"Evaluation {i}:", end=" ")

    if i < len(output["evaluated_questions"]):
        print("recovered from temporary file.")
        continue

    try:
        if config["language"] is True and "language" not in last_evaluation:
            prompt = language_prompt.format(context=question_data["context"], question=question_data)
            completion = llmApi.call_api(prompt=prompt, model=model)
            if (completion == "Yes") :
                last_evaluation["language"] = True
            elif (completion == "No"):
                last_evaluation["language"] = False
            else:
                raise Exception(f"The LLM should have responded with Yes or No, but responded with {completion}")
            print(f"language={last_evaluation['language']}", end=", ", flush=True)

        if config["grammar"] is True and "grammar" not in last_evaluation:
            prompt = grammar_prompt.format(glossary= "", context=question_data["context"], question=question_data)
            completion = llmApi.call_api(prompt=prompt, model=model)
            if 0==1:
                raise Exception(f"Test")
            if (completion == "Yes") :
                last_evaluation["grammar"] = True
            elif (completion == "No"):
                last_evaluation["grammar"] = False
            else:
                raise Exception(f"The LLM should have responded with Yes or No, but responded with {completion}")
            print(f"grammar={last_evaluation['grammar']}", end=", ", flush=True)

        if config["relevance"] is True and "relevance" not in last_evaluation:
            prompt = relevance_prompt.format(context=question_data["context"], question=question_data)
            completion = llmApi.call_api(prompt=prompt, model=model)
            if 0==1:
                raise Exception(f"Test")
            if (completion.isdigit() and 0 <= int(completion) <= 10):
                last_evaluation["relevance"] = True
            else:
                raise Exception(f"The LLM should have responded with a grade between 0 and 10, but responded with {completion}")
            print(f"relevance={last_evaluation['relevance']}", end=", ", flush=True)

        if config["options"] is True:
            print("options=[", end="", flush=True)
            for j, option in enumerate(question_data["options"]):
                if "options" in last_evaluation:
                    if j < len(last_evaluation["options"]):
                        print("R", end=" ")
                        continue
                else:
                    last_evaluation["options"] = []
                prompt = options_prompt.format(
                    context=question_data["context"], 
                    question=question_data["question"], 
                    answer=option
                )
                completion = llmApi.call_api(prompt=prompt, model=model)
                if (completion.isdigit() and 0 <= int(completion) <= 10):
                    last_evaluation["options"].append(int(completion))
                else:
                    raise Exception(f"The LLM should have responded with a grade between 0 and 10, but responded with {completion}")
                if j < len(question_data["options"]) - 1:
                    print(f"{last_evaluation['options'][j]}", end=",", flush=True)
                else:
                    print(f"{last_evaluation['options'][j]}", end="],", flush=True)

        last_evaluation["completed"] = True
    except Exception as e:
        print(e)
        last_evaluation["completed"] = False
        raise e
    finally:
        evaluated_question = {
            "question": question_data,
            "evaluation": last_evaluation
        }

        output["evaluated_questions"].append(evaluated_question)

        with open(temporary_path, "a") as outfile:
            json_evaluated_question = json.dumps(evaluated_question, indent="\t")
            outfile.write(json_evaluated_question + ",\n")
            outfile.flush()

        last_evaluation = {}
        print(" Evaluation completed.")

    #raise Exception()


with open(output_path, "w") as output_file:
    json_output = json.dumps(output, indent="\t")
    output_file.write(json_output)
    output_file.flush()

if os.path.exists(temporary_path):
    os.remove(temporary_path)
    print(f"Temporary file deleted")
