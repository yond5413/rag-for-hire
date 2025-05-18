import json
import re ## regex lib
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PyPDF2 import PdfReader
import pdfplumber 



def get_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text
def pdf_to_json(pdf_path: str,out_path: str)-> dict: 
    #ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    #ner = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english", grouped_entities=True)
    #ner = pipeline("ner", model="manishiitg/resume-ner", grouped_entities=True)
    #ner =pipeline("ner", model= "xlm-roberta-large-finetuned-conll03-english", grouped_entities=True)
    ner =pipeline("ner", model= "Omdena/bert-base-uncased-resume-ne", aggregation_strategy="simple")
    ### pdf plumber is good for extracting cv info
    text = get_text(pdf_path)
    ner_results = ner(text)
    #####################
    structured_data = {}
    for entity in ner_results:
        label = entity["entity_group"]
        value = entity["word"].strip()

        if label not in structured_data:
            structured_data[label] = [value]
        elif value not in structured_data[label]:
            structured_data[label].append(value)
    if out_path:
        with open(out_path,"w") as f:
            json.dump(structured_data,f,indent=2)
        print(f"file was save to {out_path}")
    return structured_data,text

def test(pdf_path):
    #pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    text = get_text(pdf_path)
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")
    prompt = f"""
    Extract the following info from the resume below and return as JSON:
    - Name
    - Education (school, degree, year)
    - Experience (company, title, duration)
    - Skills (languages, frameworks, tools)
    - Certifications (if applicable otherwise ommit)
    Resume:
    \"\"\"
    {text}
    \"\"\"
    """
    output = pipe(prompt, max_new_tokens=256, do_sample=False)
    print(output[0]["generated_text"])
    #output = pipe(prompt, max_length=512)[0]['generated_text']
    #print(output)
    
if __name__ =="__main__":
    input_path = '/workspaces/rag-for-hire/data/Yonathan_Daniel_Resume.pdf'
    out_path = '/workspaces/rag-for-hire/data/resume.json'
    #data,text = pdf_to_json(input_path,out_path)
    #print(data)
    #print(text)
    #print(len(text))
    #print(type(text))
    test(input_path)