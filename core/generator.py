import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from typing import List, Dict, Any

class Generator:
    def __init__(self,model_id: str = "HuggingFaceH4/zephyr-7b-beta"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True, ## Load model in 4-bit quantization for efficiency
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=512
        )
    def generate_cover_letter(self,job_description: str, resume: str) -> str:
        """
        Generate a cover letter based on the job description and resume.
        """
        prompt = self.build_prompt(job_description, resume)
        return self.generate_text(prompt)
        
    def build_prompt(self,job_description: str, resume_chunks: List[str]) -> str:
        """
        Build a prompt for the model based on the job description and resume.
        """
        chunks_str = "\n".join(f"- {chunk}" for chunk in resume_chunks)
        return f"""<|system|>
        You are a professional resume writer. Generate a tailored cover letter using:
        - The job description requirements
        - Relevant resume sections

        Instructions:
        1. Match 3-4 key skills from the job description.
        2. Use professional but concise paragraphs.
        3. Highlight relevant experience and achievements.
        <|user|>
        Job: {job_description}

        Resume Sections:
        {chunks_str}
        <|assistant|>
        """
    def generate_text(self,prompt: str) -> str:
        """
        Generate text based on the prompt.
        """
        outputs = self.pipe(
            prompt,
            temperature=0.7, # Control randomness
            top_k = 50, # top50 moslt likely tokens
            top_p = 0.95, ## Use nucleus sampling for diversity
            repetition_penalty = 1.2 ## reduces redundancy
        )
        return outputs[0]['generated_text'].replace(prompt, "") 
    
    def stream(self,prompt: str):
        """
        Stream the generated text.
        """
        for chunk in self.pipe(
            prompt,
            temperature=0.7,
            stream=True,
            max_new_tokens=512,
        ):
            yield chunk['generated_text'].replace(prompt, "")