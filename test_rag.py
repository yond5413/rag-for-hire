from core.generator import Generator
from core.retriever import Retriever
from langchain_community.document_loaders import PyPDFLoader

def main():
    loader = PyPDFLoader("./data/Yonathan_Daniel_Resume.pdf")
    pages = loader.load_and_split()
    ##################
    retriever = Retriever()
    generator = Generator()
    ####################
    retriever.add_documents(pages)
    job_desc = "We are looking for a software engineer with experience in AWS"
    chunks = retriever.search(job_desc, k=3)
    print("CHUNK RESULTS:")
    for i,ch in enumerate(chunks):
        print(f"Chunk {i+1}: {ch.page_content[:200]}...")## Print first 200 characters of each chunk
    ###################
    print("GENERATING COVER LETTER...")
    letter = generator.generate_cover_letter(job_desc, [ch.page_content for ch in chunks])
    print(letter)
    ####################

if __name__ == '__main__':
    ##TESTING RAG PIPELINE 
    main()