import os
import time
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class StreamingCallback(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.tokens = []
        self.start_time = None

    def on_llm_start(self, *args, **kwargs):
        self.start_time = time.time()
        self.tokens = []
        print("\nLLM Started generating response...", flush=True)

    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens.append(token)
        print(token, end="", flush=True)

    def on_llm_end(self, *args, **kwargs):
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"\nLLM finished generating response in {duration:.2f} seconds", flush=True)


def main():
    question = input("Enter your query: ")
    
    callbacks = [StreamingCallback()]
    
    MODEL_PATH = os.path.join('.', "models/llama3.1-8b-instruct.Q4_0_arm.gguf")
    
    model = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.1,
        max_tokens=1024,
        n_batch=2048,
        callbacks=callbacks,
        n_ctx=10000,
        n_threads=64,
        n_threads_batch=64
    )
        
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>   
    Question: {question}
    Answer: <|eot_id|>"""
    
    prompt = PromptTemplate(template=template, input_variables=['question'])
    chain = RunnablePassthrough().assign(question=lambda x: x) | prompt | model | StrOutputParser()
    
    response = chain.invoke({"question": question})
    
    print(response)
    
if __name__=="__main__":
    main()