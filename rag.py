import os
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField


# Configure paths
BASE_PATH = "."
VECTOR_DIR = os.path.join(BASE_PATH, "vector")
MODEL_PATH = os.path.join(BASE_PATH, "models/llama3.1-8b-instruct.Q4_0_arm.gguf")

# Token Streaming
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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs).replace("Context:", "").strip()

def main():
    try:        
        # Initialize LLM
        callbacks = [StreamingCallback()]
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

        # Create chain
        embedding = HuggingFaceEmbeddings(model_name="thenlper/gte-base")
        vectorstore = FAISS.load_local(VECTOR_DIR, embedding, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever().configurable_fields(
            search_kwargs=ConfigurableField(id="search_kwargs")
        ).with_config({"search_kwargs": {"k": 5}})

        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful assistant. Use the following context to answer the question.
        Context: {context}
        Question: {question}
        Answer: <|eot_id|>"""

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        while True:
            question = input("Enter your prompt (Press q to quit): ")
            
            if question == 'q':
                break
            
            # Generate response
            response = chain.invoke(question)
            print(response)
        
    except Exception as e:
        print(f"Error processing query: {e}")


if __name__ == '__main__':
    main()