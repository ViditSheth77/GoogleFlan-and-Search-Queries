from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class HuggingFaceService:
    def __init__(self, repo_id, temp=0.5, max_length=64, token=None):
        self.repo_id = repo_id
        self.temp = temp
        self.max_length = max_length
        self.token = token

    @property
    def llm(self):
        return HuggingFaceHub(
            repo_id=self.repo_id,
            huggingfacehub_api_token=self.token,
            model_kwargs={"temperature": self.temp, "max_length": self.max_length}
        )

    @property
    def system_prompt(self):
        template = "You are a helpful and accurate assistant. Please answer the following question: {query}."
        return PromptTemplate.from_template(template)

    def generate(self, query):
        llm_chain = LLMChain(prompt=self.system_prompt, llm=self.llm)
        return llm_chain({"query": query})

def get_user_queries():
    queries = []
    print("Enter your queries (type 'done' when finished):")
    while True:
        query = input("> ")
        if query.lower() == "done":
            break
        queries.append(query)
    return queries

HF_TOKEN = input("Please input your HUGGING_FACE_TOKEN: ")

hf = HuggingFaceService(repo_id="google/flan-t5-xxl", temp=0.5, max_length=64, token=HF_TOKEN)

user_queries = get_user_queries()

for query in user_queries:
    print("---------")
    answer = hf.generate(query)
    print(answer)
    # print(hf.generate(query))
