# hf = HuggingFaceService(repo_id="google/flan-t5-xxl", temp=0.5, max_length=64)

# for query in QUERIES:
#     print("---------")
#     # systm_prompt = f"You are helpful assistant with general purpose knowledge. Please asnwer the following question in one sentence: {query}"
#     print(hf.generate(query))

from transformers import pipeline

def main():
    # Load the Flan T5 XXL model via pipeline
    qa_pipeline = pipeline("question-answering", model="google/flan-t5-xxl")

    # Ask a question
    question = input("Ask a question: ")

    # Provide the context (if needed)
    context = "Your context here"  # Replace with your context if needed

    # Call the pipeline to get the answer
    answer = qa_pipeline(question=question, context=context)

    # Print the answer
    print("Answer:", answer['answer'])

if __name__ == "__main__":
    main()
