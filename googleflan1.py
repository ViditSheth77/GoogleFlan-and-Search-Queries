from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

def main():
    model_name = "google/flan-t5-xxl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    question = input("Ask a question: ")
    inputs = tokenizer(question, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end+1]))
    print("Answer:", answer)

if __name__ == "__main__":
    main()
