import os
import pandas as pd
import torch

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    MarianTokenizer,
    MarianMTModel
)


# ============================================================
# CREATE DATASET FOLDER
# ============================================================

os.makedirs("dataset", exist_ok=True)


# ============================================================
# INPUT ARTICLE
# ============================================================

article = """
Artificial Intelligence (AI) is becoming an important technology
in healthcare. It helps doctors and medical professionals analyze
large amounts of medical data quickly and accurately.

AI can be used for disease diagnosis, medical image analysis,
patient monitoring, drug discovery, and personalized treatment.
Machine learning algorithms can identify patterns in medical
images such as X-rays, CT scans, and MRI scans.

AI-powered systems can also help hospitals manage patient records,
predict health risks, and improve hospital operations. Chatbots
and virtual assistants can provide patients with basic health
information and appointment support.

However, the use of AI in healthcare also creates challenges.
Patient privacy, data security, algorithmic bias, and the need
for human supervision are important concerns.

Therefore, AI should be used responsibly in healthcare.
Doctors and healthcare professionals should work together with
AI systems to ensure that patients receive safe and effective
care.
"""


# ============================================================
# SAVE INPUT TEXT
# ============================================================

input_df = pd.DataFrame([
    {
        "Text": article
    }
])

input_df.to_csv(
    "dataset/input_text.csv",
    index=False
)

print("Input text saved successfully.")


# ============================================================
# 1. SENTIMENT ANALYSIS
# ============================================================

print("\n====================================")
print("1. SENTIMENT ANALYSIS")
print("====================================")


sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)


sentiment_result = sentiment_analyzer(article)


print("Sentiment Result:")
print(sentiment_result)


sentiment_df = pd.DataFrame(sentiment_result)

sentiment_df.to_csv(
    "dataset/sentiment_results.csv",
    index=False
)

print("Sentiment result saved.")


# ============================================================
# 2. TEXT SUMMARIZATION
# ============================================================

print("\n====================================")
print("2. TEXT SUMMARIZATION")
print("====================================")


summary_tokenizer = AutoTokenizer.from_pretrained(
    "sshleifer/distilbart-cnn-12-6"
)

summary_model = AutoModelForSeq2SeqLM.from_pretrained(
    "sshleifer/distilbart-cnn-12-6"
)


def generate_summary(
    text,
    max_length=100,
    min_length=30
):

    inputs = summary_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    summary_ids = summary_model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )

    summary = summary_tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


summary_text = generate_summary(
    article,
    max_length=100,
    min_length=30
)


print("\nGenerated Summary:")
print(summary_text)


summary_df = pd.DataFrame([
    {
        "Original_Text": article,
        "Summary": summary_text
    }
])


summary_df.to_csv(
    "dataset/summarized_text.csv",
    index=False
)

print("Summary saved.")


# ============================================================
# 3. TRANSLATION
# ============================================================

print("\n====================================")
print("3. TRANSLATION")
print("====================================")


def translate_text(text, model_name):

    # Marian models require MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained(
        model_name
    )

    model = MarianMTModel.from_pretrained(
        model_name
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    translated_ids = model.generate(
        **inputs,
        max_new_tokens=200
    )

    translated_text = tokenizer.decode(
        translated_ids[0],
        skip_special_tokens=True
    )

    return translated_text


# Use first 1000 characters for translation
translation_input = article[:1000]


# English → French
french_text = translate_text(
    translation_input,
    "Helsinki-NLP/opus-mt-en-fr"
)


# English → German
german_text = translate_text(
    translation_input,
    "Helsinki-NLP/opus-mt-en-de"
)


print("\nEnglish:")
print(translation_input)

print("\nFrench:")
print(french_text)

print("\nGerman:")
print(german_text)


translation_df = pd.DataFrame([
    {
        "English": translation_input,
        "French": french_text,
        "German": german_text
    }
])


translation_df.to_csv(
    "dataset/translated_text.csv",
    index=False
)

print("Translation results saved.")


# ============================================================
# 4. QUESTION ANSWERING
# ============================================================

print("\n====================================")
print("4. QUESTION ANSWERING")
print("====================================")


qa_model_name = "distilbert-base-cased-distilled-squad"


qa_tokenizer = AutoTokenizer.from_pretrained(
    qa_model_name
)

qa_model = AutoModelForQuestionAnswering.from_pretrained(
    qa_model_name
)


def answer_question(question, context):

    inputs = qa_tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = qa_model(
            **inputs
        )

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_index = torch.argmax(
        start_scores
    )

    end_index = torch.argmax(
        end_scores
    )

    # Make sure the end comes after the start
    if end_index < start_index:
        end_index = start_index

    answer_tokens = inputs[
        "input_ids"
    ][0][start_index:end_index + 1]

    answer = qa_tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True
    )

    start_score = torch.softmax(
        start_scores,
        dim=1
    )[0][start_index].item()

    end_score = torch.softmax(
        end_scores,
        dim=1
    )[0][end_index].item()

    score = (start_score + end_score) / 2

    return answer, score


questions = [
    "What is AI used for in healthcare?",
    "What can AI analyze in healthcare?",
    "What are the challenges of AI in healthcare?"
]


qa_results = []


for question in questions:

    answer, score = answer_question(
        question,
        article
    )

    print("\nQuestion:")
    print(question)

    print("Answer:")
    print(answer)

    print("Score:")
    print(score)

    qa_results.append(
        {
            "Question": question,
            "Answer": answer,
            "Score": score
        }
    )


qa_df = pd.DataFrame(
    qa_results
)


qa_df.to_csv(
    "dataset/qa_results.csv",
    index=False
)

print("\nQuestion-answering results saved.")


# ============================================================
# 5. SUMMARY COMPARISON
# ============================================================

print("\n====================================")
print("5. SUMMARY COMPARISON")
print("====================================")


summary_60 = generate_summary(
    article,
    max_length=60,
    min_length=20
)


summary_100 = generate_summary(
    article,
    max_length=100,
    min_length=30
)


summary_150 = generate_summary(
    article,
    max_length=150,
    min_length=40
)


print("\nShort Summary:")
print(summary_60)

print("\nMedium Summary:")
print(summary_100)

print("\nLong Summary:")
print(summary_150)


comparison_df = pd.DataFrame([
    {
        "Length": "Short",
        "Max_Length": 60,
        "Summary": summary_60
    },
    {
        "Length": "Medium",
        "Max_Length": 100,
        "Summary": summary_100
    },
    {
        "Length": "Long",
        "Max_Length": 150,
        "Summary": summary_150
    }
])


comparison_df.to_csv(
    "dataset/pipeline_comparison.csv",
    index=False
)

print("\nPipeline comparison saved.")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n====================================")
print("ALL TASKS COMPLETED SUCCESSFULLY")
print("====================================")

print("\nGenerated files:")

print("1. dataset/input_text.csv")
print("2. dataset/sentiment_results.csv")
print("3. dataset/summarized_text.csv")
print("4. dataset/translated_text.csv")
print("5. dataset/qa_results.csv")
print("6. dataset/pipeline_comparison.csv")

print("\nHugging Face assignment completed successfully.")