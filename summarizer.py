from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

model_name = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)


def split_text(text, chunk_size=300):
    words = text.split()

    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def summarize_chunk(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        early_stopping=True,
        length_penalty=1.0,
        max_new_tokens=250
    )

    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )


def calculate_accuracy(original_text, summary_text):

    vectorizer = TfidfVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform(
        [original_text, summary_text]
    )

    score = cosine_similarity(
        vectors[0],
        vectors[1]
    )[0][0]

    return round(score * 100, 2)


def generate_summary(text):

    chunks = list(split_text(text))

    summaries = []

    for chunk in chunks:
        summaries.append(
            summarize_chunk(chunk)
        )

    final_summary = "\n\n".join(summaries)

    accuracy = calculate_accuracy(
        text,
        final_summary
    )

    return final_summary, accuracy