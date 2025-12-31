from datetime import datetime
import os
from openai import OpenAI
import json
import math

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def load_memory():
    try:
        with open("memory.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def save_memory(memory):
    with open("memory.json", "w") as file:
        json.dump(memory, file, indent=2)

def cosine_similarity(vec1, vec2):
    dot_product = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    return dot_product / (norm1 * norm2)

def retrieve_similar_entries(new_embedding, memory, new_text, new_date, top_n=3):
    similarities = []
    for entry in memory:
        if entry["text"] == new_text and entry["date"] == new_date:
            continue  # skip current entry
        score = cosine_similarity(new_embedding, entry["embedding"])
        similarities.append((score, entry))
    similarities.sort(reverse=True, key=lambda x: x[0])
    return [entry for _, entry in similarities[:top_n]]

def build_productivity_prompt(today_entry, similar_entries):
    prompt = (
        "You are a helpful assistant that reads a person's journal entries and rates "
        "their daily productivity on a scale from 1 (low) to 10 (high). Then provide a short piece of advice to improve productivity.\n\n"
        f"Today's entry:\n\"\"\"\n{today_entry}\n\"\"\"\n\n"
        "Here are some similar days:\n"
    )
    for i, entry in enumerate(similar_entries, 1):
        prompt += f"{i}. {entry['date']} | {entry['text']}\n"
    prompt += "\nPlease rate today's productivity and provide brief advice."
    return prompt

def get_productivity_judgement(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=150,
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    print("Journal program started")
    memory = load_memory()

    if memory:
        print("\nYour past entries:")
        for entry in memory:
            print(f"- {entry['date']} | {entry['text']}")
        print("\n")
    else:
        print("\nNo entries found.\n")

    entry_text = input("Write your journal entry: ")
    today = datetime.now().strftime("%Y-%m-%d")

    entry_embedding = get_embedding(entry_text)
    memory.append({"date": today, "text": entry_text, "embedding": entry_embedding})
    save_memory(memory)
    print("Entry saved with embedding.")

    top_similar = retrieve_similar_entries(entry_embedding, memory, entry_text, today)
    prompt = build_productivity_prompt(entry_text, top_similar)
    judgement = get_productivity_judgement(prompt)

    print("\nAI Productivity Judge:")
    print(judgement)
