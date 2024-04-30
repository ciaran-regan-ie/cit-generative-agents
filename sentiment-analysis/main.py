import sys
sys.path.append('.')

import pandas as pd
import openai
import dotenv
import os
import json

dotenv.load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()

def get_sentiment(comment: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Determine the overall sentiment of the text as either positive, negative, or neutral, and rate the intensity of this sentiment on a scale from 0 (extremely negative) to 10 (extremely positive). Then, identify which basic emotion—anger, fear, enjoyment, sadness, disgust, or surprise—is most prominently reflected in the text. Based on the words and phrases used, along with the emotional tone, explain why you chose this sentiment rating and emotion. The response should be in JSON format with values for sentiment, intensity, emotion, and explanation."},
            {"role": "user", "content": comment}
        ]
    )

    # Parsing the response into a dictionary
    response_data = response.choices[0].message.content.strip()

    # Converting the JSON string into a Python dictionary
    sentiment_data = json.loads(response_data)

    return sentiment_data


def process_comments(file: str):
    # Reading the comments.csv file into a DataFrame
    df = pd.read_csv(file, delimiter=';', encoding='ISO-8859-1')
    df.columns = ['Author', 'Content', 'NumberOfReplies', 'NumberofThumbsUp', 'IsReply', 'Extra']

    # Remove unnecessary columns
    df = df.drop(columns=["Extra"], errors="ignore")

    # Apply get_sentiment and expand the returned dictionaries into separate columns
    sentiment_data = df['Content'].apply(get_sentiment)
    sentiment_df = pd.DataFrame(sentiment_data.tolist())

    # Concatenate the new columns to the original DataFrame
    df = pd.concat([df, sentiment_df], axis=1)

    # Save the updated DataFrame to a new CSV file
    df.to_csv('sentiment-analysis/data/comments_with_sentiment.csv', index=False, sep=';')

if __name__ == '__main__':
    file = 'sentiment-analysis/data/comments.csv'
    process_comments(file)
