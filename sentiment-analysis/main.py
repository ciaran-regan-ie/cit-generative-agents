import sys
sys.path.append('.')

import pandas as pd
import openai
import dotenv
import os
dotenv.load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()

def get_sentiment(comment: str) -> str:
    """
    Get the sentiment of a comment using OpenAI's GPT-3.5-turbo model

    Args:
        comment (str): The comment to analyze
    """

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a highly intelligent AI trained to determine the sentiment of text as positive, negative, or neutral. Reply with only postive, negative, or neutral in English"},
            {"role": "user", "content": f"{comment}"}
        ],
        max_tokens=2
    )
    
    sentiment = response.choices[0].message.content.strip().lower()
    return sentiment

if __name__ == '__main__':

    # Reading the comments.csv file into a DataFrame
    df = pd.read_csv('sentiment-analysis/data/comments.csv', delimiter=';')
    df.columns = ['user', 'comments', 'extra']

    # Dropping the extra empty column as it's not needed
    df = df.drop('extra', axis=1)

    # Applying the get_sentiment function to the comments column
    df['sentiment'] = df['comments'].apply(get_sentiment)

    # save the updated DataFrame to a new csv file
    df.to_csv('sentiment-analysis/data/comments_with_sentiment.csv', index=False, sep=';')
