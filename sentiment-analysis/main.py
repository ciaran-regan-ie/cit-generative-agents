import sys
sys.path.append('.')

import pandas as pd
import os
import json
import csv
import asyncio
import aiohttp
from aiohttp import ClientSession
from dotenv import load_dotenv

load_dotenv(".env")

API_KEY = os.getenv("OPENAI_API_KEY")

async def get_sentiment(comment: str, session: ClientSession) -> dict:
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
        }

        payload = {
            "model": "gpt-3.5-turbo",
            "response_format": { "type": "json_object" },
            "messages": [
                {"role": "system", "content": "Determine the overall sentiment of the text as either positive, negative, or neutral, and rate the intensity of this sentiment on a scale from 0 (extremely negative) to 10 (extremely positive). Then, identify which basic emotion—anger, fear, enjoyment, sadness, disgust, or surprise—is most prominently reflected in the text. Based on the words and phrases used, along with the emotional tone, explain why you chose this sentiment rating and emotion. The response should be in JSON format with values for sentiment, intensity, emotion, and explanation."},
                {"role": "user", "content": comment},
            ],
        }

        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as response:
            if response.status != 200:
                return {}

            response_data = await response.json()

            # Extract and parse the response content safely
            if 'choices' not in response_data:
                return {}

            content = response_data['choices'][0]['message']['content'].strip()
            sentiment_data = json.loads(content)
            return sentiment_data

    except (KeyError, json.JSONDecodeError) as e:
        print(f"An error occurred while parsing the sentiment data: {str(e)}")
        return {}

async def process_comments(input_file: str, output_file: str):
    # Open the input file and read rows into a DataFrame
    df = pd.read_csv(input_file, delimiter=';', encoding='ISO-8859-1')
    df.columns = ['Author', 'Content', 'NumberOfReplies', 'NumberofThumbsUp', 'IsReply', 'Extra']
    df = df.drop(columns=["Extra"], errors="ignore")

    async with ClientSession() as session:
        # Use a list comprehension to create a list of tasks
        tasks = [
            get_sentiment(row['Content'], session) for idx, row in df.iterrows()
        ]

        # Wait for all tasks to complete and get their results
        sentiment_results = await asyncio.gather(*tasks)

    # Writing to the output CSV in synchronous mode
    with open(output_file, 'w', newline='', encoding='ISO-8859-1', errors="replace") as out_file:
        writer = csv.writer(out_file, delimiter=';')
        # Writing headers for the output file
        writer.writerow(['Author', 'Content', 'NumberOfReplies', 'NumberofThumbsUp', 'IsReply', 'Sentiment', 'Intensity', 'Emotion', 'Explanation'])

        # Iterate over each row in the DataFrame and append sentiment data
        for idx, row in enumerate(df.iterrows()):
            author, row_data = row
            sentiment_data = sentiment_results[idx]

            if not sentiment_data:
                continue  # Skip this row if no sentiment data found

            # Create a list to represent the full row with sentiment data
            output_row = [
                row_data['Author'],
                row_data['Content'],
                row_data['NumberOfReplies'],
                row_data['NumberofThumbsUp'],
                row_data['IsReply'],
                sentiment_data.get('sentiment'),
                sentiment_data.get('intensity'),
                sentiment_data.get('emotion'),
                sentiment_data.get('explanation'),
            ]

            # Write the row directly to the output CSV file
            writer.writerow(output_row)

if __name__ == '__main__':
    input_file = 'sentiment-analysis/data/comments.csv'
    output_file = 'sentiment-analysis/data/comments_with_sentiment.csv'

    asyncio.run(process_comments(input_file, output_file))
