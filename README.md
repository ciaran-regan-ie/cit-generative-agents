# CIT Generative Agents Project

## Overview
This project implements sentiment analysis for YouTube comments in CSV format.

## Requirements
- Python
- OpenAI API Key

## Installation and Setup
1. Clone the repository
```
git clone https://github.com/ciaran-regan-ie/cit-generative-agents.git
```
2. Traverse to the root directory and install the required packages
```
cd cit-generative-agents
```
```
pip install -r requirements.txt
```
3. Create a `.env` file in the root directory and add your OpenAI API key
```
OPENAI_API_KEY=<your_api_key>
```
4. Ensure you have a CSV file with YouTube comments in `sentiment-analysis/data`

## Usage
Run the following command to perform sentiment analysis on the YouTube comments
```
python3 sentiment-analysis/main.py
```
The results will be saved in `sentiment-analysis/data` as `comments_with_sentiment.csv`
