import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Websites to scrape
websites = [
    "https://www.inquirer.net",
    "https://news.abs-cbn.com",
    "https://www.philstar.com",
    "https://www.pna.gov.ph",
    "https://www.rappler.com"
]

# 1. Web Scraping Functionality
import requests
from bs4 import BeautifulSoup

# Define a separate scraping function for each website since the HTML structure differs

def scrape_inquirer():
    url = "https://www.inquirer.net"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    articles = []
    # Look for specific HTML tags/classes for articles
    for item in soup.select('h2.article-title'):  # Inquirer headlines
        headline = item.text.strip()
        summary = item.find_next('p').text.strip() if item.find_next('p') else None
        articles.append({'headline': headline, 'summary': summary})
    
    return articles

def scrape_abs_cbn():
    url = "https://news.abs-cbn.com"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    articles = []
    for item in soup.select('div.news-title a'):  # ABS-CBN headlines
        headline = item.text.strip()
        summary = None  # ABS-CBN may not have easy-to-find summaries on the homepage
        articles.append({'headline': headline, 'summary': summary})
    
    return articles

def scrape_philstar():
    url = "https://www.philstar.com"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    articles = []
    for item in soup.select('h2.title a'):  # Philstar headlines
        headline = item.text.strip()
        summary = None
        articles.append({'headline': headline, 'summary': summary})
    
    return articles

def scrape_pna():
    url = "https://www.pna.gov.ph"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    articles = []
    for item in soup.select('div.post-content h1 a'):  # PNA headlines
        headline = item.text.strip()
        summary = None
        articles.append({'headline': headline, 'summary': summary})
    
    return articles

def scrape_rappler():
    url = "https://www.rappler.com"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    articles = []
    for item in soup.select('h2.entry-title a'):  # Rappler headlines
        headline = item.text.strip()
        summary = None
        articles.append({'headline': headline, 'summary': summary})
    
    return articles

# Unified function to scrape from all websites
def scrape_websites(websites_to_scrape):
    all_articles = []
    
    for website in websites_to_scrape:
        if website == "https://www.inquirer.net":
            all_articles.extend(scrape_inquirer())
        elif website == "https://news.abs-cbn.com":
            all_articles.extend(scrape_abs_cbn())
        elif website == "https://www.philstar.com":
            all_articles.extend(scrape_philstar())
        elif website == "https://www.pna.gov.ph":
            all_articles.extend(scrape_pna())
        elif website == "https://www.rappler.com":
            all_articles.extend(scrape_rappler())
    
    return all_articles


# 2. Text Processing Using NLTK
def process_text(text):
    """
    Tokenizes and cleans up text by removing stopwords and non-alphabetic characters.
    """
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    return tokens

def get_keywords(text):
    """
    Returns the most common keywords in the provided text.
    """
    tokens = process_text(text)
    fdist = FreqDist(tokens)
    
    return fdist.most_common(10)

# 3. Sentiment Analysis
def analyze_sentiment(text):
    """
    Analyzes sentiment of the provided text.
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    return sentiment

# 4. Word Cloud Visualization
def generate_wordcloud(text):
    """
    Generates a word cloud from the given text.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Plot the wordcloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# 5. CLI for User Interaction
def user_interface():
    """
    CLI for user interaction to choose actions like scraping websites or analyzing text.
    """
    all_articles = []
    
    while True:
        print("\n--- News Analyzer Menu ---")
        print("1. Scrape websites")
        print("2. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            # Scrape websites
            print("Select websites to scrape (enter numbers separated by commas):")
            for i, site in enumerate(websites, 1):
                print(f"{i}. {site}")
            site_numbers = input("Enter website numbers: ")
            selected_sites = [websites[int(i) - 1] for i in site_numbers.split(',')]
            print(f"Scraping {', '.join(selected_sites)}...")
            all_articles = scrape_websites(selected_sites)
            print(f"Scraped {len(all_articles)} articles.")
            
            while True:
                print("\n--- Analyze Articles ---")
                print("1. Perform sentiment analysis")
                print("2. Extract keywords")
                print("3. Generate word cloud")
                print("4. Back to main menu")
                
                analysis_choice = input("Enter your choice: ")
                
                if analysis_choice == '1':
                    # Sentiment analysis
                    article_headlines = [article['headline'] for article in all_articles]
                    print("Select an article to analyze:")
                    for i, headline in enumerate(article_headlines, 1):
                        print(f"{i}. {headline}")
                    article_idx = int(input("Enter article number: "))
                    text = all_articles[article_idx - 1]['summary'] or all_articles[article_idx - 1]['headline']
                    sentiment = analyze_sentiment(text)
                    print(f"Sentiment Analysis: {sentiment}")
                
                elif analysis_choice == '2':
                    # Keyword extraction
                    article_headlines = [article['headline'] for article in all_articles]
                    print("Select an article to extract keywords from:")
                    for i, headline in enumerate(article_headlines, 1):
                        print(f"{i}. {headline}")
                    article_idx = int(input("Enter article number: "))
                    text = all_articles[article_idx - 1]['summary'] or all_articles[article_idx - 1]['headline']
                    keywords = get_keywords(text)
                    print(f"Top Keywords: {keywords}")
                
                elif analysis_choice == '3':
                    # Word cloud generation
                    article_headlines = [article['headline'] for article in all_articles]
                    print("Select an article to generate word cloud from:")
                    for i, headline in enumerate(article_headlines, 1):
                        print(f"{i}. {headline}")
                    article_idx = int(input("Enter article number: "))
                    text = all_articles[article_idx - 1]['summary'] or all_articles[article_idx - 1]['headline']
                    generate_wordcloud(text)
                
                elif analysis_choice == '4':
                    break
                
                else:
                    print("Invalid choice, please try again.")
        
        elif choice == '2':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice, please try again.")

# Run the CLI
if __name__ == '__main__':
    user_interface()
