import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
from PyQt5.QtCore import Qt
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
    "https://www.bbc.com/news",
    "https://www.philstar.com",
    "https://www.manilatimes.net",
    "https://www.rappler.com"
]

# Define scraping functions (same as provided above) ...

# Text Processing, Sentiment Analysis, and Word Cloud generation functions (same as provided above) ...
# 1. Web Scraping Functionality
def scrape_inquirer():
    url = "https://newsinfo.inquirer.net"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    articles = []
    
    # Find all <h6> elements containing headlines
    for item in soup.find_all('h6'):
        headline_tag = item.find('a')
        if headline_tag:
            headline = headline_tag.text.strip()  # Extract text
            link = headline_tag['href']  # Extract the link
            
            # Append the headline and link to the articles list
            articles.append({
                'headline': headline,
                'summary': "No Summary",  # Inquirer may not have summaries on the main page
                'link': link
            })
    
    return articles

def scrape_bbc():
    """Scrape and return the latest news headlines from BBC News."""
    # URL of BBC News
    url = 'https://www.bbc.com/news'

    # Send a GET request to the website
    response = requests.get(url)

    # List to store headlines
    articles = []

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all headlines in <h2> tags
        headlines = soup.find_all('h2')

        if not headlines:
            print("No headlines found. The structure may have changed.")
            return articles

        # Iterate through the headlines, extracting and cleaning the text
        for headline in headlines:
            title = headline.get_text(strip=True)
            # Using regex to filter out empty titles
            if re.match(r'.+', title):
                # Append the headline to the articles list
                articles.append({'headline': title, 'summary': None})
    else:
        print(f"Failed to retrieve data: {response.status_code}")

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

def scrape_manilaTimes():
    url = "https://www.manilatimes.net"
    
    # Send a GET request to fetch the HTML content of the page
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return []
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # List to store extracted article data (headlines and summaries)
    articles = []
    
    # Targeting article titles with specific classes like 'article-title-h1', 'article-title-h4', and 'article-title-h5'
    headline_classes = ['article-title-h1', 'article-title-h4', 'article-title-h5']
    
    # Extract headlines for each class
    for class_name in headline_classes:
        headline_divs = soup.find_all('div', class_=class_name)
        
        for div in headline_divs:
            a_tag = div.find('a')  # Find the <a> tag inside the div
            if a_tag:
                headline = a_tag.get_text(strip=True)
                summary = None  # Placeholder for summary
                if headline:
                    articles.append({'headline': headline, 'summary': summary})
    
    # Optionally, you can add logic to extract more details like summaries if they exist in a different section

    return articles

def scrape_rappler():
    url = "https://www.rappler.com"
    
    # Send a GET request to fetch the HTML content of the page
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return []
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # List to store extracted article data (headlines and summaries)
    articles = []
    
    # Extract headlines (using <h3> and <a> tags for Rappler)
    for item in soup.select('h3 a'):  # Selecting <h3> tags with nested <a> tags
        headline = item.get_text(strip=True)
        summary = None  # Placeholder if you want to add summaries later
        if headline:
            articles.append({'headline': headline, 'summary': summary})
    
    # Extract titles from divs that contain the 'data-title' attribute (e.g., for video titles)
    for div in soup.find_all('div', attrs={'data-title': True}):
        video_title = div.get('data-title').strip()
        if video_title:
            articles.append({'headline': video_title, 'summary': None})
    
    return articles

# Unified function to scrape from all websites
def scrape_websites(websites_to_scrape):
    all_articles = []
    
    for website in websites_to_scrape:
        if website == "https://www.inquirer.net":
            all_articles.extend(scrape_inquirer())
        elif website == "https://www.bbc.com/news":
            all_articles.extend(scrape_bbc())
        elif website == "https://www.philstar.com":
            all_articles.extend(scrape_philstar())
        elif website == "https://www.manilatimes.net":
            all_articles.extend(scrape_manilaTimes())
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

def get_keywords_from_all_articles(articles):
    """
    Returns the most common keywords from all articles combined and displays a bar chart.
    """
    # Combine all headlines and summaries into a single text corpus
    combined_text = ' '.join(
        article['headline'] + ' ' + (article['summary'] or '') for article in articles
    )
    
    # Tokenize and clean the combined text
    tokens = process_text(combined_text)
    fdist = FreqDist(tokens)
    
    # Get the most common 10 keywords
    most_common_keywords = fdist.most_common(10)
    
    # Separate the keywords and their frequencies for plotting
    keywords, frequencies = zip(*most_common_keywords)
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(keywords, frequencies, color='skyblue')
    plt.title('Top 10 Keywords from All Articles')
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return most_common_keywords

# 3. Sentiment Analysis
def analyze_sentiment(text):
    """
    Analyzes sentiment of the provided text.
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    return sentiment

def analyze_sentiment_overall(articles):
    """
    Analyzes the sentiment of all articles and calculates an overall sentiment summary.
    Returns: a dictionary with the count of positive, negative, neutral articles, and the overall sentiment.
    """
    sia = SentimentIntensityAnalyzer()
    positive, negative, neutral = 0, 0, 0  # Counters for sentiment types
    total_compound = 0  # To calculate average sentiment

    for article in articles:
        text = article['summary'] or article['headline']
        sentiment = sia.polarity_scores(text)
        total_compound += sentiment['compound']
        
        # Categorize the sentiment based on compound score
        if sentiment['compound'] >= 0.05:
            positive += 1
        elif sentiment['compound'] <= -0.05:
            negative += 1
        else:
            neutral += 1

    # Determine the overall sentiment
    average_compound = total_compound / len(articles)
    if average_compound >= 0.05:
        overall_sentiment = "Mostly Positive"
    elif average_compound <= -0.05:
        overall_sentiment = "Mostly Negative"
    else:
        overall_sentiment = "Neutral"
    
    return {
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'overall_sentiment': overall_sentiment
    }

# 4. Word Cloud Visualization
def generate_wordcloud_from_all_articles(articles):
    """
    Generates a word cloud from the headlines and summaries of all articles.
    """
    # Combine all headlines and summaries into a single text corpus
    combined_text = ' '.join(
        article['headline'] + ' ' + (article['summary'] or '') for article in articles
    )
    
    # Clean and tokenize the combined text
    tokens = word_tokenize(combined_text.lower())
    tokens = [word for word in tokens if word.isalpha()]  # Keep only alphabetic words
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    
    # Join tokens back into a string for word cloud generation
    cleaned_text = ' '.join(tokens)
    
    # Generate and display the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
    
    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
class NewsAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.all_articles = []  # Store scraped articles here

        # Set up the UI
        self.setWindowTitle('News Analyzer')
        self.setGeometry(100, 100, 800, 600)
        
        self.layout = QVBoxLayout()
        
        # Scrape Button
        self.scrape_button = QPushButton('Scrape Websites')
        self.scrape_button.clicked.connect(self.scrape_websites)
        self.layout.addWidget(self.scrape_button)
        
        # Display Area
        self.display_area = QTextEdit()
        self.display_area.setReadOnly(True)
        self.layout.addWidget(self.display_area)
        
        # Keyword Extraction Button
        self.keyword_button = QPushButton('Extract Keywords and Show Bar Chart')
        self.keyword_button.clicked.connect(self.display_keywords)
        self.layout.addWidget(self.keyword_button)
        
        # Sentiment Analysis Button
        self.sentiment_button = QPushButton('Overall Sentiment Analysis')
        self.sentiment_button.clicked.connect(self.analyze_sentiment)
        self.layout.addWidget(self.sentiment_button)
        
        # Word Cloud Button
        self.wordcloud_button = QPushButton('Generate Word Cloud')
        self.wordcloud_button.clicked.connect(self.display_wordcloud)
        self.layout.addWidget(self.wordcloud_button)
        
        # Set the main layout
        self.setLayout(self.layout)
    
    def scrape_websites(self):
        """
        Scrape selected websites and display the number of articles scraped.
        """
        self.display_area.clear()
        self.all_articles = scrape_websites(websites)
        self.display_area.append(f"Scraped {len(self.all_articles)} articles.")
    
    def display_keywords(self):
        """
        Extract keywords and display a bar chart.
        """
        if self.all_articles:
            self.display_area.append("\nExtracting keywords and generating bar chart...")
            keywords = get_keywords_from_all_articles(self.all_articles)
            self.display_area.append(f"Top Keywords: {keywords}")
        else:
            self.display_area.append("No articles available for keyword extraction.")
    
    def analyze_sentiment(self):
        """
        Perform sentiment analysis on all articles and display the result.
        """
        if self.all_articles:
            self.display_area.append("\nPerforming overall sentiment analysis...")
            sentiment_summary = analyze_sentiment_overall(self.all_articles)
            self.display_area.append(f"\nOverall Sentiment Analysis:\n")
            self.display_area.append(f"Positive articles: {sentiment_summary['positive']}")
            self.display_area.append(f"Negative articles: {sentiment_summary['negative']}")
            self.display_area.append(f"Neutral articles: {sentiment_summary['neutral']}")
            self.display_area.append(f"Overall Sentiment: {sentiment_summary['overall_sentiment']}")
        else:
            self.display_area.append("No articles available for sentiment analysis.")
    
    def display_wordcloud(self):
        """
        Generate and display a word cloud from all articles.
        """
        if self.all_articles:
            self.display_area.append("\nGenerating word cloud...")
            generate_wordcloud_from_all_articles(self.all_articles)
        else:
            self.display_area.append("No articles available for word cloud generation.")

# Create and run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NewsAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
