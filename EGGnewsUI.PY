from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, 
    QTextEdit, QCheckBox, QDialog, QDialogButtonBox, QVBoxLayout, QGroupBox,
    QScrollArea, QComboBox
)
import sys
import re
import requests
from bs4 import BeautifulSoup
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

websites = [
    "https://www.foxnews.com/",
    "https://www.bbc.com/news",
    "https://www.philstar.com",
    "https://www.manilatimes.net",
    "https://www.rappler.com"
]

# 1. Web Scraping Functionality
def scrape_foxnews():
    """Scrape and return the latest news headlines from Fox News."""
    news_url = "https://www.foxnews.com/"
    
    response = requests.get(news_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract headline text
    headlines = soup.find_all('h3')  # Adjust this if necessary

    # List to store articles with headline and link
    articles = []
    seen_headlines = set()  # Track seen headlines to avoid duplicates

    for headline in headlines:
        headline_text = headline.get_text(strip=True)
        link = headline.find('a')['href'] if headline.find('a') else None  # Extract link if available

        if is_valid_headline(headline_text) and headline_text not in seen_headlines:
            seen_headlines.add(headline_text)
            articles.append({
                'headline': headline_text,
                'summary': None,  # Summary can be added later if needed
                'link': link
            })

    return articles

def is_valid_headline(headline):
    """Check if the headline is valid based on predefined criteria."""
    return len(headline) > 10 and not headline.startswith(('Fox Nation', 'Features & Faces', 'Political cartoons of the day'))

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
    """Scrape and return the latest news headlines from Philstar."""

    # URL of the Philstar website
    url = 'https://www.philstar.com/'

    # Send a GET request to the website
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all the headlines inside h2 tags
    headlines = soup.find_all('h2')

    # Exclude the first and last h2 tags
    filtered_headlines = headlines[1:-1]

    # List to store articles with headline and link
    articles = []

    # Iterate through each filtered headline
    for headline in filtered_headlines:
        headline_text = headline.text.strip()
        link = headline.find('a')['href'] if headline.find('a') else None  # Extract link if available

        # Append the headline and link to the articles list
        articles.append({
            'headline': headline_text,
            'summary': None,  # Summary can be added later if needed
            'link': link
        })

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
        if website == "https://www.foxnews.com/":
            all_articles.extend(scrape_foxnews())
        elif website == "https://www.bbc.com/news":
            all_articles.extend(scrape_bbc())
        elif website == "https://www.philstar.com":
            all_articles.extend(scrape_philstar())
        elif website == "https://www.manilatimes.net":
            all_articles.extend(scrape_manilaTimes())
        elif website == "https://www.rappler.com":
            all_articles.extend(scrape_rappler())
    
    return all_articles

# Class for showing a list of articles in a scrollable dialog
class ArticleDisplayDialog(QDialog):
    def __init__(self, articles, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Articles")
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout()
        
        # Scroll area for articles
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Widget for scroll area
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Adding articles as QLabel
        for article in articles:
            headline_label = QLabel(f"• {article['headline']}")
            headline_label.setWordWrap(True)
            scroll_layout.addWidget(headline_label)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        # Close button
        close_button = QDialogButtonBox(QDialogButtonBox.Close)
        close_button.rejected.connect(self.reject)
        layout.addWidget(close_button)

        self.setLayout(layout)

# Class for selecting a website and displaying its articles
class WebsiteArticleDialog(QDialog):
    def __init__(self, websites, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select a Website")
        
        layout = QVBoxLayout()
        
        # ComboBox for website selection
        self.website_combo = QComboBox()
        self.website_combo.addItems(websites)
        layout.addWidget(self.website_combo)
        
        # Button box for OK and Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_selected_website(self):
        return self.website_combo.currentText()
    
class WebsiteSelectionDialog(QDialog):
    def __init__(self, websites, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Websites")
        self.websites = websites
        self.selected_websites = []

        layout = QVBoxLayout()

        self.checkboxes = []
        for website in websites:
            checkbox = QCheckBox(website)
            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def accept(self):
        self.selected_websites = [
            cb.text() for cb in self.checkboxes if cb.isChecked()
        ]
        super().accept()

    def reject(self):
        super().reject()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("News Analyzer")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.info_label = QLabel("Select an action below:")
        self.layout.addWidget(self.info_label)

        # Group buttons in a single area
        self.button_group = QGroupBox()
        self.button_layout = QVBoxLayout()

        self.scrape_button = QPushButton("Scrape Websites")
        self.scrape_button.clicked.connect(self.show_website_selection)
        self.button_layout.addWidget(self.scrape_button)
        
        self.articles_button = QPushButton("Display articles")
        self.articles_button.clicked.connect(self.show_my_articles)
        self.button_layout.addWidget(self.articles_button)

        self.analyze_sentiment_button = QPushButton("Analyze Sentiment")
        self.analyze_sentiment_button.clicked.connect(self.analyze_articles_sentiment)
        self.button_layout.addWidget(self.analyze_sentiment_button)

        self.extract_keywords_button = QPushButton("Extract Keywords")
        self.extract_keywords_button.clicked.connect(self.extract_keywords_from_articles)
        self.button_layout.addWidget(self.extract_keywords_button)

        self.generate_wordcloud_button = QPushButton("Generate Word Cloud")
        self.generate_wordcloud_button.clicked.connect(self.generate_wordcloud)
        self.button_layout.addWidget(self.generate_wordcloud_button)

        # Set layout to the button group
        self.button_group.setLayout(self.button_layout)
        self.layout.addWidget(self.button_group)

        # Text display area for showing results
        self.results_display = QTextEdit()
        self.layout.addWidget(self.results_display)

        # Initialize articles list
        self.articles = []

    def show_website_selection(self):
        dialog = WebsiteSelectionDialog(websites)
        if dialog.exec_() == QDialog.Accepted:
            selected_websites = dialog.selected_websites
            if selected_websites:
                self.scrape_selected_websites(selected_websites)
            else:
                self.results_display.append("No websites selected.")

    def scrape_selected_websites(self, selected_websites):
        self.articles = scrape_websites(selected_websites)
        self.results_display.append(f"Scraped {len(self.articles)} articles.")
        if self.articles:
            self.results_display.append("Articles scraped successfully.")
        else:
            self.results_display.append("No articles were scraped.")

    def analyze_articles_sentiment(self):
        if not self.articles:
            self.results_display.append("No articles to analyze. Please scrape some websites first.")
            return

        sentiment_summary = analyze_sentiment_overall(self.articles)
        self.results_display.append("\nSentiment Analysis:")
        self.results_display.append(f"Positive: {sentiment_summary['positive']}")
        self.results_display.append(f"Negative: {sentiment_summary['negative']}")
        self.results_display.append(f"Neutral: {sentiment_summary['neutral']}")
        self.results_display.append(f"\nOverall Sentiment: {sentiment_summary['overall_sentiment']}")

    def extract_keywords_from_articles(self):
        if not self.articles:
            self.results_display.append("No articles to analyze. Please scrape some websites first.")
            return

        keywords = get_keywords_from_all_articles(self.articles)
        self.results_display.append("Keywords extracted from all articles:")
        for keyword, frequency in keywords:
            self.results_display.append(f"{keyword}: {frequency}")

    def generate_wordcloud(self):
        if not self.articles:
            self.results_display.append("No articles to analyze. Please scrape some websites first.")
            return

        self.results_display.append("Generating Word Cloud...")
        generate_wordcloud_from_all_articles(self.articles)
        
    def show_my_articles(self):
        # Show dialog to select website
        dialog = WebsiteArticleDialog(websites)
        if dialog.exec_() == QDialog.Accepted:
            selected_website = dialog.get_selected_website()
            self.results_display.append(f"Fetching articles from: {selected_website}")
            
            # Scrape articles from the selected website
            articles = scrape_websites([selected_website])
            
            if articles:
                # Display articles in a new dialog
                article_dialog = ArticleDisplayDialog(articles)
                article_dialog.exec_()
            else:
                self.results_display.append("No articles found.")

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
