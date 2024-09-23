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
                print("1. Perform sentiment analysis on a specific article")
                print("2. Perform overall sentiment analysis on all articles")
                print("3. Extract keywords")
                print("4. Generate word cloud from all articles")
                print("5. Back to main menu")
                
                analysis_choice = input("Enter your choice: ")
                
                if analysis_choice == '1':
                    # Sentiment analysis for a specific article
                    article_headlines = [article['headline'] for article in all_articles]
                    print("Select an article to analyze:")
                    for i, headline in enumerate(article_headlines, 1):
                        print(f"{i}. {headline}")
                    article_idx = int(input("Enter article number: "))
                    text = all_articles[article_idx - 1]['summary'] or all_articles[article_idx - 1]['headline']
                    sentiment = analyze_sentiment(text)
                    print(f"Sentiment Analysis: {sentiment}")
                
                elif analysis_choice == '2':
                    # Overall sentiment analysis for all articles
                    if all_articles:
                        print("\nPerforming overall sentiment analysis on all articles...")
                        sentiment_summary = analyze_sentiment_overall(all_articles)
                        print(f"\nOverall Sentiment Analysis:\n")
                        print(f"Positive articles: {sentiment_summary['positive']}")
                        print(f"Negative articles: {sentiment_summary['negative']}")
                        print(f"Neutral articles: {sentiment_summary['neutral']}")
                        print(f"Overall Sentiment: {sentiment_summary['overall_sentiment']}")
                    else:
                        print("No articles available for sentiment analysis.")
                
                elif analysis_choice == '3':
                    # Keyword extraction from all articles
                    if all_articles:
                        print("\nExtracting keywords from all articles and generating bar chart...")
                        keywords = get_keywords_from_all_articles(all_articles)
                        print(f"Top Keywords from All Articles: {keywords}")
                    else:
                        print("No articles available for keyword extraction.")
                
                elif analysis_choice == '4':
                    # Word cloud generation from all articles
                    print("\nGenerating word cloud from all articles...")
                    generate_wordcloud_from_all_articles(all_articles)
                
                elif analysis_choice == '5':
                    break
                
                else:
                    print("Invalid choice, please try again.")
        
        elif choice == '2':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice, please try again.")
            
if __name__ == '__main__':
    user_interface()
