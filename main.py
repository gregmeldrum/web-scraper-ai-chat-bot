from scraper import scrape
from embeddings import embed
from chat import chat

# Starting URL
start_url = "https://docs.solace.com/Cloud/Event-Portal/event-portal-lp.htm"
# Starting depth
start_depth = 3

scrape(start_url, start_depth)
embed()
chat()

