"""
Step 4 -- PostgreSQL Schema
Creates three tables in stockdb:
    - price_features -> X_t (OHLCV + technical indicators)
    - sentiment_scores -> S_t and H_t(daily sentiment + entropy)
    - fused_features -> F_t (fused feature vector for the model)
"""

from sqlalchemy import(
    create_engine, Column, String, Float, Integer, Date, UniqueConstraint
)
from sqlalchemy.orm import declarative_base

DB_URL = "postgresql://stockuser:stockpass@localhost:5432/stockdb" #postgresql://user:password@host:port/database

engine = create_engine(DB_URL) # open a connection to Postgres. 
Base = declarative_base() # gives us a base class. Every table we define will inherit from it.

class PriceFeatures(Base): # inherits from Base so SQLAlchemy know that this is a tabkle definition, not a regular Python class.
    __tablename__ = "price_features"

    ticker = Column(String, primary_key=True) # the ticker and date together creates a composite primary key 
    date = Column(Date, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    sma_20 = Column(Float)
    rsi_14 = Column(Float)
    obv = Column(Float)
    adx_14 = Column(Float)
    returns = Column(Float)
    log_returns = Column(Float)


class SentimentScores(Base):
    __tablename__ = "sentiment_scores"

    ticker = Column(String, primary_key=True)
    date = Column(Date, primary_key=True)
    sentiment = Column(Float) # the daily average  of positive - negative scores for all posts on that day
    entropy = Column(Float) # the daily average Shanion entropy across all posts. High entropy = model is uncertain about sentiment direction
    post_count = Column(Integer) # no.of posts collected. Useful to know if a sentiment value is based on 1 or 50 posts
    avg_score = Column(Float) # avg reddit upvote score for all posts on that day. Useful to know if the posts are popular or not.



class FusedFeatures(Base):
    __tablename__ = "fused_features"

    ticker = Column(String, primary_key=True)
    date = Column(Date, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)     
    close = Column(Float)
    volume = Column(Float)
    sma_20 = Column(Float)
    rsi_14 = Column(Float)
    obv = Column(Float)
    adx_14 = Column(Float)
    returns = Column(Float)
    log_returns = Column(Float)
    sentiment = Column(Float)
    entropy = Column(Float)
    alpha = Column(Float) # the weight given to sentiment scores when fusing with price features. 0 = only price features, 1 = only sentiment scores.  α_t = 1 - H_t / log(3)  (H_t = daily entropy score, log(3) = max entropy for 3-class classification). If entropy is high then α_t is low, meaning we trust the sentiment scores less and rely more on price features. If entropy is low then α_t is high, meaning we trust the sentiment scores more and rely less on price features.
    post_count = Column(Integer)


def main():
    print(" Creating tables in stockdb....")
    Base.metadata.create_all(engine) # SQLAlchemy looks at all the classes that inherit from Base, translates it to SQL CREATE TABLE statements and runs against Postgres
    print("Done. Tables created:")
    for table in Base.metadata.tables: # dictionary of all taables SQLAlchemy knows about.
        print(f" -{table}")

if __name__ == "__main__":
    main()