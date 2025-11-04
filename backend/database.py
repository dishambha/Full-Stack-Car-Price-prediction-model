# backend/database.py

from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import Generator

# --- 1. CONFIGURATION ---
# The database file will be created inside the 'backend' folder
SQLALCHEMY_DATABASE_URL = "sqlite:///./prediction_history.db"

# Create a SQLAlchemy engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False} # Required for SQLite
)

# Session and Base
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 2. DATABASE SCHEMA ---

class PredictionLog(Base):
    """Database model to log every prediction request and result."""
    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, default=lambda: datetime.now().isoformat())
    
    # Input Features
    vehicle_type = Column(String(10))
    input_features = Column(Text) 
    
    # Prediction Results
    predicted_mean = Column(Float)
    predicted_min = Column(Float)
    predicted_max = Column(Float)

# --- 3. INITIALIZATION AND DEPENDENCY ---

def init_db():
    """Create the database tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)

def get_db() -> Generator:
    """Yields a database session that closes after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()