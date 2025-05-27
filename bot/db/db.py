import io
import os
import time

from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config.config import DB_PATH

Base = declarative_base()

class Generation(Base):
    __tablename__ = 'generations'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    timestamp = Column(String, nullable=False)
    prompt = Column(Text,   nullable=False)
    file_path = Column(String, nullable=False)


engine = create_engine(
    f'sqlite:///{DB_PATH}',
    connect_args={'check_same_thread': False}
)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)


def save_generation(user_id: int, prompt: str, img_buf: io.BytesIO):
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{user_id}_{ts}.png"
    os.makedirs('results', exist_ok=True)
    path = os.path.join('results', filename)
    with open(path, 'wb') as f:
        f.write(img_buf.getbuffer())

    db = SessionLocal()
    gen = Generation(user_id=user_id, timestamp=ts, prompt=prompt, file_path=path)
    db.add(gen)
    db.commit()
    db.close()
    

def get_last_prompt(user_id: int) -> str:
    db = SessionLocal()
    last = db.query(Generation).filter(Generation.user_id == user_id).order_by(Generation.id.desc()).first()
    db.close()
    return last.prompt if last else None
