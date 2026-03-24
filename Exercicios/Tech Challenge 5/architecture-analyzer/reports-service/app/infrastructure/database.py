import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

engine = None
AsyncSessionLocal = None


class Base(DeclarativeBase):
    pass


async def init_db(database_url: str, retries: int = 5, delay: float = 3.0) -> None:
    global engine, AsyncSessionLocal

    for attempt in range(1, retries + 1):
        try:
            engine = create_async_engine(database_url, echo=False, pool_pre_ping=True)
            AsyncSessionLocal = sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )
            async with engine.begin() as conn:
                from . import models  # noqa: F401
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Banco de dados inicializado com sucesso")
            return
        except Exception as e:
            logger.warning(f"Tentativa {attempt}/{retries} falhou: {e}")
            if attempt < retries:
                await asyncio.sleep(delay)
    raise RuntimeError("Não foi possível conectar ao banco de dados")


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session
