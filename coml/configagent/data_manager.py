from .schema import (
    Space,
    Context,
    Solution,
    Guideline,
    EmbeddingCache,
)


def create_tables() -> None:
    Space.create_table()
    Context.create_table()
    Solution.create_table()
    Guideline.create_table()
    EmbeddingCache.create_table()
