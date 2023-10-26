___all__ = [
    "Condition",
    "Parameter",
    "Space",
    "Context",
    "Solution",
    "Guideline",
    "EmbeddingCache",
]

from dataclasses import dataclass, asdict, is_dataclass, fields
from pathlib import Path
from typing import (
    Literal,
    Dict,
    Any,
    Optional,
    List,
    TypeVar,
    Type,
    Generic,
    Iterable,
    TypedDict,
    Union,
    Tuple,
)

import numpy as np
import sqlite3
from typing_extensions import Self

# Override this variable to change the database path
database_path = Path.home() / ".coml" / "configagent.sqlite"


def database_connection():
    return sqlite3.connect(database_path)


Config = Dict[str, Any]

PrimaryKey = TypeVar("PrimaryKey")


class SqliteSerializable(Generic[PrimaryKey]):
    database_schema: str
    primary_key: Union[Tuple[str, ...], str]

    def to_tuple(self) -> tuple:
        """Converts the dataclass to a tuple of values that can be inserted into SQL."""
        raise NotImplementedError()

    @classmethod
    def from_tuple(cls: Type[Self], *tup: Any) -> Self:
        """Converts the tuple of values from SQL into a dataclass."""
        raise NotImplementedError()

    @classmethod
    def create_table(cls) -> None:
        with database_connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE {cls.__name__.upper()} (
                    {cls.database_schema}
                )
            """
            )

    @classmethod
    def insert_many(cls: Type[Self], rows: List[Self]) -> None:
        assert is_dataclass(cls)
        with database_connection() as conn:
            field_names = [f.name for f in fields(cls)]
            conn.executemany(
                f"""
                    INSERT INTO {cls.__name__.upper()} ({", ".join(field_names)})
                    VALUES ({", ".join("?" * len(field_names))}))
                """,
                [row.to_tuple() for row in rows],
            )

    @classmethod
    def query_all(cls: Type[Self]) -> Iterable[Self]:
        with database_connection() as conn:
            for row in conn.execute(
                f"""
                SELECT *
                FROM {cls.__name__.upper()}
            """
            ).fetchall():
                yield cls.from_tuple(row)

    @classmethod
    def get(cls: Type[Self], pk: PrimaryKey) -> Self:
        with database_connection() as conn:
            if isinstance(cls.primary_key, str):
                row = conn.execute(
                    f"""
                    SELECT *
                    FROM {cls.__name__.upper()}
                    WHERE {cls.primary_key} = ?
                """,
                    (pk,),
                ).fetchone()
            else:
                assert isinstance(pk, tuple)
                row = conn.execute(
                    f"""
                    SELECT *
                    FROM {cls.__name__.upper()}
                    WHERE {" AND ".join(f"{k} = ?" for k in cls.primary_key)}
                """,
                    pk,
                ).fetchone()
            return cls.from_tuple(row)


class Condition(TypedDict):
    match: Optional[Config]


@dataclass
class Parameter:
    name: str
    dtype: Literal["int", "float", "str", "bool", "any"]
    categorical: bool
    choices: List[str]
    low: Optional[float]
    high: Optional[float]
    log_distributed: Optional[bool]
    condition: Optional[Condition]
    quantiles: Optional[float]


@dataclass
class Space(SqliteSerializable[str]):
    id: str
    name: str
    description: str
    parameters: List[Parameter]

    database_schema = """
        id VARCHAR(64) NOT NULL PRIMARY KEY,
        name VARCHAR(256) NOT NULL,
        description TEXT NOT NULL,
        parameters JSON NOT NULL
    """
    primary_key = "id"

    def to_tuple(self):
        return (
            self.id,
            self.name,
            self.description,
            [asdict(p) for p in self.parameters],
        )

    @classmethod
    def from_tuple(cls, id, name, description, parameters):
        return cls(
            id=id,
            name=name,
            description=description,
            parameters=[Parameter(**p) for p in parameters],
        )


@dataclass
class Context(SqliteSerializable[str]):
    id: str
    type: Literal["dataset", "goal", "session"]
    name: str
    description: str

    database_schema = """
        id VARCHAR(64) NOT NULL PRIMARY KEY,
        type VARCHAR(64) NOT NULL,
        name VARCHAR(256) NOT NULL,
        description TEXT NOT NULL
    """
    primary_key = "id"

    def to_tuple(self):
        return (self.id, self.type, self.name, self.description)

    @classmethod
    def from_tuple(cls, id, type, name, description):
        return cls(id=id, type=type, name=name, description=description)


@dataclass
class Solution(SqliteSerializable[str]):
    id: str
    space: Space
    context: List[Context]
    config: Config
    cano_config: Config
    metric: Optional[float]
    feedback: Optional[str]

    database_schema = """
        id VARCHAR(64) NOT NULL PRIMARY KEY,
        space VARCHAR(64) NOT NULL,
        context JSON NOT NULL,
        config JSON NOT NULL,
        cano_config JSON NOT NULL,
        metric FLOAT,
        feedback STR
    """
    primary_key = "id"

    def to_tuple(self):
        return (
            self.id,
            self.space.id,
            [c.id for c in self.context],
            self.config,
            self.cano_config,
            self.metric,
            self.feedback,
        )

    @classmethod
    def from_tuple(cls, id, space, context, config, cano_config, metric, feedback):
        return cls(
            id=id,
            space=Space.get(space),
            context=[Context.get(c) for c in context],
            config=config,
            cano_config=cano_config,
            metric=metric,
            feedback=feedback,
        )


@dataclass
class Guideline(SqliteSerializable[str]):
    id: str
    space: Space
    context: List[Context]
    guideline: str

    database_schema = """
        id VARCHAR(64) NOT NULL PRIMARY KEY,
        space VARCHAR(64) NOT NULL,
        context JSON NOT NULL,
        guideline TEXT NOT NULL
    """
    primary_key = "id"

    def to_tuple(self):
        return (self.id, self.space.id, [c.id for c in self.context], self.guideline)

    @classmethod
    def from_tuple(cls, id, space, context, guideline):
        return cls(
            id=id,
            space=Space.get(space),
            context=[Context.get(c) for c in context],
            guideline=guideline,
        )


@dataclass
class EmbeddingCache(SqliteSerializable[Tuple[str, str]]):
    text: str
    model: str
    embedding: np.ndarray

    database_schema = """
        text TEXT NOT NULL,
        model VARCHAR(64) NOT NULL,
        embedding BLOB NOT NULL,
        PRIMARY KEY (text, model)
    """
    primary_id = ("text", "model")

    def to_tuple(self):
        return (self.text, self.model, self.embedding.tobytes())

    @classmethod
    def from_tuple(cls, text, model, embedding):
        return cls(
            text=text, model=model, embedding=np.frombuffer(embedding, dtype=np.float32)
        )
