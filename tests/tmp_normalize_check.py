import sys
from pathlib import Path
import types

# ensure workspace root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# stub lancedb if not installed so we can import the client module for unit checks
if "lancedb" not in sys.modules:
    sys.modules["lancedb"] = types.ModuleType("lancedb")

from app.clients import lancedb_store


class FakeType:
    list_size = 4


class FakeField:
    type = FakeType()


class FakeSchema:
    def field(self, name):
        return FakeField()


class FakeTable:
    schema = FakeSchema()


def main():
    table = FakeTable()
    a = lancedb_store._normalize_query_vector([0.1, 0.2], table)
    b = lancedb_store._normalize_query_vector([1.0] * 6, table)
    print("normalized short:", a)
    print("normalized long:", b)


if __name__ == '__main__':
    main()
