"""Memory privacy: retention, a customer's export request, and deleting everything for a user."""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent_adapter.memory import MemoryRecord, MemoryScope, SQLiteMemory


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        # --8<-- [start:retention]
        memory = SQLiteMemory(Path(folder) / "memory.db", retention_days=30)
        ann = MemoryScope(user_id="ann")
        long_ago = datetime.now(UTC) - timedelta(days=45)
        memory.add_sync(
            [
                MemoryRecord(scope=ann, text="Bought a museum pass", created_at=long_ago),
                MemoryRecord(scope=MemoryScope(user_id="ann", session_id="chat-2"), text="Vegan"),
                MemoryRecord(scope=MemoryScope(user_id="ben"), text="Likes trains"),
            ]
        )
        print([hit.record.text for hit in memory.search_sync("", scope=ann)])  # past 30 days: gone
        # --8<-- [end:retention]

        # --8<-- [start:privacy]
        exported = memory.export_sync(ann)  # every record Ann has, for her data request
        print([(record.text, record.scope.session_id) for record in exported])
        print(memory.delete_sync(scope=ann))  # erase Ann: all her sessions, nobody else
        print(memory.export_sync(ann), len(memory.export_sync(MemoryScope(user_id="ben"))))
        # --8<-- [end:privacy]


if __name__ == "__main__":
    main()
