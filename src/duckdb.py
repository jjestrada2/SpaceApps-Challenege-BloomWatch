# Copyright (C) 2025 Bunting Labs, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import asyncio
import time
import duckdb
import json
import re
from fastapi import HTTPException, status

from src.fs_lru import layer_cache

DUCKDB_RESERVED_KEYWORDS = {
    "select",
    "from",
    "where",
    "table",
    "group",
    "order",
    "insert",
    "update",
    "delete",
    "join",
    "on",
    "into",
    "and",
    "or",
    "not",
    "as",
    "by",
    "limit",
    "offset",
    "union",
    "distinct",
    "case",
    "when",
    "then",
    "else",
    "end",
    "create",
    "drop",
    "alter",
    "null",
    "is",
    "in",
    "like",
    "having",
}


def quoted_col_for(name: str) -> str:
    if not name:
        return '"{}"'.format(name)

    # If it's not a valid unquoted identifier, quote it
    if (
        not re.match(r"^[a-z_][a-z0-9_]*$", name)  # Valid unquoted SQL identifier
        or name.lower() in DUCKDB_RESERVED_KEYWORDS  # Reserved keyword
        or any(c.isupper() for c in name)  # Mixed/capital case
    ):
        return f'"{name}"'

    return name


async def execute_duckdb_query(
    sql_query: str, layer_id: str, max_n_rows: int = 25, timeout: int = 10
):
    start_time = time.time()
    cache = layer_cache()
    # Acquire cached geopackage path in async context
    async with cache.layer_filename(layer_id) as gpkg_path:

        def query_func():
            con = duckdb.connect(":memory:")
            # Extensions are cached locally
            con.install_extension("spatial")
            con.load_extension("spatial")

            # Create table from cached parquet file
            con.execute(f"""
                CREATE OR REPLACE TABLE {layer_id} AS
                SELECT * FROM ST_Read('{gpkg_path}');
            """)

            cursor = con.execute(sql_query)
            headers = [col[0] for col in cursor.description]
            rows = cursor.fetchall()[:max_n_rows]
            result_json = json.loads(json.dumps(rows))

            return {
                "status": "success",
                "duration_ms": 1000 * (time.time() - start_time),
                "result": result_json,
                "headers": headers,
                "row_count": len(rows),
                "query": sql_query,
            }

        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, query_func), timeout=timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"DuckDB query timed out after {timeout} seconds",
            )
