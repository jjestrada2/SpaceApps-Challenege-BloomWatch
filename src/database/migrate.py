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
import redis.asyncio as redis
from pathlib import Path
from alembic import command
from alembic.config import Config
from concurrent.futures import ThreadPoolExecutor
import os


async def run_migrations():
    """Run Alembic migrations programmatically with Redis lock"""
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_client = redis.Redis(host=redis_host, port=6379)

    async with redis_client.lock("migration_lock", timeout=60, blocking_timeout=30):
        # Get the project root directory (mundi-public)
        project_root = Path(__file__).parent.parent.parent
        alembic_cfg = Config(project_root / "alembic.ini")

        # Set the script location to absolute path
        alembic_cfg.set_main_option("script_location", str(project_root / "alembic"))

        def run_upgrade():
            """Run the synchronous alembic upgrade in a thread"""
            command.upgrade(alembic_cfg, "head")

        try:
            # Run synchronous Alembic command in a thread pool with timeout
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                await asyncio.wait_for(
                    loop.run_in_executor(executor, run_upgrade), timeout=30.0
                )
            print("✅ Database migrations completed successfully")
            return True
        except asyncio.TimeoutError:
            print("❌ Migration failed: Timeout after 30 seconds")
            raise Exception("Migration timeout")
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            raise


# For running standalone
if __name__ == "__main__":
    asyncio.run(run_migrations())
