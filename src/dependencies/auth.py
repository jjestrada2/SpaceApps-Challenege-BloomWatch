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

import os
from typing import List
from fastapi import HTTPException, status, Request
from urllib.parse import urlparse


async def require_auth(
    request: Request,
) -> List[str]:
    allowed_origins_env = os.environ.get("MUNDI_EMBED_ALLOWED_ORIGINS")
    if not allowed_origins_env:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )

    allowed_origins = [
        origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()
    ]
    if not allowed_origins:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )

    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")

    origin_allowed = False
    if origin and origin in allowed_origins:
        origin_allowed = True
    elif referer:
        referer_origin = f"{urlparse(referer).scheme}://{urlparse(referer).netloc}"
        if referer_origin in allowed_origins:
            origin_allowed = True

    if not origin_allowed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )

    return allowed_origins
