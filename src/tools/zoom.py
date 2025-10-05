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

from typing import Any, Dict
from pydantic import BaseModel, Field, model_validator
from src.routes.websocket import kue_ephemeral_action
import asyncio
from src.tools.pyd import MundiToolCallMetaArgs


class ZoomToBoundsArgs(BaseModel):
    bounds: list[float] = Field(
        ...,
        description="Bounding box in WGS84 format [xmin, ymin, xmax, ymax]",
        min_length=4,
        max_length=4,
    )
    zoom_description: str = Field(
        ...,
        description='Complete message to display to the user while zooming, e.g. "Zooming to 39 selected parcels near Ohio"',
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "ZoomToBoundsArgs":
        b = self.bounds
        if len(b) != 4:
            raise ValueError("Bounds must contain exactly 4 numbers")
        west, south, east, north = b
        # numeric check is implicit because type is float; still guard for NaN
        for x in b:
            if x != x:  # NaN check
                raise ValueError("Bounds must be valid numbers")
        if west >= east or south >= north:
            raise ValueError("Invalid bounds: west < east and south < north required")
        if not (
            -180 <= west <= 180
            and -180 <= east <= 180
            and -90 <= south <= 90
            and -90 <= north <= 90
        ):
            raise ValueError("Bounds must be within WGS84 range")
        return self


async def zoom_to_bounds(
    args: ZoomToBoundsArgs, mundi: MundiToolCallMetaArgs
) -> Dict[str, Any]:
    """Zoom the map to a specific bounding box in WGS84 coordinates. This will save the user's current zoom location to history and navigate to the new bounds."""
    bounds = args.bounds
    description = args.zoom_description
    async with kue_ephemeral_action(
        mundi.conversation_id,
        description,
        update_style_json=False,
        bounds=bounds,
    ):
        await asyncio.sleep(0.5)
    return {"status": "success", "bounds": bounds}
