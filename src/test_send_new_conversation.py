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

import pytest
import os
import random
from pathlib import Path
from unittest.mock import patch, AsyncMock
from openai.types.chat import (
    ChatCompletionMessage,
)


class MockChoice:
    def __init__(self, content: str, tool_calls=None):
        self.message = ChatCompletionMessage(
            content=content, tool_calls=tool_calls, role="assistant"
        )


class MockResponse:
    def __init__(self, content: str, tool_calls=None):
        self.choices = [MockChoice(content, tool_calls)]


@pytest.fixture
def sync_test_map_with_vector_layers(sync_auth_client):
    map_payload = {
        "title": "Geoprocessing Test Map",
        "description": "Test map for geoprocessing operations with vector layers",
    }
    map_response = sync_auth_client.post("/api/maps/create", json=map_payload)
    assert map_response.status_code == 200, f"Failed to create map: {map_response.text}"
    current_map_id = map_response.json()["id"]
    layer_ids = {}

    def _upload_layer(file_name, layer_name_in_db, map_id):
        file_path = str(Path(__file__).parent.parent / "test_fixtures" / file_name)
        if not os.path.exists(file_path):
            pytest.skip(f"Test file {file_path} not found")
        with open(file_path, "rb") as f:
            layer_response = sync_auth_client.post(
                f"/api/maps/{map_id}/layers",
                files={"file": (file_name, f, "application/octet-stream")},
                data={"layer_name": layer_name_in_db},
            )
            assert layer_response.status_code == 200, (
                f"Failed to upload layer {file_name}: {layer_response.text}"
            )
            response_data = layer_response.json()
            return response_data["id"], response_data["dag_child_map_id"]

    random.seed(42)
    layer_id, current_map_id = _upload_layer(
        "barcelona_beaches.fgb", "Barcelona Beaches", current_map_id
    )
    layer_ids["beaches_layer_id"] = layer_id

    layer_id, current_map_id = _upload_layer(
        "barcelona_cafes.fgb", "Barcelona Cafes", current_map_id
    )
    layer_ids["cafes_layer_id"] = layer_id

    layer_id, current_map_id = _upload_layer(
        "idaho_weatherstations.geojson", "Idaho Weather Stations", current_map_id
    )
    layer_ids["idaho_stations_layer_id"] = layer_id

    return {"map_id": current_map_id, **layer_ids}


@pytest.mark.anyio
async def test_chat_completions(
    sync_test_map_with_vector_layers,
    sync_auth_client,
    websocket_url_for_map,
):
    map_id = sync_test_map_with_vector_layers["map_id"]

    response_queue = [
        MockResponse("hello"),
        MockResponse("hi"),
    ]

    with patch("src.routes.message_routes.get_openai_client") as mock_get_client:
        mock_client = AsyncMock()

        async def mock_create(*args, **kwargs):
            return response_queue.pop(0)

        mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
        mock_get_client.return_value = mock_client

        # First /send call to NEW
        response = sync_auth_client.post(
            f"/api/maps/conversations/NEW/maps/{map_id}/send",
            json={
                "message": {
                    "role": "user",
                    "content": "first message",
                },
                "selected_feature": None,
            },
        )
        assert response.status_code == 200
        data = response.json()
        conversation_id = data["conversation_id"]

        with sync_auth_client.websocket_connect(
            websocket_url_for_map(map_id, conversation_id)
        ) as websocket:
            # Second /send call to conversation_id
            response = sync_auth_client.post(
                f"/api/maps/conversations/{conversation_id}/maps/{map_id}/send",
                json={
                    "message": {
                        "role": "user",
                        "content": "second message",
                    },
                    "selected_feature": None,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["conversation_id"] == conversation_id
            assert data["sent_message"]["role"] == "user"
            assert data["sent_message"]["content"] == "second message"
            assert data["status"] == "processing_started"

            # Receive the second response
            receive_json = websocket.receive_json()
            assert receive_json["role"] == "user"
            assert receive_json["content"] == "second message"
            assert receive_json["conversation_id"] == conversation_id

            receive_json = websocket.receive_json()
            assert receive_json["ephemeral"]
            assert receive_json["action"] == "Kue is thinking..."
            assert receive_json["status"] == "active"

            receive_json = websocket.receive_json()
            assert receive_json["ephemeral"]
            assert receive_json["action"] == "Kue is thinking..."
            assert receive_json["status"] == "completed"

            receive_json = websocket.receive_json()
            assert receive_json["role"] == "assistant"
            assert receive_json["content"] == "hi"
            assert receive_json["conversation_id"] == conversation_id
