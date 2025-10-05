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
import json
import os
import random
from pathlib import Path
from unittest.mock import patch, AsyncMock
from src.structures import get_async_db_connection
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function


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
    project_id = map_response.json()["project_id"]
    layer_ids = {}

    def _upload_layer(file_name, layer_name_in_db, map_id):
        file_path = str(
            Path(__file__).parent.parent.parent / "test_fixtures" / file_name
        )
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

    return {"map_id": current_map_id, "project_id": project_id, **layer_ids}


@pytest.mark.anyio
async def test_chat_completions(
    sync_test_map_with_vector_layers,
    sync_auth_client,
    websocket_url_for_map,
):
    layer_id = sync_test_map_with_vector_layers["beaches_layer_id"]
    map_id = sync_test_map_with_vector_layers["map_id"]

    def create_response_queue():
        return [
            MockResponse(
                "I'll help you buffer the beaches layer with a 100-unit distance.",
                [
                    ChatCompletionMessageToolCall(
                        id="call_1",
                        type="function",
                        function=Function(
                            name="native_buffer",
                            arguments=json.dumps({"INPUT": layer_id, "DISTANCE": 100}),
                        ),
                    )
                ],
            ),
            MockResponse(
                "Now I'll add the buffered layer to your map.",
                [
                    ChatCompletionMessageToolCall(
                        id="call_2",
                        type="function",
                        function=Function(
                            name="add_layer_to_map",
                            arguments=json.dumps(
                                {
                                    "layer_id": "$LAST_LAYER_ID",
                                    "new_name": "Buffered Beaches",
                                }
                            ),
                        ),
                    )
                ],
            ),
            MockResponse(
                "I've successfully created a buffer around the beaches with a 100-unit distance. The buffered layer has been added to your map.",
                None,
            ),
        ]

    response_queue = create_response_queue()

    with patch("src.routes.message_routes.get_openai_client") as mock_get_client:
        mock_client = AsyncMock()

        async def mock_create(*args, **kwargs):
            response = response_queue.pop(0)
            for tool_call in response.choices[0].message.tool_calls or []:
                if "$LAST_LAYER_ID" in tool_call.function.arguments:
                    async with get_async_db_connection() as conn:
                        layers = await conn.fetch(
                            "SELECT layer_id FROM map_layers WHERE source_map_id = $1 ORDER by created_on desc limit 1",
                            map_id,
                        )
                        assert layers is not None
                    actual_layer_id = layers[0]["layer_id"]
                    tool_call.function.arguments = tool_call.function.arguments.replace(
                        "$LAST_LAYER_ID", actual_layer_id
                    )
            return response

        mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
        mock_get_client.return_value = mock_client

        # Create a new conversation
        response = sync_auth_client.post(
            "/api/conversations",
            json={"project_id": sync_test_map_with_vector_layers["project_id"]},
        )
        assert response.status_code == 200
        data = response.json()
        conversation_id = data["id"]

        with sync_auth_client.websocket_connect(
            websocket_url_for_map(map_id, conversation_id)
        ) as websocket:
            response = sync_auth_client.post(
                f"/api/maps/conversations/{conversation_id}/maps/{map_id}/send",
                json={
                    "message": {
                        "role": "user",
                        "content": "Buffer the beaches layer with a distance of 100",
                    },
                    "selected_feature": None,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing_started"
            assert "message_id" in data

            # our own message
            sent_msg = websocket.receive_json()
            assert sent_msg["role"] == "user"
            assert (
                sent_msg["content"] == "Buffer the beaches layer with a distance of 100"
            )
            assert not sent_msg["has_tool_calls"]
            assert sent_msg["tool_calls"] == []
            assert sent_msg["map_id"] == map_id
            assert "created_at" in sent_msg
            assert sent_msg["conversation_id"] == conversation_id

            # Kue is thinking
            msg = websocket.receive_json()
            assert msg["ephemeral"] and msg["action"] == "Kue is thinking..."

            msg = websocket.receive_json()
            assert msg["ephemeral"] and msg["action"] == "Kue is thinking..."
            assert msg["status"] == "completed"

            # Assistant response with tool call
            msg = websocket.receive_json()
            assert msg["role"] == "assistant"
            assert (
                "I'll help you buffer the beaches layer with a 100-unit distance"
                in msg["content"]
            )
            assert msg["has_tool_calls"]
            assert len(msg["tool_calls"]) == 1
            assert msg["tool_calls"][0]["tagline"] == "native:buffer"
            assert msg["tool_calls"][0]["icon"] == "qgis"
            assert msg["conversation_id"] == conversation_id

            # Execute geoprocessing
            msg = websocket.receive_json()
            assert msg["ephemeral"] and msg["action"] == "QGIS running native:buffer..."
            assert msg["status"] == "active"

            # QGIS processing completion
            msg = websocket.receive_json()
            assert msg["ephemeral"] and msg["action"] == "QGIS running native:buffer..."
            assert msg["status"] == "completed"

            # Tool response message after tool execution completes
            msg = websocket.receive_json()
            assert msg["role"] == "tool"
            assert msg["tool_response"]["id"] == "call_1"
            assert msg["tool_response"]["status"] == "success"

            # Reach in for the tool call to actually check it worked.. its hard to tell later
            async with get_async_db_connection() as conn:
                messages = await conn.fetch(
                    "SELECT id, sender_id, message_json, created_at FROM chat_completion_messages WHERE map_id = $1 ORDER BY created_at",
                    map_id,
                )

                # Find the tool response message for call_1
                tool_response = None
                for m in messages:
                    msg_json = json.loads(dict(m)["message_json"])
                    if (
                        msg_json.get("role") == "tool"
                        and msg_json.get("tool_call_id") == "call_1"
                    ):
                        tool_response = json.loads(msg_json["content"])
                        break

                assert tool_response is not None
                assert tool_response["status"] == "success"
                assert "completed successfully" in tool_response["message"]
                assert tool_response["algorithm_id"] == "native:buffer"
                assert isinstance(tool_response["qgis_result"], dict)
                assert "created_layers" in tool_response

                # Check the created layer details
                created_layers = tool_response["created_layers"]
                assert len(created_layers) == 1
                created_layer = created_layers[0]
                assert created_layer["param_name"] == "OUTPUT"
                assert created_layer["layer_type"] == "vector"
                assert "layer_id" in created_layer
                assert "layer_name" in created_layer

                tool_call_created_layer_id = created_layer["layer_id"]

            # loops once
            msg = websocket.receive_json()
            assert msg["ephemeral"] and msg["action"] == "Kue is thinking..."
            assert msg["status"] == "active"

            msg = websocket.receive_json()
            assert msg["ephemeral"] and msg["action"] == "Kue is thinking..."
            assert msg["status"] == "completed"

            # Adding layer to map
            assistant_msg = websocket.receive_json()
            assert assistant_msg["role"] == "assistant"
            assert (
                "Now I'll add the buffered layer to your map."
                in assistant_msg["content"]
            )
            assert assistant_msg["role"] == "assistant"
            assert assistant_msg["has_tool_calls"]
            assert assistant_msg["tool_calls"][0]["tagline"] == "Adding layer to map..."
            assert assistant_msg["tool_calls"][0]["icon"] == "map-plus"

            msg = websocket.receive_json()
            assert msg["ephemeral"] and msg["action"] == "Adding layer to map..."
            assert msg["status"] == "active"

            # handle racing
            msg1 = websocket.receive_json()
            msg2 = websocket.receive_json()

            messages = [msg1, msg2]

            ephemeral_msg = next((m for m in messages if m.get("ephemeral")), None)
            assert ephemeral_msg is not None
            assert ephemeral_msg["action"] == "Adding layer to map..."
            assert ephemeral_msg["status"] == "completed"
            assert ephemeral_msg["updates"]["style_json"]

            tool_msg = next((m for m in messages if m.get("role") == "tool"), None)
            assert tool_msg is not None
            assert tool_msg["tool_response"]["id"] == "call_2"
            assert tool_msg["tool_response"]["status"] == "success"

            async with get_async_db_connection() as conn:
                layers = await conn.fetch(
                    "SELECT layer_id, metadata FROM map_layers WHERE source_map_id = $1 ORDER by created_on desc limit 1",
                    map_id,
                )
                layer_metadata = json.loads(layers[0]["metadata"])
                created_layer_id = layers[0]["layer_id"]

                assert layer_metadata["feature_count"] == 37
                assert layer_metadata["geometry_type"] == "multipolygon"

            msg = websocket.receive_json()
            assert msg["ephemeral"] and msg["action"] == "Kue is thinking..."

            msg = websocket.receive_json()
            assert msg["ephemeral"] and msg["action"] == "Kue is thinking..."
            assert msg["status"] == "completed"

            assistant_msg = websocket.receive_json()
            assert assistant_msg["role"] == "assistant"
            assert (
                "I've successfully created a buffer around the beaches with a 100-unit distance"
                in assistant_msg["content"]
            )

            async with get_async_db_connection() as conn:
                map_with_layer = await conn.fetchrow(
                    "SELECT id FROM user_mundiai_maps WHERE $1 = ANY(layers)",
                    created_layer_id,
                )
                assert map_with_layer is not None
                assert map_with_layer["id"] == map_id

                # Test layer description for the tool-created layer
                response = sync_auth_client.get(
                    f"/api/layer/{tool_call_created_layer_id}/describe"
                )
                assert response.status_code == 200, (
                    f"Failed to get layer description: {response.text}"
                )
                assert "Buffered Beaches" in response.text
                assert "Geometry Type: multipolygon" in response.text
                assert "-98.0536" in response.text
                assert "-58.7340" in response.text
                assert "102.3725" in response.text
                assert "141.4905" in response.text
                assert "lifeguard" in response.text
