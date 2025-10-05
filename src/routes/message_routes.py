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

from fastapi import APIRouter, HTTPException, status, Request, Depends
from fastapi.responses import JSONResponse
from typing import List, Union
from collections import defaultdict
from pydantic import BaseModel
import logging
import os
import json
import re
from fastapi import BackgroundTasks
from opentelemetry import trace
import io
import csv
import asyncio
import traceback
from src.dependencies.dag import get_map
from fastapi import UploadFile
import httpx
from typing import Callable
from redis import Redis
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)
from openai.types.chat import ChatCompletionMessageToolCall
from openai import APIError

from src.symbology.llm import generate_maplibre_layers_for_layer_id

from src.routes.layer_router import (
    set_layer_style as set_layer_style_route,
    SetStyleRequest,
)
from src.structures import (
    async_conn,
    SanitizedMessage,
    convert_mundi_message_to_sanitized,
)
from src.utils import get_openai_client
from src.routes.postgres_routes import (
    generate_id,
    get_map_description,
    internal_upload_layer,
    InternalLayerUploadResponse,
)
from src.geoprocessing.dispatch import (
    UnsupportedAlgorithmError,
    InvalidInputFormatError,
    get_tools,
)
from src.dependencies.conversation import get_or_create_conversation
from src.duckdb import execute_duckdb_query
from src.utils import get_async_s3_client, get_bucket_name
from src.dependencies.postgis import get_postgis_provider
from src.dependencies.layer_describer import LayerDescriber, get_layer_describer
from src.dependencies.chat_completions import ChatArgsProvider, get_chat_args_provider

# Ensure tool results with dates/decimals serialize cleanly for tool messages
def _json_default(o):
    try:
        # Handle datetime/date-like objects
        iso = getattr(o, "isoformat", None)
        if callable(iso):
            return iso()
    except Exception:
        pass
    # Fallback to string for any non-serializable types (e.g., Decimal)
    return str(o)

def json_dumps_safe(obj) -> str:
    return json.dumps(obj, default=_json_default)
from src.dependencies.map_state import (
    MapStateProvider,
    get_map_state_provider,
    SelectedFeature,
)
from src.dependencies.system_prompt import (
    SystemPromptProvider,
    get_system_prompt_provider,
)
from src.dependencies.session import (
    verify_session_required,
    UserContext,
)
from src.dependencies.postgres_connection import (
    PostgresConnectionManager,
    get_postgres_connection_manager,
)
from src.database.models import (
    MundiChatCompletionMessage,
    MundiMap,
    MapLayer,
    Conversation,
)
from src.routes.websocket import kue_ephemeral_action, kue_notify_error
from src.tools.pyd import tool_from as tool_from_pyd
from src.dependencies.pydantic_tools import (
    get_pydantic_tool_calls,
    PydanticToolRegistry,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


redis = Redis(
    host=os.environ["REDIS_HOST"],
    port=int(os.environ["REDIS_PORT"]),
    decode_responses=True,
)


async def geocode_address(address: str) -> tuple[float, float] | None:
    """Resolve a text address to (latitude, longitude).

    Uses Nominatim by default. Override with GEOCODER_URL and GEOCODER_USER_AGENT.
    Returns None if not found or on error.
    """
    if not address or not isinstance(address, str):
        return None
    try:
        url = os.environ.get(
            "GEOCODER_URL", "https://nominatim.openstreetmap.org/search"
        )
        headers = {
            "User-Agent": os.environ.get(
                "GEOCODER_USER_AGENT", "mundi.ai/1.0 (geocode)"
            )
        }
        params = {"q": address, "format": "jsonv2", "limit": 1}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params, headers=headers)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data:
            return None
        lat = float(data[0]["lat"])  # type: ignore[index]
        lon = float(data[0]["lon"])  # type: ignore[index]
        return (lat, lon)
    except Exception:
        return None


async def label_conversation_inline(conversation_id: int):
    """Generate a title for a conversation using OpenAI"""
    try:
        async with async_conn("label_conversation") as conn:
            messages = await conn.fetch(
                """
                SELECT message_json
                FROM chat_completion_messages
                WHERE conversation_id = $1
                ORDER BY created_at ASC
                LIMIT 5
                """,
                conversation_id,
            )

            if not messages:
                return

            conversation_content = []
            for msg in messages:
                message_data = json.loads(msg["message_json"])
                role = message_data.get("role", "")
                content = message_data.get("content", "")
                if content and role in ["user", "assistant"]:
                    conversation_content.append(f"{role}: {content[:200]}")

            if not conversation_content:
                return

            content_summary = "\n".join(conversation_content)

            request = Request({"type": "http", "method": "POST", "headers": []})
            openai_client = get_openai_client(request)

            response = await openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a short, descriptive title (3-6 words) for this conversation. The title should capture the main topic or request. Only return the title, nothing else.",
                    },
                    {"role": "user", "content": f"Conversation:\n{content_summary}"},
                ],
                max_tokens=20,
                temperature=0.3,
            )

            title = response.choices[0].message.content.strip()
            if title and len(title) > 0:
                await conn.execute(
                    """
                    UPDATE conversations
                    SET title = $1, updated_at = CURRENT_TIMESTAMP
                    WHERE id = $2
                    """,
                    title,
                    conversation_id,
                )
                print(f"Generated title for conversation {conversation_id}: {title}")

    except Exception as e:
        print(f"Error labeling conversation {conversation_id}: {e}")


# Create router
router = APIRouter()


class ChatCompletionMessageRow(BaseModel):
    id: int
    map_id: str
    sender_id: str
    message_json: Union[
        ChatCompletionMessageParam,
        ChatCompletionMessage,
        dict,
    ]
    created_at: str


async def get_all_conversation_messages(
    conversation_id: int,
    session: UserContext,
) -> List[MundiChatCompletionMessage]:
    async with async_conn("get_all_conversation_messages") as conn:
        db_messages = await conn.fetch(
            """
            SELECT ccm.*
            FROM chat_completion_messages ccm
            JOIN conversations c ON ccm.conversation_id = c.id
            WHERE ccm.conversation_id = $1
            AND c.owner_uuid = $2
            AND c.soft_deleted_at IS NULL
            ORDER BY ccm.created_at ASC
            """,
            conversation_id,
            session.get_user_id(),
        )

        messages: list[MundiChatCompletionMessage] = []
        for msg in db_messages:
            msg_dict = dict(msg)
            # Parse message_json ... when using raw asyncpg
            msg_dict["message_json"] = json.loads(msg_dict["message_json"])
            messages.append(MundiChatCompletionMessage(**msg_dict))
        return messages


class LayerInfo(BaseModel):
    layer_id: str
    name: str
    type: str
    geometry_type: str | None = None
    feature_count: int | None = None

    @classmethod
    def from_map_layer(cls, layer: MapLayer) -> "LayerInfo":
        return cls(
            layer_id=layer.layer_id,
            name=layer.name,
            type=layer.type,
            geometry_type=layer.geometry_type,
            feature_count=layer.feature_count,
        )


class LayerDiff(BaseModel):
    added_layers: List[LayerInfo]
    removed_layers: List[LayerInfo]


class MapNode(BaseModel):
    map_id: str
    messages: List[SanitizedMessage]
    fork_reason: str | None = None
    created_on: str
    diff_from_previous: LayerDiff | None = None


class MapTreeResponse(BaseModel):
    project_id: str
    tree: List[MapNode]


@router.get(
    "/{map_id}/tree",
    operation_id="get_map_tree",
    response_model=MapTreeResponse,
)
async def get_map_tree(
    map: MundiMap = Depends(get_map),
    conversation_id: int | None = None,
    session: UserContext = Depends(verify_session_required),
):
    leaf_map_id = map.id
    project_id = map.project_id

    # TODO: if you add a message to a previous map, it interrupts the chain.
    # adding a message should be considered creating a new node in the DAG...
    async with async_conn("describe_map_tree") as conn:
        # Collect all map IDs in the parent chain
        map_ids: list[str] = []
        current_map_id: str | None = leaf_map_id

        while current_map_id:
            map_ids.insert(0, current_map_id)

            # Get parent map ID
            parent_result = await conn.fetchrow(
                """
                SELECT parent_map_id
                FROM user_mundiai_maps
                WHERE id = $1 AND soft_deleted_at IS NULL
                """,
                current_map_id,
            )
            if not parent_result:
                break

            if parent_result["parent_map_id"] in map_ids:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Encountered loop in DAG inside describe_map_tree",
                )

            current_map_id = parent_result["parent_map_id"]

        # Fetch all map data including layers
        db_maps = await conn.fetch(
            """
            SELECT id, fork_reason, created_on, layers
            FROM user_mundiai_maps
            WHERE id = ANY($1) AND soft_deleted_at IS NULL
            ORDER BY array_position($1, id)
            """,
            map_ids,
        )
        db_maps: List[MundiMap] = [MundiMap(**dict(map)) for map in db_maps]

        # Fetch all unique layer IDs from all maps in the chain
        all_layer_ids = set()
        for db_map in db_maps:
            if db_map.layers:
                all_layer_ids.update(db_map.layers)

        # Fetch all layer data
        layers_by_id = {}
        if all_layer_ids:
            db_layers = await conn.fetch(
                """
                SELECT layer_id, owner_uuid, name, s3_key, type,
                       postgis_connection_id, postgis_query, metadata, bounds, geometry_type,
                       feature_count, size_bytes, source_map_id, created_on, last_edited
                FROM map_layers
                WHERE layer_id = ANY($1)
                """,
                list(all_layer_ids),
            )
            for layer_row in db_layers:
                layer_dict = dict(layer_row)
                layer_dict["metadata_json"] = layer_dict.pop("metadata")
                layers_by_id[layer_dict["layer_id"]] = MapLayer(**layer_dict)

        # Fetch all messages from the conversation if conversation_id is provided
        db_messages = []
        if conversation_id is not None:
            conv_ok = await conn.fetchrow(
                """
                SELECT 1
                FROM conversations c
                WHERE c.id = $1
                  AND c.owner_uuid = $2
                  AND c.project_id = $3
                  AND c.soft_deleted_at IS NULL
                """,
                conversation_id,
                session.get_user_id(),
                map.project_id,
            )
            if not conv_ok:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found",
                )

            db_messages = await conn.fetch(
                """
                SELECT ccm.*
                FROM chat_completion_messages ccm
                WHERE ccm.conversation_id = $1
                ORDER BY ccm.created_at ASC
                """,
                conversation_id,
            )
    # Group messages by map_id
    # some maps may have no messages
    messages_by_map: defaultdict[str, List[SanitizedMessage]] = defaultdict(list)
    for msg in db_messages:
        msg_dict = dict(msg)
        # Parse message_json when using raw asyncpg
        msg_dict["message_json"] = json.loads(msg_dict["message_json"])
        cc_message = MundiChatCompletionMessage(**msg_dict)
        if cc_message.message_json["role"] == "system":
            continue
        sanitized_payload = convert_mundi_message_to_sanitized(cc_message)

        messages_by_map[sanitized_payload.map_id].append(sanitized_payload)

    # Create MapNode objects with layer diffs
    nodes: List[MapNode] = []
    for i, map in enumerate(db_maps):
        # Calculate diff from previous map
        diff_from_previous = None
        if i > 0:
            prev_map = db_maps[i - 1]
            prev_layers = set(prev_map.layers or [])
            current_layers = set(map.layers or [])

            added_layer_ids = current_layers - prev_layers
            removed_layer_ids = prev_layers - current_layers

            added_layers = [
                LayerInfo.from_map_layer(layers_by_id[layer_id])
                for layer_id in added_layer_ids
                if layer_id in layers_by_id
            ]
            removed_layers = [
                LayerInfo.from_map_layer(layers_by_id[layer_id])
                for layer_id in removed_layer_ids
                if layer_id in layers_by_id
            ]

            diff_from_previous = LayerDiff(
                added_layers=added_layers, removed_layers=removed_layers
            )

        node = MapNode(
            map_id=map.id,
            messages=messages_by_map[map.id],
            fork_reason=map.fork_reason,
            created_on=map.created_on.isoformat(),
            diff_from_previous=diff_from_previous,
        )
        nodes.append(node)

    return MapTreeResponse(project_id=project_id, tree=nodes)


class RecoverableToolCallError(Exception):
    def __init__(self, message: str, tool_call_id: str):
        self.message = message
        self.tool_call_id = tool_call_id
        super().__init__(message)


def is_layer_id(s: str) -> bool:
    return isinstance(s, str) and s[0] == "L" and len(s) == 12


def check_postgis_readonly(plan: dict):
    if plan.get("Node Type") == "ModifyTable":
        raise ValueError("Write operations not allowed")
    for child in plan.get("Plans", []):
        check_postgis_readonly(child)


async def run_geoprocessing_tool(
    tool_call: ChatCompletionToolMessageParam,
    conn,
    user_id: str,
    map_id: str,
    conversation_id: int,
):
    function_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    all_tools = get_tools()
    for tool in all_tools:
        if function_name == tool["function"]["name"]:
            tool_def = tool
            break
    assert tool_def is not None

    algorithm_id = tool_def["function"]["name"].replace("_", ":")

    mapped_args = tool_args.copy()
    mapped_args["map_id"] = map_id
    mapped_args["user_uuid"] = user_id

    with tracer.start_as_current_span(f"geoprocessing.{algorithm_id}") as span:
        try:
            async with (
                kue_ephemeral_action(
                    conversation_id, f"QGIS running {algorithm_id}..."
                ),
                async_conn("get_layer_for_geoprocessing") as conn,
            ):
                input_params = {}
                input_urls = {}

                for key, val in mapped_args.items():
                    if key == "OUTPUT":
                        continue
                    elif is_layer_id(val):
                        # Get OGR source for any layer type (S3, remote URL, PostGIS)
                        try:
                            layer_row = await conn.fetchrow(
                                """
                                SELECT *
                                FROM map_layers
                                WHERE layer_id = $1 AND owner_uuid = $2
                                """,
                                val,
                                user_id,
                            )
                            if not layer_row:
                                raise HTTPException(404, f"Layer {val} not found")
                            layer = MapLayer(**dict(layer_row))

                            ogr_source_context = await layer.get_ogr_source(
                                never_return_local_file=True
                            )
                            async with ogr_source_context as ogr_source:
                                input_urls[key] = ogr_source
                        except Exception:
                            raise RecoverableToolCallError(
                                f"Layer {val} could not be accessed for geoprocessing",
                                tool_call.id,
                            )
                    else:
                        input_params[key] = str(val)

                map_data = await conn.fetchrow(
                    """
                    SELECT project_id FROM user_mundiai_maps
                    WHERE id = $1
                    """,
                    map_id,
                )
                project_id = map_data["project_id"]

                output_layer_mappings = {}

                # Generate presigned PUT URLs for all output parameters
                s3_client = await get_async_s3_client()
                bucket_name = get_bucket_name()
                output_presigned_put_urls = {}

                # Generate output layer ID and S3 key for this output
                output_layer_id = generate_id(prefix="L")
                # Determine file extension based on tool description
                tool_description = tool_def["function"]["description"].lower()
                vector_count = tool_description.count("vector")
                raster_count = tool_description.count("raster")

                if vector_count > raster_count:
                    file_extension = ".fgb"
                    layer_type = "vector"
                else:
                    file_extension = ".tif"
                    layer_type = "raster"

                output_s3_key = (
                    f"uploads/{user_id}/{project_id}/{output_layer_id}{file_extension}"
                )

                # Generate presigned PUT URL for this output
                output_presigned_url = await s3_client.generate_presigned_url(
                    "put_object",
                    Params={
                        "Bucket": bucket_name,
                        "Key": output_s3_key,
                        "ContentType": "application/x-www-form-urlencoded",
                    },
                    ExpiresIn=3600,  # 1 hour
                )

                output_presigned_put_urls["OUTPUT"] = output_presigned_url
                output_layer_mappings["OUTPUT"] = {
                    "layer_id": output_layer_id,
                    "s3_key": output_s3_key,
                    "layer_type": layer_type,
                    "file_extension": file_extension,
                }

                qgis_request = {
                    "algorithm_id": algorithm_id,
                    "qgis_inputs": input_params,
                    "output_presigned_put_urls": output_presigned_put_urls,
                    "input_urls": input_urls,
                }

                # Call QGIS processing service
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        os.environ["QGIS_PROCESSING_URL"] + "/run_qgis_process",
                        json=qgis_request,
                        timeout=30.0,
                    )

                if response.status_code != 200:
                    return {
                        "status": "error",
                        "error": f"QGIS processing failed: {response.status_code} - {response.text}",
                        "algorithm_id": algorithm_id,
                    }

                qgis_result = response.json()

                # Check if all layer outputs were successfully uploaded
                upload_results = qgis_result.get("upload_results", {})

                for param_name in output_layer_mappings.keys():
                    if (
                        param_name not in upload_results
                        or not upload_results[param_name]["uploaded"]
                    ):
                        return {
                            "status": "error",
                            "error": f"QGIS processing completed but output file {param_name} was not uploaded successfully",
                            "qgis_result": qgis_result,
                        }

                # Create new layers from the uploaded results
                created_layers = []

                for param_name, layer_info in output_layer_mappings.items():
                    # Download the output file from S3
                    downloaded_file = await s3_client.get_object(
                        Bucket=bucket_name, Key=layer_info["s3_key"]
                    )
                    file_content = await downloaded_file["Body"].read()

                    # Create an UploadFile-like object
                    filename = f"{layer_info['layer_id']}{layer_info['file_extension']}"
                    upload_file = UploadFile(
                        filename=filename,
                        file=io.BytesIO(file_content),
                    )

                    upload_result: InternalLayerUploadResponse = (
                        await internal_upload_layer(
                            map_id=map_id,
                            file=upload_file,
                            layer_name=filename,
                            add_layer_to_map=False,
                            user_id=user_id,
                            project_id=project_id,
                        )
                    )

                    created_layers.append(
                        {
                            "param_name": param_name,
                            "layer_id": upload_result.id,
                            "layer_name": filename,
                            "layer_type": layer_info["layer_type"],
                        }
                    )

                # Prepare the response
                result = {
                    "status": "success",
                    "message": f"{function_name} completed successfully",
                    "algorithm_id": algorithm_id,
                    "qgis_result": qgis_result,
                    "created_layers": created_layers,
                }

                # Add instructions about available layers
                if created_layers:
                    layer_names = [layer["layer_name"] for layer in created_layers]
                    layer_ids = [layer["layer_id"] for layer in created_layers]
                    result["kue_instructions"] = (
                        f"New layers available: {', '.join(layer_names)} "
                        f"(IDs: {', '.join(layer_ids)}), not added to map. "
                        'Use "add_layer_to_map" with the layer_id and descriptive new_name for layers that should be visible to the user. DO NOT include feature count or CRS in name, those are already visible to the user.'
                    )

                return result

        except UnsupportedAlgorithmError as e:
            return {
                "status": "error",
                "error": f"Unsupported algorithm parameter: {str(e)}",
            }
        except InvalidInputFormatError as e:
            return {
                "status": "error",
                "error": f"Invalid input format: {str(e)}",
            }
        except Exception as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.set_attribute("error.traceback", traceback.format_exc())
            return {
                "status": "error",
                "error": "Unexpected error running geoprocessing, this is likely a Mundi bug.",
                "algorithm_id": algorithm_id,
            }


async def process_chat_interaction_task(
    request: Request,  # Keep request for get_map_messages
    map_id: str,
    session: UserContext,  # Pass session for auth
    user_id: str,  # Pass user_id directly
    chat_args: ChatArgsProvider,
    map_state: MapStateProvider,
    conversation: Conversation,
    system_prompt_provider: SystemPromptProvider,
    connection_manager: PostgresConnectionManager,
    pydantic_tool_calls: PydanticToolRegistry,
):
    # kick it off with a quick sleep, to detach from the event loop blocking /send
    await asyncio.sleep(0.1)

    async with async_conn("process_chat_interaction_task") as conn:

        async def add_chat_completion_message(
            message: Union[ChatCompletionMessage, ChatCompletionMessageParam],
        ):
            message_dict = (
                message.model_dump() if isinstance(message, BaseModel) else message
            )

            await conn.execute(
                """
                INSERT INTO chat_completion_messages
                (map_id, sender_id, message_json, conversation_id)
                VALUES ($1, $2, $3, $4)
                """,
                map_id,
                user_id,
                json.dumps(message_dict),
                conversation.id,
            )

        with tracer.start_as_current_span("app.process_chat_interaction") as span:
            for i in range(25):
                # Check if the message processing has been cancelled
                if redis.get(f"messages:{map_id}:cancelled"):
                    redis.delete(f"messages:{map_id}:cancelled")
                    break

                # Refresh messages to include any new system messages we just added
                with tracer.start_as_current_span("kue.fetch_messages"):
                    updated_messages_response = await get_all_conversation_messages(
                        conversation.id, session
                    )

                openai_messages = [
                    msg.message_json for msg in updated_messages_response
                ]

                with tracer.start_as_current_span("kue.fetch_unattached_layers"):
                    unattached_layers = await conn.fetch(
                        """
                        SELECT ml.layer_id, ml.created_on, ml.last_edited, ml.type, ml.name
                        FROM map_layers ml
                        WHERE ml.owner_uuid = $1
                        AND NOT EXISTS (
                            SELECT 1 FROM user_mundiai_maps m
                            WHERE ml.layer_id = ANY(m.layers) AND m.owner_uuid = $2
                        )
                        ORDER BY ml.created_on DESC
                        LIMIT 10
                        """,
                        user_id,
                        user_id,
                    )

                layer_enum = {}
                for layer in unattached_layers:
                    layer_name = (
                        layer.get("name") or f"Unnamed Layer ({layer['layer_id'][:8]})"
                    )
                    layer_enum[layer["layer_id"]] = (
                        f"{layer_name} (type: {layer.get('type', 'unknown')}, created: {layer['created_on']})"
                    )

                client = get_openai_client(request)

                tools_payload = [
                    {
                        "type": "function",
                        "function": {
                            "name": "new_layer_from_postgis",
                            "description": "Creates a new layer, given a PostGIS connection and query, and adds it to the map so the user can see it. Layer will automatically pull data from PostGIS. Modify style using the set_layer_style tool.",
                            "strict": True,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "postgis_connection_id": {
                                        "type": "string",
                                        "description": "Unique PostGIS connection ID used as source",
                                    },
                                    "query": {
                                        "type": "string",
                                        "description": "SQL query to execute against PostGIS database for this layer, should list fetched columns for attributes that might be used for symbology (+ shape geometry). This query MUST alias the geometry column as 'geom' AND have a unique numeric id aliased as 'id'. Include newlines+spaces at ~55 column wrap",
                                    },
                                    "layer_name": {
                                        "type": "string",
                                        "description": "Sets a human-readable name for this layer. This name will appear in the layer list/legend for the user.",
                                    },
                                },
                                "required": [
                                    "postgis_connection_id",
                                    "query",
                                    "layer_name",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                      {
                        "type": "function",
                        "function": {
                            "name": "bloom_study",
                            "description": "Analyzes the most recent almond bloom using the Enhanced Bloom Index (EBI = (R + G + B) / ((G / B) × (R − B + ε))), an index developed to detect peak flowering in almond orchards, and predicts upcoming bloom periods near a given location using the Carbohydrate–Temperature (C–T) model, which links almond metabolic activity to current temperature dynamics under changing climate conditions.",
                            "strict": True,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "address": {
                                        "type": "string",
                                        "description": "Street address or location to analyze (e.g., '123 Main St, City, State').",
                                    },
                                },
                                "required": ["address"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "pest_detection",
                            "description": "Displays crop health using the DCBVI index (derived from Landsat 8 imagery). Green areas indicate healthy crops, red areas indicate stress. The model estimates the likelihood of dusky cotton bug infestation and returns a true/false result based on the confidence of the analysis.",
                            "strict": True,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "address": {
                                        "type": "string",
                                        "description": "Street address or location to analyze (e.g., '123 Main St, City, State').",
                                    },
                                },
                                "required": ["address"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "add_layer_to_map",
                            "description": "Shows a newly created or existing unattached layer on the user's current map and layer list. Use this after a geoprocessing step that creates a layer, or if the user asks to see an existing layer that isn't currently on their map.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "layer_id": {
                                        "type": "string",
                                        "description": "The ID of the layer to add to the map. Choose from available unattached layers.",
                                        "enum": list(layer_enum.keys())
                                        if layer_enum
                                        else ["NO_UNATTACHED_LAYERS"],
                                    },
                                    "new_name": {
                                        "type": "string",
                                        "description": "Sets a new human-readable name for this layer. This name will appear in the layer list/legend for the user.",
                                    },
                                },
                                "required": ["layer_id", "new_name"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "set_layer_style",
                            "description": "Creates a new style for a layer with MapLibre JSON layers and immediately applies it as the active style",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "layer_id": {
                                        "type": "string",
                                        "description": "The ID of the layer to create and apply a style for",
                                    },
                                    "maplibre_json_layers_str": {
                                        "type": "string",
                                        "description": 'JSON string of MapLibre layer objects. Example: [{"id": "LZJ5RmuZr6qN-line", "type": "line", "source": "LZJ5RmuZr6qN", "paint": {"line-color": "#1E90FF"}}]',
                                    },
                                },
                                "required": ["layer_id", "maplibre_json_layers_str"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "query_duckdb_sql",
                            "description": "Execute a SQL query against vector layer data using DuckDB. Use query_postgis_database for layers created from PostGIS connections instead.",
                            "strict": True,
                            "parameters": {
                                "type": "object",
                                "required": ["layer_ids", "sql_query", "head_n_rows"],
                                "properties": {
                                    "layer_ids": {
                                        "type": "array",
                                        "description": "Load these vector layer IDs as tables",
                                        "items": {"type": "string"},
                                    },
                                    "sql_query": {
                                        "type": "string",
                                        "description": "DuckDB-flavored SELECT ... SQL query. Include newlines+spaces at ~55 column wrap for readability e.g. SELECT name_en,county\n    FROM LCH6Na2SBvJr\n    ORDER BY id",
                                    },
                                    "head_n_rows": {
                                        "type": "number",
                                        "description": "Truncate result to n rows (increase gingerly, MUST specify returned columns), n=20 is good",
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "query_postgis_database",
                            "description": "Execute SQL queries on connected PostgreSQL/PostGIS databases. Use for data analysis, spatial queries, and exploring database tables. The query MUST include a LIMIT clause with a value less than 1000.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "postgis_connection_id": {
                                        "type": "string",
                                        "description": "User's PostGIS connection ID to query against",
                                    },
                                    "sql_query": {
                                        "type": "string",
                                        "description": "SQL query to execute. Use newlines+spaces at ~55 column wrap. Examples: 'SELECT COUNT(*) FROM table_name', 'SELECT * FROM spatial_table LIMIT 10', 'SELECT column_name FROM information_schema.columns WHERE table_name = \"my_table\"'. Use standard SQL syntax.",
                                    },
                                },
                                "required": ["postgis_connection_id", "sql_query"],
                                "additionalProperties": False,
                            },
                        },
                    },
                ]

                # add pydantic-defined tools to the payload
                for name, (fn, arg_model, _mundi_model) in pydantic_tool_calls.items():
                    tools_payload.append(tool_from_pyd(fn, arg_model))

                all_tools = get_tools()
                tools_payload.extend(all_tools)
                geoprocessing_function_names = [
                    tool["function"]["name"] for tool in all_tools
                ]

                if not layer_enum:
                    add_layer_tool = next(
                        tool
                        for tool in tools_payload
                        if tool["function"]["name"] == "add_layer_to_map"
                    )
                    add_layer_tool["function"]["parameters"]["properties"][
                        "layer_id"
                    ].pop("enum", None)

                # Replace the thinking ephemeral updates with context manager
                async with kue_ephemeral_action(conversation.id, "Kue is thinking..."):
                    chat_completions_args = await chat_args.get_args(
                        user_id, "send_map_message_async"
                    )
                    with tracer.start_as_current_span(
                        "kue.openai.chat.completions.create"
                    ):
                        # chat.completions.create fails for bad messages and tools, so
                        # if we have orphaned tool calls then we'll get an error - but not
                        # handling it properly makes for a horrible user experience
                        try:
                            response = await client.chat.completions.create(
                                **chat_completions_args,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": system_prompt_provider.get_system_prompt(),
                                    }
                                ]
                                + openai_messages,
                                tools=tools_payload if tools_payload else None,
                                tool_choice="auto" if tools_payload else None,
                            )
                        except APIError as e:
                            if e.code == "context_length_exceeded":
                                await kue_notify_error(
                                    conversation.id,
                                    "Maximum context length for LLM has been reached. Please create a new chat to continue using the chat feature.",
                                )
                            else:
                                await kue_notify_error(
                                    conversation.id,
                                    "Error connecting to LLM. If trying again doesn't work, create a new chat in the top right to reset the chat history.",
                                )
                            span.set_status(
                                trace.Status(trace.StatusCode.ERROR, str(e))
                            )
                            span.set_attribute(
                                "error.traceback", traceback.format_exc()
                            )
                            break
                        except Exception as e:
                            await kue_notify_error(
                                conversation.id,
                                "Error connecting to LLM. This is probably a bug with Mundi, please open a new issue on GitHub.",
                            )
                            span.set_status(
                                trace.Status(trace.StatusCode.ERROR, str(e))
                            )
                            span.set_attribute(
                                "error.traceback", traceback.format_exc()
                            )
                            break
                assistant_message: ChatCompletionMessageParam = response.choices[
                    0
                ].message

                # after chat completions is a pretty common spot to get a cancelled message
                if redis.get(f"messages:{map_id}:cancelled"):
                    redis.delete(f"messages:{map_id}:cancelled")
                    break

                # Store the assistant message in the database
                await add_chat_completion_message(assistant_message)

                # If no tool calls, break
                if not assistant_message.tool_calls:
                    break

                # Fetch project_id for this map once for all tool calls
                async with async_conn("tool.project_id_for_map") as proj_conn:
                    row = await proj_conn.fetchrow(
                        "SELECT project_id FROM user_mundiai_maps WHERE id = $1",
                        map_id,
                    )
                    assert row is not None
                    current_project_id: str = row["project_id"]

                # Process each tool call returned by the assistant

                for tool_call in assistant_message.tool_calls:
                    tool_call: ChatCompletionMessageToolCall = tool_call
                    function_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_result = {}

                    if function_name in pydantic_tool_calls:
                        fn, ArgModel, MundiModel = pydantic_tool_calls[function_name]
                        try:
                            parsed_args = ArgModel(**(tool_args or {}))

                        except Exception as e:
                            tool_result = {
                                "status": "error",
                                "error": f"Invalid arguments for {function_name}: {e}",
                            }
                            await add_chat_completion_message(
                                ChatCompletionToolMessageParam(
                                    role="tool",
                                    tool_call_id=tool_call.id,
                                    content=json_dumps_safe(tool_result),
                                ),
                            )
                            continue

                        try:
                            mundi_args = MundiModel(
                                user_uuid=user_id,
                                conversation_id=conversation.id,
                                map_id=map_id,
                                project_id=current_project_id,
                                session=session,
                            )
                            # Execute tool (all tools are async)
                            tool_result = await fn(parsed_args, mundi_args)

                        except Exception:
                            tool_result = {
                                "status": "error",
                                "error": "Tool execution failed. Please try again or adjust the inputs.",
                            }

                        await add_chat_completion_message(
                            ChatCompletionToolMessageParam(
                                role="tool",
                                tool_call_id=tool_call.id,
                                content=json_dumps_safe(tool_result),
                            ),
                        )
                        continue

                    span.add_event(
                        "kue.tool_call_started",
                        {"tool_name": function_name},
                    )

                    with tracer.start_as_current_span(f"kue.{function_name}") as span:

                        if function_name == "bloom_study":
                            address = tool_args.get("address")
                           

                            async with kue_ephemeral_action(
                                conversation.id,
                                "Analyzing bloom events...",
                            ):
                                if not address or not isinstance(address, str) or not address.strip():
                                    print("Invalid or missing address")
                                    tool_result = {
                                        "status": "error",
                                        "error": "Missing or invalid 'address'. Provide a non-empty text address.",
                                    }
                                else:
                                    print(f"Geocoding address: {address}")
                                    cleaned_address = address.strip()

                                    coords = await geocode_address(address)
                                    if not coords:
                                        tool_result = {
                                            "status": "error",
                                            "error": "Could not geocode address to coordinates",
                                        }
                                    else:
                                        latitude, longitude = coords
                                        print(f"Geocoded address to {latitude}, {longitude}")
                                        try:
                                            prediction_row = await conn.fetchrow(
                                                """
                                                  SELECT
                                                    predicted_bloom_start ,
                                                    predicted_bloom_peak,
                                                    confidence
                                                FROM bloom_predictions
                                                WHERE latitude = $1
                                                  AND longitude = $2
                                                  ORDER BY created_at DESC
                                                LIMIT 1
                                                """,
                                                latitude,
                                                longitude,
                                            )

                                            print(f"Prediction row: {prediction_row}")

                                            observation_row = await conn.fetchrow(
                                                """
                                                SELECT
                                                    date_of_max_ebi,
                                                    ebi_value,
                                                    image_url
                                                FROM bloom_observations
                                                WHERE latitude = $1
                                                  AND longitude = $2
                                                ORDER BY created_at DESC
                                                LIMIT 1
                                                """,
                                                latitude,
                                                longitude,
                                            )

                                            print(f"Observation row: {observation_row}")

                                            prediction = (
                                                {
                                                    "prediction_bloom_start": prediction_row["predicted_bloom_start"],
                                                    "prediction_bloom_peak": prediction_row["predicted_bloom_peak"],
                                                    "confidence": prediction_row["confidence"],
                                                }
                                                if prediction_row
                                                else None
                                                                                                    
                                            )
                                            print(f"Prediction: {prediction}")
                                            observation = (
                                                {
                                                    "date_of_max_ebi": observation_row["date_of_max_ebi"],
                                                    "ebi_value": observation_row["ebi_value"],
                                                    "image_url": observation_row["image_url"],
                                                }
                                                if observation_row
                                                else None
                                            )
                                            print(f"Observation: {observation}")

                                            # Fallback: if DB has no data, call API endpoints (POST)
                                            try:
                                                if prediction is None or observation is None:
                                                    transport = httpx.ASGITransport(app=request.app)
                                                    async with httpx.AsyncClient(transport=transport, base_url="http://app") as client:
                                                        # Call POST /api/bloom-prediction if missing
                                                        if prediction is None:
                                                            try:
                                                                pred_resp = await client.post(
                                                                    "/api/bloom-prediction",
                                                                    json={"latitude": latitude, "longitude": longitude},
                                                                )
                                                                if pred_resp.status_code == 200:
                                                                    pred_json = pred_resp.json()
                                                                    prediction = {
                                                                        "prediction_bloom_start": pred_json.get("predicted_bloom_start"),
                                                                        "prediction_bloom_peak": pred_json.get("predicted_bloom_peak"),
                                                                        "confidence": pred_json.get("confidence"),
                                                                    }
                                                                    print(f"Prediction fallback via endpoint: {prediction}")
                                                                else:
                                                                    print(f"Prediction endpoint error: {pred_resp.status_code} {pred_resp.text}")
                                                            except Exception as e:
                                                                print(f"Prediction endpoint call failed: {e}")

                                                        # Call POST /api/bloom-detection if missing
                                                        if observation is None:
                                                            try:
                                                                obs_resp = await client.post(
                                                                    "/api/bloom-detection",
                                                                    json={"latitude": latitude, "longitude": longitude},
                                                                )
                                                                if obs_resp.status_code == 200:
                                                                    obs_json = obs_resp.json()
                                                                    observation = {
                                                                        "date_of_max_ebi": obs_json.get("date_of_max_ebi"),
                                                                        "ebi_value": obs_json.get("ebi_value"),
                                                                        "image_url": obs_json.get("image_url"),
                                                                    }
                                                                    print(f"Observation fallback via endpoint: {observation}")
                                                                else:
                                                                    print(f"Observation endpoint error: {obs_resp.status_code} {obs_resp.text}")
                                                            except Exception as e:
                                                                print(f"Observation endpoint call failed: {e}")
                                            except Exception as e:
                                                print(f"Bloom endpoint fallback wrapper failed: {e}")

                                        except Exception as e:
                                            tool_result = {
                                                "status": "error",
                                                "error": f"Failed to load bloom data: {str(e)}",
                                            }

                                # If we have a georeferenced image URL, add it as a raster layer to the map
                                try:
                                    ebi_url = None
                                    layer_add_result = None
                                    added_layer_name = None
                                    if observation and isinstance(observation, dict):
                                        ebi_url = observation.get("image_url")

                                    if ebi_url and isinstance(ebi_url, str) and ebi_url.startswith("http"):
                                        # Name the layer using the observation date if available
                                        layer_name = "Bloom EBI"
                                        try:
                                            if observation.get("date_of_max_ebi"):
                                                layer_name = f"Bloom EBI {observation['date_of_max_ebi']}"
                                        except Exception:
                                            pass
                                        added_layer_name = layer_name

                                        # Add via the remote-layer endpoint so the DAG is preserved
                                        async with kue_ephemeral_action(
                                            conversation.id,
                                            "Adding bloom raster layer to map...",
                                            update_style_json=True,
                                        ):
                                            transport = httpx.ASGITransport(app=request.app)
                                            async with httpx.AsyncClient(transport=transport, base_url="http://app") as client:
                                                resp = await client.post(
                                                    f"/api/maps/{map_id}/layers/remote",
                                                    json={
                                                        "url": ebi_url,
                                                        "name": layer_name,
                                                        "source_type": "raster",
                                                        "add_layer_to_map": True,
                                                    },
                                                )
                                            if resp.status_code == 200:
                                                try:
                                                    layer_add_result = resp.json()
                                                except Exception:
                                                    layer_add_result = None
                                            else:
                                                print(
                                                    f"Failed to add bloom layer to map: {resp.status_code} {getattr(resp,'text', '')}"
                                                )
                                    else:
                                        print("No valid bloom image URL to add as a layer.")
                                except Exception as e:
                                    print(f"Exception while adding bloom layer: {e}")

                                kue_instructions = (
                                            "Explain this result in 2–4 sentences. If a raster layer was added, mention its name and suggest zooming to it. "
                                            "Clarifies that yellow areas indicate blooming almond orchards, analyzed using the Enhanced Bloom Index (EBI = (R + G + B) / ((G / B) × (R − B + ε))), an index developed to detect peak flowering in almond crops. It also predicts upcoming bloom periods near a given location using the Carbohydrate–Temperature (C–T) model, which relates almond metabolic activity to current temperature patterns influenced by climate change. Using Landsat 8 Datat"
                                            "Avoid internal IDs."
                                        )

                                tool_result = {
                                    "status": "success",
                                    "address": cleaned_address,
                                    "latitude": latitude,
                                    "longitude": longitude,
                                    "prediction": prediction,
                                    "observation": observation,
                                    # Surface layer addition details for the frontend to optionally switch maps
                                    **({
                                        "layer_upload": layer_add_result,
                                        "dag_child_map_id": layer_add_result.get("dag_child_map_id"),
                                        "dag_parent_map_id": layer_add_result.get("dag_parent_map_id"),
                                    } if layer_add_result else {}),
                                    "message": (
                                                    "No recent prediction or observation found for these coordinates"
                                                    if not prediction and not observation
                                                    else "Retrieved latest prediction and/or observation"
                                                ),
                                            }

                                # Provide clearer user-facing explanation and instructions
                                try:
                                    pred_start = (
                                        prediction.get("prediction_bloom_start") if isinstance(prediction, dict) else None
                                    )
                                    pred_peak = (
                                        prediction.get("prediction_bloom_peak") if isinstance(prediction, dict) else None
                                    )
                                    pred_conf = (
                                        prediction.get("confidence") if isinstance(prediction, dict) else None
                                    )
                                    obs_date = (
                                        observation.get("date_of_max_ebi") if isinstance(observation, dict) else None
                                    )

                                    def _fmt(v):
                                        try:
                                            return str(v)
                                        except Exception:
                                            return None

                                    msg_parts = []
                                    if pred_start or pred_peak:
                                        s = "Bloom forecast"
                                        if pred_start:
                                            s += f" starts {_fmt(pred_start)}"
                                        if pred_peak:
                                            s += f", peaks {_fmt(pred_peak)}"
                                        if isinstance(pred_conf, (int, float)):
                                            s += f" (confidence {round(pred_conf*100) if pred_conf <= 1 else round(pred_conf)}%)."
                                        else:
                                            s += "."
                                        msg_parts.append(s)
                                    if obs_date:
                                        msg_parts.append(f"Recent EBI maximum on {_fmt(obs_date)}.")
                                    explanation = " ".join(msg_parts) if msg_parts else (
                                        f"No recent bloom data available for {cleaned_address}."
                                    )

                                    tool_result["message"] = explanation
                                    tool_result["kue_instructions"] = (
                                        "Explain this result in 2–4 sentences. If a raster layer was added, mention its name and suggest zooming to it. "
                                        "Clarify that yellow areas indicate blooming almond orchards, analyzed using the Enhanced Bloom Index (EBI = (R + G + B) / ((G / B) × (R − B + ε))) to detect peak flowering. "
                                        "Also note that the tool predicts upcoming bloom periods near the location using the Carbohydrate–Temperature (C–T) model, which relates almond metabolic activity to current temperature patterns influenced by climate change. "
                                        "Avoid internal IDs."
                                    )
                                    if layer_add_result and added_layer_name:
                                        tool_result["added_layer_name"] = added_layer_name
                                except Exception:
                                    pass
                                
                                print(f"Tool result: {tool_result}")
                                            


                            await add_chat_completion_message(
                                ChatCompletionToolMessageParam(
                                    role="tool",
                                    tool_call_id=tool_call.id,
                                    content=json_dumps_safe(tool_result),
                                )
                            )

                        elif function_name == "pest_detection":
                            address = tool_args.get("address")
                           

                            async with kue_ephemeral_action(
                                conversation.id,
                                "Detecting pests events...",
                            ):
                                if not address or not isinstance(address, str) or not address.strip():
                                    print("Invalid or missing address")
                                    tool_result = {
                                        "status": "error",
                                        "error": "Missing or invalid 'address'. Provide a non-empty text address.",
                                    }
                                else:
                                    print(f"Geocoding address: {address}")
                                    cleaned_address = address.strip()

                                    coords = await geocode_address(address)
                                    if not coords:
                                        tool_result = {
                                            "status": "error",
                                            "error": "Could not geocode address to coordinates",
                                        }
                                    else:
                                        latitude, longitude = coords
                                        print(f"Geocoded address to {latitude}, {longitude}")

                                        # Ensure observation is defined even on failures
                                        observation = None

                                        # Query the pest detection endpoint
                                        try:
                                            transport = httpx.ASGITransport(app=request.app)
                                            async with httpx.AsyncClient(transport=transport, base_url="http://app") as client:
                                                try:
                                                    obs_resp = await client.post(
                                                        "/api/pest-detection",
                                                        json={"latitude": latitude, "longitude": longitude},
                                                    )
                                                    if obs_resp.status_code == 200:
                                                        obs_json = obs_resp.json()
                                                        observation = {
                                                            "detected_pest": obs_json.get("detected_pest"),
                                                            "confidence_score": obs_json.get("confidence_score"),
                                                            "image_url": obs_json.get("image_url"),
                                                        }
                                                        print(f"Pest Observation via endpoint: {observation}")
                                                    else:
                                                        print(
                                                            f"Observation endpoint error: {obs_resp.status_code} {obs_resp.text}"
                                                        )
                                                except Exception as e:
                                                    print(f"Observation endpoint call failed: {e}")
                                        except Exception as e:
                                            print(f"Pest endpoint fallback wrapper failed: {e}")


                                        try:
                                            pest_url = None
                                            layer_add_result = None
                                            added_layer_name = None
                                            if observation and isinstance(observation, dict):
                                                pest_url = observation.get("image_url")

                                            if pest_url and isinstance(pest_url, str) and pest_url.startswith("http"):
                                                # Give the raster a helpful name
                                                added_layer_name = (
                                                    f"DCBVI Pest Stress ({observation.get('confidence_score')})"
                                                    if observation and observation.get("confidence_score") is not None
                                                    else "DCBVI Pest Stress"
                                                )

                                                # Add via the remote-layer endpoint so the DAG is preserved
                                                async with kue_ephemeral_action(
                                                    conversation.id,
                                                    "Adding DCBVI raster layer to map...",
                                                    update_style_json=True,
                                                ):
                                                    transport = httpx.ASGITransport(app=request.app)
                                                    async with httpx.AsyncClient(transport=transport, base_url="http://app") as client:
                                                        resp = await client.post(
                                                            f"/api/maps/{map_id}/layers/remote",
                                                            json={
                                                                "url": pest_url,
                                                                "name": added_layer_name,
                                                                "source_type": "raster",
                                                                "add_layer_to_map": True,
                                                            },
                                                        )
                                                    if resp.status_code == 200:
                                                        try:
                                                            layer_add_result = resp.json()
                                                        except Exception:
                                                            layer_add_result = None
                                                    else:
                                                        print(
                                                            f"Failed to add pest layer to map: {resp.status_code} {getattr(resp,'text', '')}"
                                                        )
                                            else:
                                                print("No valid pest image URL to add as a layer.")
                                        except Exception as e:
                                            print(f"Exception while adding pest layer: {e}")

                                        # Build a user-facing explanation and guidance for the LLM
                                        detected = (
                                            observation.get("detected_pest") if isinstance(observation, dict) else None
                                        )
                                        conf = (
                                            observation.get("confidence_score") if isinstance(observation, dict) else None
                                        )
                                        conf_pct = None
                                        try:
                                            if isinstance(conf, (int, float)):
                                                conf_pct = round(conf * 100) if conf <= 1 else round(conf)
                                        except Exception:
                                            conf_pct = None

                                        if detected is False:
                                            explanation = (
                                                f"Pest stress likely near {cleaned_address}"
                                                + (f" (confidence {conf_pct}%)." if conf_pct is not None else ".")
                                            )
                                        elif detected is True:
                                            explanation = (
                                                f"No pest stress detected near {cleaned_address}"
                                                + (f" (confidence {conf_pct}%)." if conf_pct is not None else ".")
                                            )
                                        else:
                                            explanation = (
                                                f"No recent detection result available for {cleaned_address}."
                                            )

                                        kue_instructions = (
                                            "Explain this result in 2–4 sentences. If a raster layer was added, mention its name and suggest zooming to it. "
                                            "Clarify that red areas indicate vegetation distress and may correlate with pests like Oxycarenus laetus and greens with healthy vegetation. "
                                            "Avoid internal IDs."
                                        )

                                        tool_result = {
                                            "status": "success",
                                            "address": cleaned_address,
                                            "latitude": latitude,
                                            "longitude": longitude,
                                            "observation": observation,
                                            "layer_upload": layer_add_result,
                                            "added_layer_name": added_layer_name,
                                            "message": explanation,
                                            "kue_instructions": kue_instructions,
                                        }
                                        print(f"Tool result: {tool_result}")
                                            


                                await add_chat_completion_message(
                                    ChatCompletionToolMessageParam(
                                        role="tool",
                                        tool_call_id=tool_call.id,
                                        content=json_dumps_safe(tool_result),
                                    )
                                )   
                            
                                                     

                        elif function_name == "new_layer_from_postgis":
                            postgis_connection_id = tool_args.get(
                                "postgis_connection_id"
                            )
                            query = tool_args.get("query")
                            # seemingly innocuous but deeply insidious
                            query = query.rstrip().rstrip(";")
                            layer_name = tool_args.get("layer_name")

                            if not postgis_connection_id or not query:
                                tool_result = {
                                    "status": "error",
                                    "error": "Missing required parameters (postgis_connection_id or query).",
                                }
                            else:
                                # Verify the PostGIS connection exists and user has access
                                connection_result = await conn.fetchrow(
                                    """
                                    SELECT connection_uri FROM project_postgres_connections
                                    WHERE id = $1 AND user_id = $2
                                    """,
                                    postgis_connection_id,
                                    user_id,
                                )

                                if not connection_result:
                                    tool_result = {
                                        "status": "error",
                                        "error": f"PostGIS connection '{postgis_connection_id}' not found or you do not have access to it.",
                                    }
                                else:
                                    async with kue_ephemeral_action(
                                        conversation.id,
                                        "Adding layer from PostGIS...",
                                        update_style_json=True,
                                    ):
                                        try:
                                            # Use connection manager for PostGIS operations
                                            pg = await connection_manager.connect_to_postgres(
                                                postgis_connection_id
                                            )
                                            try:
                                                # 1. Make sure the SQL parsers and planners are happy
                                                explain_result = await pg.fetch(
                                                    f"EXPLAIN (FORMAT JSON) {query}"
                                                )

                                                # Parse the JSON string from QUERY PLAN
                                                query_plan = json.loads(
                                                    explain_result[0]["QUERY PLAN"]
                                                )
                                                check_postgis_readonly(
                                                    query_plan[0]["Plan"]
                                                )

                                                # Get column names using prepared statement
                                                prepared = await pg.prepare(
                                                    f"SELECT * FROM ({query}) AS sub LIMIT 1"
                                                )
                                                column_info = prepared.get_attributes()
                                                column_names = [
                                                    attr.name for attr in column_info
                                                ]

                                                # Make sure it returns a geometry column called geom and id
                                                if "geom" not in column_names:
                                                    raise ValueError(
                                                        "Query must return a column named 'geom'"
                                                    )
                                                if "id" not in column_names:
                                                    raise ValueError(
                                                        "Query must return a column named 'id'"
                                                    )

                                                attribute_names = [
                                                    name
                                                    for name in column_names
                                                    if name not in ["geom", "id"]
                                                ]

                                                # Calculate feature count, bounds, and geometry type for the PostGIS layer
                                                feature_count = None
                                                bounds = None
                                                geometry_type = None
                                                metadata_dict = {}

                                                # Calculate feature count
                                                count_result = await pg.fetchval(
                                                    f"SELECT COUNT(*) FROM ({query}) AS sub"
                                                )
                                                feature_count = (
                                                    int(count_result)
                                                    if count_result is not None
                                                    else None
                                                )

                                                # Detect geometry type for styling
                                                geometry_type_result = (
                                                    await pg.fetchrow(
                                                        f"""
                                                        SELECT ST_GeometryType(geom) as geom_type, COUNT(*) as count
                                                        FROM ({query}) AS sub
                                                        WHERE geom IS NOT NULL
                                                        GROUP BY ST_GeometryType(geom)
                                                        ORDER BY count DESC
                                                        LIMIT 1
                                                        """
                                                    )
                                                )

                                                if (
                                                    geometry_type_result
                                                    and geometry_type_result[
                                                        "geom_type"
                                                    ]
                                                ):
                                                    # Convert PostGIS geometry type to standard format
                                                    geometry_type = (
                                                        geometry_type_result[
                                                            "geom_type"
                                                        ]
                                                        .replace("ST_", "")
                                                        .lower()
                                                    )

                                                    # Calculate bounds with proper SRID handling
                                                    # ST_Extent returns BOX2D with SRID 0, so we need to set the SRID before transforming
                                                    bounds_result = await pg.fetchrow(
                                                        f"""
                                                        WITH extent_data AS (
                                                            SELECT
                                                                ST_Extent(geom) as extent_geom,
                                                                (SELECT ST_SRID(geom) FROM ({query}) AS sub2 WHERE geom IS NOT NULL LIMIT 1) as original_srid
                                                            FROM ({query}) AS sub
                                                            WHERE geom IS NOT NULL
                                                        )
                                                        SELECT
                                                            CASE
                                                                WHEN original_srid = 4326 THEN
                                                                    ST_XMin(extent_geom)
                                                                ELSE
                                                                    ST_XMin(ST_Transform(ST_SetSRID(extent_geom, original_srid), 4326))
                                                            END as xmin,
                                                            CASE
                                                                WHEN original_srid = 4326 THEN
                                                                    ST_YMin(extent_geom)
                                                                ELSE
                                                                    ST_YMin(ST_Transform(ST_SetSRID(extent_geom, original_srid), 4326))
                                                            END as ymin,
                                                            CASE
                                                                WHEN original_srid = 4326 THEN
                                                                    ST_XMax(extent_geom)
                                                                ELSE
                                                                    ST_XMax(ST_Transform(ST_SetSRID(extent_geom, original_srid), 4326))
                                                            END as xmax,
                                                            CASE
                                                                WHEN original_srid = 4326 THEN
                                                                    ST_YMax(extent_geom)
                                                                ELSE
                                                                    ST_YMax(ST_Transform(ST_SetSRID(extent_geom, original_srid), 4326))
                                                             END as ymax,
                                                             original_srid
                                                         FROM extent_data
                                                        WHERE extent_geom IS NOT NULL
                                                        """
                                                    )

                                                    if bounds_result and all(
                                                        v is not None
                                                        for v in bounds_result
                                                    ):
                                                        bounds = [
                                                            float(
                                                                bounds_result["xmin"]
                                                            ),
                                                            float(
                                                                bounds_result["ymin"]
                                                            ),
                                                            float(
                                                                bounds_result["xmax"]
                                                            ),
                                                            float(
                                                                bounds_result["ymax"]
                                                            ),
                                                        ]
                                                        # Capture original SRID into metadata if available
                                                        if (
                                                            "original_srid"
                                                            in bounds_result
                                                            and bounds_result[
                                                                "original_srid"
                                                            ]
                                                            is not None
                                                        ):
                                                            try:
                                                                metadata_dict[
                                                                    "original_srid"
                                                                ] = int(
                                                                    bounds_result[
                                                                        "original_srid"
                                                                    ]
                                                                )
                                                            except (
                                                                ValueError,
                                                                TypeError,
                                                            ):
                                                                pass
                                                else:
                                                    print(
                                                        "Warning: No geometry column found in PostGIS query"
                                                    )
                                            finally:
                                                await pg.close()

                                            # Generate a new layer ID
                                            layer_id = generate_id(prefix="L")

                                            # Generate default style if geometry type was detected
                                            maplibre_layers = None
                                            if geometry_type:
                                                try:
                                                    maplibre_layers = generate_maplibre_layers_for_layer_id(
                                                        layer_id, geometry_type
                                                    )
                                                    # PostGIS layers use MVT tiles, so source-layer is 'reprojectedfgb'
                                                    # This matches the expectation in the style generation function
                                                    print(
                                                        f"Generated default style for PostGIS layer {layer_id} with geometry type {geometry_type}"
                                                    )
                                                except Exception as e:
                                                    print(
                                                        f"Warning: Failed to generate default style for PostGIS layer: {str(e)}"
                                                    )
                                                    maplibre_layers = None

                                            # Create the layer in the database
                                            await conn.execute(
                                                """
                                                INSERT INTO map_layers
                                                (layer_id, owner_uuid, name, type, postgis_connection_id, postgis_query, metadata, feature_count, bounds, geometry_type, source_map_id, created_on, last_edited, postgis_attribute_column_list)
                                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, $12)
                                                """,
                                                layer_id,
                                                user_id,
                                                layer_name,
                                                "postgis",
                                                postgis_connection_id,
                                                query,
                                                json.dumps(metadata_dict),
                                                feature_count,
                                                bounds,
                                                geometry_type,
                                                map_id,
                                                attribute_names,
                                            )

                                            # Create default style in separate table if we have geometry type
                                            if maplibre_layers:
                                                style_id = generate_id(prefix="S")
                                                await conn.execute(
                                                    """
                                                    INSERT INTO layer_styles
                                                    (style_id, layer_id, style_json, created_by, created_on)
                                                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                                                    """,
                                                    style_id,
                                                    layer_id,
                                                    json.dumps(maplibre_layers),
                                                    user_id,
                                                )

                                                await conn.execute(
                                                    """
                                                    INSERT INTO map_layer_styles
                                                    (map_id, layer_id, style_id)
                                                    VALUES ($1, $2, $3)
                                                    """,
                                                    map_id,
                                                    layer_id,
                                                    style_id,
                                                )

                                            # layers may be NULL, not necessarily initialized to []
                                            await conn.execute(
                                                """
                                                UPDATE user_mundiai_maps
                                                SET layers = CASE
                                                    WHEN layers IS NULL THEN ARRAY[$1]
                                                    ELSE array_append(layers, $1)
                                                END
                                                WHERE id = $2 AND (layers IS NULL OR NOT ($1 = ANY(layers)))
                                                """,
                                                layer_id,
                                                map_id,
                                            )

                                            tool_result = {
                                                "status": "success",
                                                "message": f"PostGIS layer created successfully with ID: {layer_id} and added to map",
                                                "layer_id": layer_id,
                                                "query": query,
                                                "added_to_map": True,
                                            }
                                        except Exception as e:
                                            tool_result = {
                                                "status": "error",
                                                "error": f"Query validation failed: {str(e)}",
                                            }

                            await add_chat_completion_message(
                                ChatCompletionToolMessageParam(
                                    role="tool",
                                    tool_call_id=tool_call.id,
                                    content=json_dumps_safe(tool_result),
                                )
                            )
                        elif function_name == "add_layer_to_map":
                            layer_id_to_add = tool_args.get("layer_id")
                            new_name = tool_args.get("new_name")

                            async with kue_ephemeral_action(
                                conversation.id,
                                "Adding layer to map...",
                                update_style_json=True,
                            ):
                                layer_exists = await conn.fetchrow(
                                    """
                                    SELECT layer_id FROM map_layers
                                    WHERE layer_id = $1 AND owner_uuid = $2
                                    """,
                                    layer_id_to_add,
                                    user_id,
                                )

                                if not layer_exists:
                                    tool_result = {
                                        "status": "error",
                                        "error": f"Layer ID '{layer_id_to_add}' not found or you do not have permission to use it.",
                                    }
                                else:
                                    await conn.execute(
                                        """
                                        UPDATE map_layers SET name = $1 WHERE layer_id = $2
                                        """,
                                        new_name,
                                        layer_id_to_add,
                                    )

                                    await conn.execute(
                                        """
                                        UPDATE user_mundiai_maps
                                        SET layers = CASE
                                            WHEN layers IS NULL THEN ARRAY[$1]
                                            ELSE array_append(layers, $1)
                                        END
                                        WHERE id = $2 AND (layers IS NULL OR NOT ($1 = ANY(layers)))
                                        """,
                                        layer_id_to_add,
                                        map_id,
                                    )
                                    tool_result = {
                                        "status": f"Layer '{new_name}' (ID: {layer_id_to_add}) added to map '{map_id}'.",
                                        "layer_id": layer_id_to_add,
                                        "name": new_name,
                                    }

                                await add_chat_completion_message(
                                    ChatCompletionToolMessageParam(
                                        role="tool",
                                        tool_call_id=tool_call.id,
                                        content=json_dumps_safe(tool_result),
                                    )
                                )
                        elif function_name == "query_duckdb_sql":
                            layer_id = tool_args.get("layer_ids", [None])[
                                0
                            ]  # Use first layer or None
                            sql_query = tool_args.get("sql_query")
                            head_n_rows = tool_args.get("head_n_rows", 20)

                            layer_exists = await conn.fetchrow(
                                """
                                SELECT layer_id FROM map_layers
                                WHERE layer_id = $1 AND owner_uuid = $2
                                """,
                                layer_id,
                                user_id,
                            )

                            if not layer_exists:
                                tool_result = {
                                    "status": "error",
                                    "error": f"Layer ID '{layer_id}' not found or you do not have permission to access it.",
                                }
                                await add_chat_completion_message(
                                    ChatCompletionToolMessageParam(
                                        role="tool",
                                        tool_call_id=tool_call.id,
                                        content=json_dumps_safe(tool_result),
                                    )
                                )
                                continue

                            try:
                                # Execute the query using the async function
                                async with kue_ephemeral_action(
                                    conversation.id,
                                    "Querying with SQL...",
                                    layer_id=layer_id,
                                ):
                                    result = await execute_duckdb_query(
                                        sql_query=sql_query,
                                        layer_id=layer_id,
                                        max_n_rows=head_n_rows,
                                        timeout=10,
                                    )

                                # Convert result to CSV format
                                # write header + rows to an in-memory buffer
                                buf = io.StringIO()
                                writer = csv.writer(buf)
                                writer.writerow(result["headers"])
                                writer.writerows(result["result"])

                                result_text = buf.getvalue()

                                if len(result_text) > 25000:
                                    tool_result = {
                                        "status": "error",
                                        "error": f"DuckDB CSV result too large: {len(result_text)} characters exceeds 25,000 character limit, try reducing columns or head_n_rows",
                                    }
                                else:
                                    tool_result = {
                                        "status": "success",
                                        "result": result_text,
                                        "row_count": result["row_count"],
                                        "query": sql_query,
                                    }
                            except HTTPException as e:
                                tool_result = {
                                    "status": "error",
                                    "error": f"DuckDB query error: {e.detail}",
                                }
                            except Exception as e:
                                tool_result = {
                                    "status": "error",
                                    "error": f"Error executing SQL query: {str(e)}",
                                }

                            await add_chat_completion_message(
                                ChatCompletionToolMessageParam(
                                    role="tool",
                                    tool_call_id=tool_call.id,
                                    content=json_dumps_safe(tool_result),
                                )
                            )
                        elif function_name == "set_layer_style":
                            layer_id = tool_args.get("layer_id")
                            maplibre_json_layers_str = tool_args.get(
                                "maplibre_json_layers_str"
                            )

                            if not layer_id or not maplibre_json_layers_str:
                                tool_result = {
                                    "status": "error",
                                    "error": "Missing required parameters (layer_id or maplibre_json_layers_str).",
                                }
                            else:
                                try:
                                    layers = json.loads(maplibre_json_layers_str)

                                    layer_row = await conn.fetchrow(
                                        """
                                        SELECT *
                                        FROM map_layers
                                        WHERE layer_id = $1 AND owner_uuid = $2
                                        """,
                                        layer_id,
                                        user_id,
                                    )
                                    if not layer_row:
                                        raise HTTPException(
                                            404, f"Layer {layer_id} not found"
                                        )
                                    layer = MapLayer(**dict(layer_row))

                                    async with kue_ephemeral_action(
                                        conversation.id,
                                        f"Styling layer {layer.name}...",
                                        update_style_json=True,
                                    ):
                                        style_response = await set_layer_style_route(
                                            request=SetStyleRequest(
                                                maplibre_json_layers=layers,
                                                map_id=map_id,
                                            ),
                                            layer=layer,
                                            user_id=user_id,
                                        )

                                    tool_result = {
                                        "status": "success",
                                        "style_id": style_response.style_id,
                                        "layer_id": style_response.layer_id,
                                        "message": f"Style {style_response.style_id} created and applied to layer {layer_id}",
                                    }

                                except json.JSONDecodeError as e:
                                    tool_result = {
                                        "status": "error",
                                        "error": f"Invalid JSON format: {str(e)}",
                                        "layer_id": layer_id,
                                    }
                                except Exception as e:
                                    tool_result = {
                                        "status": "error",
                                        "error": f"Failed to create and apply style: {str(e)}",
                                        "layer_id": layer_id,
                                    }

                            await add_chat_completion_message(
                                ChatCompletionToolMessageParam(
                                    role="tool",
                                    tool_call_id=tool_call.id,
                                    content=json_dumps_safe(tool_result),
                                ),
                            )
                        elif function_name == "query_postgis_database":
                            postgis_connection_id = tool_args.get(
                                "postgis_connection_id"
                            )
                            sql_query = tool_args.get("sql_query")

                            if not postgis_connection_id or not sql_query:
                                tool_result = {
                                    "status": "error",
                                    "error": "Missing required parameters (postgis_connection_id or sql_query)",
                                }
                            else:
                                # Verify the PostGIS connection exists and user has access
                                connection_result = await conn.fetchrow(
                                    """
                                    SELECT connection_uri FROM project_postgres_connections
                                    WHERE id = $1 AND user_id = $2
                                    """,
                                    postgis_connection_id,
                                    user_id,
                                )

                                if not connection_result:
                                    tool_result = {
                                        "status": "error",
                                        "error": f"PostGIS connection '{postgis_connection_id}' not found or you do not have access to it.",
                                    }
                                else:
                                    try:
                                        # Check if LIMIT is already present and validate it
                                        limited_query = sql_query.strip()
                                        limit_match = re.search(
                                            r"\bLIMIT\s+(\d+)\b",
                                            limited_query,
                                            re.IGNORECASE,
                                        )

                                        if limit_match:
                                            limit_value = int(limit_match.group(1))
                                            if limit_value > 1000:
                                                tool_result = {
                                                    "status": "error",
                                                    "error": f"LIMIT value {limit_value} exceeds maximum allowed limit of 1000",
                                                }
                                                await add_chat_completion_message(
                                                    ChatCompletionToolMessageParam(
                                                        role="tool",
                                                        tool_call_id=tool_call.id,
                                                        content=json_dumps_safe(tool_result),
                                                    ),
                                                )
                                                continue
                                        else:
                                            # No LIMIT found, require explicit LIMIT
                                            tool_result = {
                                                "status": "error",
                                                "error": "Query must include a LIMIT clause with a value less than 1000",
                                            }
                                            await add_chat_completion_message(
                                                ChatCompletionToolMessageParam(
                                                    role="tool",
                                                    tool_call_id=tool_call.id,
                                                    content=json_dumps_safe(tool_result),
                                                ),
                                            )
                                            continue

                                        async with kue_ephemeral_action(
                                            conversation.id,
                                            "Querying PostgreSQL database...",
                                        ):
                                            postgres_conn = await connection_manager.connect_to_postgres(
                                                postgis_connection_id
                                            )
                                            try:
                                                # Execute the query
                                                rows = await postgres_conn.fetch(
                                                    limited_query
                                                )

                                                if not rows:
                                                    tool_result = {
                                                        "status": "success",
                                                        "message": "Query executed successfully but returned no rows",
                                                        "row_count": 0,
                                                        "query": limited_query,
                                                    }
                                                else:
                                                    # Convert rows to list of dicts
                                                    result_data = [
                                                        dict(row) for row in rows
                                                    ]

                                                    # Format the result as a readable string
                                                    if (
                                                        len(result_data) == 1
                                                        and len(result_data[0]) == 1
                                                    ):
                                                        # Single value result
                                                        single_value = list(
                                                            result_data[0].values()
                                                        )[0]
                                                        result_text = f"Query result: {single_value}"
                                                    else:
                                                        # Table format
                                                        if result_data:
                                                            headers = list(
                                                                result_data[0].keys()
                                                            )
                                                            result_lines = [
                                                                "\t".join(headers)
                                                            ]
                                                            for row in result_data:
                                                                result_lines.append(
                                                                    "\t".join(
                                                                        str(
                                                                            row.get(
                                                                                h, ""
                                                                            )
                                                                        )
                                                                        for h in headers
                                                                    )
                                                                )
                                                            result_text = "\n".join(
                                                                result_lines
                                                            )
                                                        else:
                                                            result_text = "No results"

                                                    # Check if result is too large
                                                    if len(result_text) > 25000:
                                                        tool_result = {
                                                            "status": "error",
                                                            "error": f"Query result too large: {len(result_text)} characters exceeds 25,000 character limit. Try reducing the number of columns or rows.",
                                                        }
                                                    else:
                                                        tool_result = {
                                                            "status": "success",
                                                            "result": result_text,
                                                            "row_count": len(
                                                                result_data
                                                            ),
                                                            "query": limited_query,
                                                        }
                                            finally:
                                                await postgres_conn.close()

                                    except Exception as e:
                                        tool_result = {
                                            "status": "error",
                                            "error": f"PostgreSQL query error: {str(e)}",
                                            "query": limited_query,
                                        }

                            await add_chat_completion_message(
                                ChatCompletionToolMessageParam(
                                    role="tool",
                                    tool_call_id=tool_call.id,
                                    content=json_dumps_safe(tool_result),
                                ),
                            )

                        elif function_name in geoprocessing_function_names:
                            tool_result = await run_geoprocessing_tool(
                                tool_call,
                                conn,
                                user_id,
                                map_id,
                                conversation.id,
                            )
                            await add_chat_completion_message(
                                ChatCompletionToolMessageParam(
                                    role="tool",
                                    tool_call_id=tool_call.id,
                                    content=json_dumps_safe(tool_result),
                                ),
                            )
                        else:
                            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

            # Async connections auto-commit, no need for explicit commit

        # Label the conversation if it still has the default "title pending"
        # if conversation.title == "title pending":
        #     await label_conversation_inline(conversation.id)

        # Unlock the map when processing is complete
        redis.delete(f"chat_lock:{conversation.id}")


class MessageSendRequest(BaseModel):
    message: ChatCompletionUserMessageParam
    selected_feature: SelectedFeature | None


class MessageSendResponse(BaseModel):
    conversation_id: int
    sent_message: SanitizedMessage
    message_id: str
    status: str


@router.post(
    "/conversations/{conversation_id}/maps/{map_id}/send",
    response_model=MessageSendResponse,
    operation_id="send_map_message",
)
async def send_map_message(
    request: Request,
    map_id: str,
    body: MessageSendRequest,
    background_tasks: BackgroundTasks,
    await_end: bool = False,
    conversation: Conversation = Depends(get_or_create_conversation),
    session: UserContext = Depends(verify_session_required),
    postgis_provider: Callable = Depends(get_postgis_provider),
    layer_describer: LayerDescriber = Depends(get_layer_describer),
    chat_args: ChatArgsProvider = Depends(get_chat_args_provider),
    map_state: MapStateProvider = Depends(get_map_state_provider),
    system_prompt_provider: SystemPromptProvider = Depends(get_system_prompt_provider),
    connection_manager: PostgresConnectionManager = Depends(
        get_postgres_connection_manager
    ),
    pydantic_tool_calls: PydanticToolRegistry = Depends(get_pydantic_tool_calls),
):
    # get_conversation authenticates
    user_id = session.get_user_id()

    # Check if map is already being processed
    lock_key = f"chat_lock:{conversation.id}"
    if redis.get(lock_key):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conversation is currently being processed by another request",
        )

    # Lock the conversation for processing
    redis.set(lock_key, "locked", ex=30)  # 30 second expiry

    # Use map state provider to generate system messages
    messages_response = await get_all_conversation_messages(conversation.id, session)
    current_messages = [msg.message_json for msg in messages_response]

    current_map_description = await get_map_description(
        request,
        map_id,
        session,
        postgis_provider=postgis_provider,
        layer_describer=layer_describer,
        connection_manager=connection_manager,
    )
    description_text = current_map_description.body.decode("utf-8")

    # Get system messages from the provider
    system_messages = await map_state.get_system_messages(
        current_messages, description_text, body.selected_feature
    )

    async with async_conn("send_map_message.update_messages") as conn:
        # Add any generated system messages to the database
        for system_msg in system_messages:
            system_message = ChatCompletionSystemMessageParam(
                role="system",
                content=system_msg["content"],
            )

            await conn.execute(
                """
                INSERT INTO chat_completion_messages
                (map_id, sender_id, message_json, conversation_id)
                VALUES ($1, $2, $3, $4)
                """,
                map_id,
                user_id,
                json.dumps(system_message),
                conversation.id,
            )

        # Add user's message to DB
        user_msg_db = await conn.fetchrow(
            """
            INSERT INTO chat_completion_messages
            (map_id, sender_id, message_json, conversation_id)
            VALUES ($1, $2, $3, $4)
            RETURNING id, conversation_id, map_id, sender_id, message_json, created_at
            """,
            map_id,
            user_id,
            json.dumps(body.message),
            conversation.id,
        )

        user_msg_dict = dict(user_msg_db)
        user_msg_dict["message_json"] = json.loads(user_msg_dict["message_json"])

        user_msg = MundiChatCompletionMessage(**user_msg_dict)
        sanitized_user_msg = convert_mundi_message_to_sanitized(user_msg)

    # Start processing either synchronously (await_end=True) or in background
    if await_end:
        await process_chat_interaction_task(
            request,
            map_id,
            session,
            user_id,
            chat_args,
            map_state,
            conversation,
            system_prompt_provider,
            connection_manager,
            pydantic_tool_calls,
        )
    else:
        background_tasks.add_task(
            process_chat_interaction_task,
            request,
            map_id,
            session,
            user_id,
            chat_args,
            map_state,
            conversation,
            system_prompt_provider,
            connection_manager,
            pydantic_tool_calls,
        )

    return MessageSendResponse(
        conversation_id=conversation.id,
        sent_message=sanitized_user_msg,
        message_id=str(user_msg_db["id"]),
        status="processing_started",
    )


@router.post(
    "/{map_id}/messages/cancel",
    operation_id="cancel_map_message",
    response_class=JSONResponse,
)
async def cancel_map_message(
    request: Request,
    map_id: str,
    session: UserContext = Depends(verify_session_required),
):
    async with async_conn("cancel_map_message") as conn:
        # Authenticate and check map
        map_result = await conn.fetchrow(
            "SELECT owner_uuid FROM user_mundiai_maps WHERE id = $1 AND soft_deleted_at IS NULL",
            map_id,
        )

        if not map_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Map not found"
            )

        if session.get_user_id() != str(map_result["owner_uuid"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        redis.set(f"messages:{map_id}:cancelled", "true", ex=300)  # 5 minute expiry

        return JSONResponse(content={"status": "cancelled"})
    
