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


@pytest.fixture
async def test_map_id(auth_client):
    payload = {
        "title": "Upload Without Adding Test Map",
        "description": "Test map for layer upload without adding to map",
    }

    response = await auth_client.post(
        "/api/maps/create",
        json=payload,
    )

    assert response.status_code == 200, f"Failed to create map: {response.text}"
    map_id = response.json()["id"]

    return map_id


@pytest.mark.anyio
async def test_upload_without_adding_to_map(test_map_id, auth_client):
    file_path = "test_fixtures/airports.fgb"

    file_size = os.path.getsize(file_path)
    print(f"Testing with file: {file_path}, size: {file_size} bytes")

    with open(file_path, "rb") as f:
        files = {"file": ("airports.fgb", f)}
        data = {
            "layer_name": "Airports",
            "add_layer_to_map": "false",
        }

        response = await auth_client.post(
            f"/api/maps/{test_map_id}/layers",
            files=files,
            data=data,
        )

        assert response.status_code == 200, f"Failed to upload layer: {response.text}"
        layer_id = response.json()["id"]
        print(f"Created layer with ID: {layer_id}")

    response = await auth_client.get(
        f"/api/maps/{test_map_id}/layers",
    )

    assert response.status_code == 200, f"Failed to get layers: {response.text}"
    layers_response = response.json()
    print(f"Layers response: {layers_response}")

    layers = layers_response["layers"]

    assert not any(layer["id"] == layer_id for layer in layers), (
        "Layer was incorrectly added to map"
    )

    response = await auth_client.get(
        f"/api/layer/{layer_id}.geojson",
    )

    assert response.status_code == 200, f"Failed to access layer: {response.text}"
    assert response.headers["Content-Type"] == "application/geo+json"

    geojson = response.json()
    assert "features" in geojson
    assert len(geojson["features"]) > 0


@pytest.mark.anyio
async def test_upload_with_adding_to_map(test_map_id, auth_client):
    file_path = "test_fixtures/airports.fgb"

    with open(file_path, "rb") as f:
        files = {"file": ("airports.fgb", f)}
        data = {
            "layer_name": "Airports Default",
        }

        response = await auth_client.post(
            f"/api/maps/{test_map_id}/layers",
            files=files,
            data=data,
        )

        assert response.status_code == 200, f"Failed to upload layer: {response.text}"
        upload_response = response.json()
        layer_id = upload_response["id"]
        child_map_id = upload_response["dag_child_map_id"]
        print(f"Created layer with ID: {layer_id}")
        print(f"Created child map with ID: {child_map_id}")

    # Check the child map for the layer, not the parent map
    response = await auth_client.get(
        f"/api/maps/{child_map_id}/layers",
    )

    assert response.status_code == 200, f"Failed to get layers: {response.text}"
    layers_response = response.json()

    layers = layers_response["layers"]

    assert any(layer["id"] == layer_id for layer in layers), (
        "Layer was not added to map"
    )

    # Verify the parent map does NOT contain the new layer (DAG immutability)
    parent_response = await auth_client.get(
        f"/api/maps/{test_map_id}/layers",
    )
    assert parent_response.status_code == 200, (
        f"Failed to get parent layers: {parent_response.text}"
    )
    parent_layers = parent_response.json()["layers"]
    assert len(parent_layers) == 0, "Parent map should not contain the uploaded layer"
