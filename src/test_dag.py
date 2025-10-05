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


@pytest.mark.anyio
async def test_describe_dag(auth_client, test_map_with_vector_layers):
    description = await auth_client.get(
        f"/api/maps/{test_map_with_vector_layers['map_id']}/tree"
    )
    print(description.json())
    print(test_map_with_vector_layers)
    assert description.status_code == 200
    data = description.json()

    assert data["project_id"] == test_map_with_vector_layers["project_id"]

    # Test that tree contains 4 maps
    assert "tree" in data
    assert len(data["tree"]) == 4

    # Test that the first entry has no fork_reason
    assert data["tree"][0]["fork_reason"] is None

    # Test that the last 3 entries have user_edit fork_reason
    for i in range(1, 4):
        assert data["tree"][i]["fork_reason"] == "user_edit"

    # Test that all entries have empty messages
    for entry in data["tree"]:
        assert entry["messages"] == []


@pytest.mark.anyio
async def test_dag_layer_diffs(auth_client, test_map_with_vector_layers):
    """Test that layer diffs are calculated correctly between consecutive maps."""
    description = await auth_client.get(
        f"/api/maps/{test_map_with_vector_layers['map_id']}/tree"
    )
    assert description.status_code == 200
    data = description.json()

    tree = data["tree"]

    # First map should have no diff (no previous map)
    assert tree[0]["diff_from_previous"] is None

    # Second map should have 1 added layer (beaches), 0 removed
    assert tree[1]["diff_from_previous"] is not None
    assert len(tree[1]["diff_from_previous"]["added_layers"]) == 1
    assert len(tree[1]["diff_from_previous"]["removed_layers"]) == 0
    assert (
        tree[1]["diff_from_previous"]["added_layers"][0]["layer_id"]
        == test_map_with_vector_layers["beaches_layer_id"]
    )

    # Third map should have 1 added layer (cafes), 0 removed
    assert tree[2]["diff_from_previous"] is not None
    assert len(tree[2]["diff_from_previous"]["added_layers"]) == 1
    assert len(tree[2]["diff_from_previous"]["removed_layers"]) == 0
    assert (
        tree[2]["diff_from_previous"]["added_layers"][0]["layer_id"]
        == test_map_with_vector_layers["cafes_layer_id"]
    )

    # Fourth map should have 1 added layer (idaho stations), 0 removed
    assert tree[3]["diff_from_previous"] is not None
    assert len(tree[3]["diff_from_previous"]["added_layers"]) == 1
    assert len(tree[3]["diff_from_previous"]["removed_layers"]) == 0
    assert (
        tree[3]["diff_from_previous"]["added_layers"][0]["layer_id"]
        == test_map_with_vector_layers["idaho_stations_layer_id"]
    )
