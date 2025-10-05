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

from abc import ABC, abstractmethod
import os


class ChatArgsProvider(ABC):
    @abstractmethod
    async def get_args(self, user_uuid: str, route_name: str) -> dict:
        pass


class DefaultChatArgsProvider(ChatArgsProvider):
    async def get_args(self, user_uuid: str, route_name: str) -> dict:
        # feel free to customize below depending on which ollama model you're using
        # or whichever provider you want to use. you can also change depending on
        # user or route.
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")
        return {"model": model}


def get_chat_args_provider() -> ChatArgsProvider:
    return DefaultChatArgsProvider()
