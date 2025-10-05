# Copyright (C) 2025 Bunting Labs, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

FROM camptocamp/qgis-server:3.44-gdal3.10-desktop
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-psycopg2 python3-pyqt5 python3-owslib \
        python3-lxml python3-yaml python3-jinja2 \
        python3-fastapi \
    && rm -rf /var/lib/apt/lists/*

ENV QT_QPA_PLATFORM=offscreen
ENV XDG_RUNTIME_DIR=/tmp/xdg-runtime
RUN mkdir -p /tmp && chmod 777 /tmp
RUN mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"

WORKDIR /app

COPY server.py .

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8817"]

