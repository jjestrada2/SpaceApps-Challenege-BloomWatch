#!/usr/bin/env node
/*
 * Copyright (C) 2025 Bunting Labs, Inc.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
const geoViewport = require('@mapbox/geo-viewport');

let inputData = '';

process.stdin.on('data', (chunk) => {
  inputData += chunk;
});

process.stdin.on('end', () => {
  try {
    const payload = JSON.parse(inputData);

    if (!payload.bbox || !payload.width || !payload.height) {
      throw new Error('Missing required parameters: bbox, width, and height are required');
    }

    const boundsArr = typeof payload.bbox === 'string'
      ? payload.bbox.split(',').map(Number)
      : payload.bbox;

    const viewport = geoViewport.viewport(
      boundsArr,
      [payload.width, payload.height],
      undefined,
      undefined,
      512,
      true
    );

    console.log(JSON.stringify({
      zoom: viewport.zoom,
      center: viewport.center
    }));
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
});

process.stdin.resume();

