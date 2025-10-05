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

const sharp = require('sharp');
const maplibregl = require('@maplibre/maplibre-gl-native');
const geoViewport = require('@mapbox/geo-viewport');

let inputData = '';

process.stdin.on('data', (chunk) => {
  inputData += chunk;
});

process.stdin.on('end', async () => {
  const payload = JSON.parse(inputData);

  const style = typeof payload.style === 'string' ? JSON.parse(payload.style) : payload.style;

  const options = {
    width: parseInt(payload.width, 10),
    height: parseInt(payload.height, 10),
    pixelRatio: parseFloat(payload.ratio) || 1,
  };

  const map = new maplibregl.Map(options);
  map.load(style);
  maplibregl.on('message', (msg) => {
    try {
      console.log(JSON.stringify(msg));
    } catch (e) {
      try {
        console.log(JSON.stringify({
          class: msg && msg.class || 'Unknown',
          severity: msg && msg.severity || 'INFO',
          text: msg && msg.text || String(msg)
        }));
      } catch (_) {
      }
    }
  })

  if (payload.center) {
    const center = Array.isArray(payload.center) ? payload.center : [0, 0];
    const zoom = payload.zoom || 0;

    map.setCenter(center);
    options.center = center;
    map.setZoom(zoom);
    options.zoom = zoom;
    if (payload.bearing) {
      map.setBearing(payload.bearing);
    }

    if (payload.pitch) {
      map.setPitch(payload.pitch);
    }
  } else if (payload.bounds) {
    const boundsArr = typeof payload.bounds === 'string'
      ? payload.bounds.split(',').map(Number)
      : payload.bounds;

    const viewport = geoViewport.viewport(
      boundsArr,
      [options.width, options.height],
      undefined,
      undefined,
      512,
      true
    );

    map.setCenter(viewport.center);
    options.center = viewport.center;
    map.setZoom(viewport.zoom);
    options.zoom = viewport.zoom;
  }

  map.render(options, (err, buffer) => {
    if (err) {
      try {
        console.error(JSON.stringify({
          type: 'RenderError',
          severity: 'ERROR',
          message: (err && err.message) || String(err)
        }));
      } catch (_) {
        try { console.error('Render error:', String(err)); } catch (_) {}
      }
    } else {
      var image = sharp(buffer, {
        raw: {
          width: options.width,
          height: options.height,
          channels: 4
        }
      });

      image.toFile(process.argv[2], function (err) {
        if (err) throw err;
      });
    }
  });
});

process.stdin.resume();
