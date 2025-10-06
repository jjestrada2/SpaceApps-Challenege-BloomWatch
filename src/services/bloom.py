from datetime import date, timedelta, datetime as dt
from typing import Dict, Any
import ee
import os

# Como hacemos para que se llame una sola vez al iniciar la app?
def initialize_earth_engine():
    """Initialize Earth Engine with service account credentials"""    
    try:
        service_account = 'gee-farmane@vaulted-channel-234121.iam.gserviceaccount.com'
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        credentials_path = os.path.join(project_root, 'src', 'services', 'vaulted-channel-234121-376df8d2d29a.json')
        credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
        ee.Initialize(credentials)
        return True
    except Exception as e:
        # print(f"Earth Engine initialization failed: {e}")
        return False

def filter_cloudy_images(s2_collection, cloud_threshold=90):
    """Filter out images with high cloud percentage"""
    return s2_collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))

def mask_clouds(img):
    """Cloud mask for S2"""
    qa = img.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).neq(0).Or(qa.bitwiseAnd(1 << 11).neq(0)).Not()
    return img.updateMask(cloud_mask).copyProperties(img, ['system:time_start'])

def calculate_ebi(img):
    """Calculate Enhanced Bloom Index (EBI)"""
    # Scale bands to reflectance
    R = img.select('B4').multiply(1e-4)
    G = img.select('B3').multiply(1e-4) 
    B = img.select('B2').multiply(1e-4).max(1e-6)  # Prevent division by zero
    
    # EBI calculation
    brightness = R.add(G).add(B)
    greenness = G.divide(B)
    soil_sig = R.subtract(B).add(1.0)  # EPS = 1.0
    ebi = brightness.divide(greenness.multiply(soil_sig)).rename('EBI')
    
    return img.addBands(ebi).copyProperties(img, ['system:time_start'])

def extract_ebi_mean(img, roi):
    """Extract mean EBI for each image"""
    mean_ebi = img.select('EBI').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=10,
        maxPixels=1e10,
        bestEffort=True
    ).get('EBI')
    
    date = img.date().format('YYYY-MM-dd')
    return ee.Feature(None, {
        'date': date,
        'mean_ebi': mean_ebi,
        'timestamp': img.date().millis()
    })

def get_ebi_geotiff_url(s2_collection, peak_date, roi):
    """Peak EBI GeoTIFF URL from GEE"""
    try:                
        # Filter collection
        peak_date_str = peak_date.strftime('%Y-%m-%d')
        peak_day_start = ee.Date(peak_date_str)
        peak_day_end = peak_day_start.advance(1, 'day')
        
        peak_image = s2_collection.filterDate(peak_day_start, peak_day_end).first()
        
        export_image = peak_image.select('EBI').clip(roi)
        # Get min/max values for scaling
        minMax = export_image.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=roi,
            scale=10,
            maxPixels=1e10,
            bestEffort=True
        )
        
        ebi_min = minMax.get('EBI_min')
        ebi_max = minMax.get('EBI_max')
        
        color_palette = ['0d0887', '6300a7', 'ab2494', 'e34f6f', 'fb9f3a', 'f0f921']
        
        # Create scaled visualization
        rgb_image = export_image.visualize(
            min=ebi_min,
            max=ebi_max,
            palette=color_palette
        )
        
        # Get download URL
        url = rgb_image.getDownloadURL({
            'scale': 10,
            'crs': 'EPSG:4326',
            'region': roi,
            'format': 'GEO_TIFF'
        })
        
        return url

    except Exception as e:
        # print(f"    Download failed: {e}")
        return None

async def detect_bloom(latitude: float, longitude: float) -> Dict[str, Any]:
    """Detect bloom peak date and EBI value for a given lat/lon using Sentinel-2 data

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        Dictionary containing:
        - latitude: Input latitude
        - longitude: Input longitude 
        - date_of_max_ebi: Date of peak bloom
        - ebi_value: Peak EBI value
        - image_url: Peak EBI GeoTIFF URL from GEE
    """
    try:
        # Initialize Earth Engine
        if not initialize_earth_engine():
            raise Exception("Failed to initialize Earth Engine")
        
        # Create ROI from lat/lon (100 hectares square around the center point)
        point = ee.Geometry.Point([longitude, latitude])
        roi = point.buffer(500).bounds()  # Creates a 100-hectare square
        
        # Get current year and define bloom season
        current_year = dt.now().year
        start_date = ee.Date.fromYMD(current_year, 1, 15)
        end_date = ee.Date.fromYMD(current_year, 4, 15)
        
        # Load Sentinel-2 data
        s2_collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(roi)
                        .filterDate(start_date, end_date)
                        .map(mask_clouds)
                        .map(calculate_ebi)
                        .map(lambda img: img.clip(roi)))
        
        num_images = s2_collection.size().getInfo()
        if num_images == 0:
            # If no data for current year, try previous year
            current_year = current_year - 1
            start_date = ee.Date.fromYMD(current_year, 1, 15)
            end_date = ee.Date.fromYMD(current_year, 4, 15)
            
            s2_collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                            .filterBounds(roi)
                            .filterDate(start_date, end_date)
                            .map(mask_clouds)
                            .map(calculate_ebi)
                            .map(lambda img: img.clip(roi)))
            
            num_images = s2_collection.size().getInfo()
            
            if num_images == 0:
                raise Exception("No Sentinel-2 data available for the location")

        #Filter out cloudy images
        s2_collection = filter_cloudy_images(s2_collection, cloud_threshold=75)
        
        # Extract time series
        time_series_fc = s2_collection.map(lambda img: extract_ebi_mean(img, roi)).filter(ee.Filter.notNull(['mean_ebi']))
        time_series_data = time_series_fc.getInfo()
        
        if not time_series_data['features']:
            raise Exception("No valid EBI data extracted")
        
        # Find peak bloom
        max_ebi = 0
        peak_date = None
        
        for feature in time_series_data['features']:
            props = feature['properties']
            ebi_val = props['mean_ebi']
            date_str = props['date']
            
            if ebi_val and ebi_val > max_ebi:
                max_ebi = ebi_val
                peak_date = dt.strptime(date_str, '%Y-%m-%d').date()
        
        if peak_date is None:
            raise Exception("Could not determine peak bloom date")
        
        # Get EBI GeoTIFF URL
        image_url = get_ebi_geotiff_url(s2_collection, peak_date, roi)
        
        return {
            "latitude": latitude,
            "longitude": longitude,
            "date_of_max_ebi": peak_date,
            "ebi_value": round(max_ebi, 3),
            "image_url": image_url,
        }
        
    except Exception as e:
        # print(f"Bloom detection failed: {e}")
        return {
            "latitude": latitude,
            "longitude": longitude,
            "date_of_max_ebi": date.today(),
            "ebi_value": 0.00,
            "image_url": "https://example.com/tiles/ebi/mock.png",
        }


async def predict_bloom(latitude: float, longitude: float) -> Dict[str, Any]:
    """Return dummy bloom prediction data for a given lat/lon.

    This is a placeholder. Replace with real logic that uses
    historical phenology and a predictive model.
    """
    # Use fixed seasonal dates: start on Feb 20, peak on Mar 12 of the current year
    year = dt.now().year
    start = date(year, 2, 20)
    peak = date(year, 3, 12)
    return {
        "latitude": latitude,
        "longitude": longitude,
        "predicted_bloom_start": start,
        "predicted_bloom_peak": peak,
        "confidence": 0.82,
    }


# ========================
# Utilidad para CMIP6
# ========================
def to_df(data, var):
    header, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=header)
    df = df[["time", var]].dropna()
    df["date"] = pd.to_datetime(df["time"], unit="ms")
    df[var] = df[var] - 273.15  # Kelvin → °C
    return df[["date", var]]


# ========================
# Predicción de floración
# ========================
async def predict_bloom(lat: float, lon: float, year: int = 2026,
                        chill_req: int = 400, heat_req: int = 200, tbase: float = 7.5,
                        model: str = "ACCESS-CM2", scenario: str = "ssp245",
                        project_id: str = "maps-474019") -> Dict[str, Any]:
    """
    Predice floración (bloom) usando datos NASA POWER y CMIP6.
    """

    #service_account = os.getenv("SERVICE_ACCOUNT_EMAIL")
    #key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    #project = os.getenv("PROJECT_ID")

    if not initialize_earth_engine():
            raise Exception("Failed to initialize Earth Engine")

    # Temporada para el año especificado: octubre (año-1) a junio (año)
    season_start = date(year - 1, 10, 1)
    season_end = date(year, 6, 30)
    today = date.today()

    # ========================================
    # 1. NASA POWER (solo si el año incluye "hoy" o está en el pasado)
    # ========================================
    df_real = pd.DataFrame(columns=["datetime", "temp", "step"])
    if season_start <= today <= season_end or year <= today.year:
        actual_end = min(today, season_end)

        URL = (
            "https://power.larc.nasa.gov/api/temporal/hourly/point"
            f"?parameters=T2M&community=AG&latitude={lat}&longitude={lon}"
            f"&start={season_start.strftime('%Y%m%d')}&end={actual_end.strftime('%Y%m%d')}"
            "&format=JSON"
        )
        print("Fetching NASA POWER:", URL)

        r = requests.get(URL)
        data = r.json()

        if "properties" in data and "parameter" in data["properties"]:
            t2m = data["properties"]["parameter"]["T2M"]
            df_real = pd.DataFrame({
                "datetime": pd.to_datetime(list(t2m.keys()), format="%Y%m%d%H"),
                "temp": list(t2m.values()),
                "step": "hour"
            })
            df_real = df_real[df_real["temp"] > -900]

    # ========================================
    # 2. CMIP6 (futuro)
    # ========================================
    #creds = ee.ServiceAccountCredentials(service_account, key_path)
    #ee.Initialize(creds, project=project)
    point = ee.Geometry.Point([lon, lat])
    cmip6 = ee.ImageCollection("NASA/GDDP-CMIP6")

    start_future = max(today, season_start)
    tasmax = (cmip6.filterDate(start_future.strftime("%Y-%m-%d"), season_end.strftime("%Y-%m-%d"))
              .filter(ee.Filter.eq("model", model))
              .filter(ee.Filter.eq("scenario", scenario))
              .select("tasmax")).getRegion(point, 27830).getInfo()

    tasmin = (cmip6.filterDate(start_future.strftime("%Y-%m-%d"), season_end.strftime("%Y-%m-%d"))
              .filter(ee.Filter.eq("model", model))
              .filter(ee.Filter.eq("scenario", scenario))
              .select("tasmin")).getRegion(point, 27830).getInfo()

    df_future = pd.DataFrame()
    if tasmax and tasmin:
        df_max = to_df(tasmax, "tasmax")
        df_min = to_df(tasmin, "tasmin")
        df_future = pd.merge(df_max, df_min, on="date", how="inner")
        df_future["temp"] = (df_future["tasmax"] + df_future["tasmin"]) / 2
        df_future = df_future.rename(columns={"date": "datetime"})
        df_future["step"] = "day"
        df_future = df_future[["datetime", "temp", "step"]]
    else:
        print("CMIP6 devolvió vacío, verificar modelo/escenario.")

    # ========================================
    # 3. Unión y cálculo chill/heat
    # ========================================
    frames = [d for d in [df_real, df_future] if not d.empty]
    df_all = pd.concat(frames).sort_values("datetime").reset_index(drop=True)

    chill = 0
    heat_accum = 0
    chill_date = None
    bloom_date = None
    chills, heats = [], []

    for _, row in df_all.iterrows():
        T = row["temp"]
        if T < -100:
            chills.append(chill)
            heats.append(heat_accum)
            continue

        if row["step"] == "hour":
            if 0 <= T <= 8.3:
                chill += 1
            if chill_date:
                heat_accum += max(0, T - tbase) / 24.0
        else:
            if T <= 8.3:
                chill += 24
            if chill_date:
                heat_accum += max(0, T - tbase)

        if not chill_date and chill >= chill_req:
            chill_date = row["datetime"]

        if chill_date and not bloom_date and heat_accum >= heat_req:
            bloom_date = row["datetime"]

        chills.append(chill)
        heats.append(heat_accum)

    df_all["Chill_accum"] = chills
    df_all["Heat_accum"] = heats

    # ========================================
    # 4. Resultados y reintento si no hay bloom
    # ========================================
    #relaxed_attempt = 0
    #chill_req_final = chill_req
    #heat_req_final = heat_req

    if not bloom_date:
        retry_factor = 0.9  # reduce umbrales un 10% por intento
        max_retries = 3

        for i in range(max_retries):
            chill = 0
            heat_accum = 0
            chill_date = None
            bloom_date = None
            #relaxed_attempt = i + 1

            # Ajuste progresivo de requerimientos
            chill_req_relaxed = int(chill_req * (retry_factor ** (i + 1)))
            heat_req_relaxed = int(heat_req * (retry_factor ** (i + 1)))

            for _, row in df_all.iterrows():
                T = row["temp"]
                if T < -100:
                    continue

                if row["step"] == "hour":
                    if 0 <= T <= 8.3:
                        chill += 1
                    if chill_date:
                        heat_accum += max(0, T - tbase) / 24.0
                else:
                    if T <= 8.3:
                        chill += 24
                    if chill_date:
                        heat_accum += max(0, T - tbase)

                if not chill_date and chill >= chill_req_relaxed:
                    chill_date = row["datetime"]

                if chill_date and not bloom_date and heat_accum >= heat_req_relaxed:
                    bloom_date = row["datetime"]

            if bloom_date:
                #chill_req_final = chill_req_relaxed
                #heat_req_final = heat_req_relaxed
                break

        # Si aún no se consigue floración
        if not bloom_date:
            bloom_date = date(year, 4, 30)  # Fecha fallback razonable
            #relaxed_attempt = "none"

    # ========================================
    # 5. Resultado final
    # ========================================
    confidence = confidence_score(bloom_date)


    return {
        "latitude": lat,
        "longitude": lon,
        "predicted_bloom_start": bloom_date,
        "predicted_bloom_peak": bloom_date,
        "confidence": confidence,
    }


async def generate_timelapse(
    latitude: float,
    longitude: float,
    make_gif: bool = False,
    year: int | None = None,
    fps: int = 6,
) -> Dict[str, Any]:
    """Dummy timelapse generator. Returns placeholder URLs only.

    Real logic intentionally omitted per request.
    """
    placeholder = "https://example.com/tiles/ebi/mock.gif"
    return {
        "latitude": latitude,
        "longitude": longitude,
        "timelapse_url": placeholder,
        "gif_url": placeholder if make_gif else None,
    }

