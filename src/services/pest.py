from datetime import date, timedelta, datetime as dt
from typing import Dict, Any
import ee
import os

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
        return False

def filter_cloudy_images(s2_collection, cloud_threshold=90):
    """Filter out images with high cloud percentage"""
    return s2_collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))

def mask_clouds(img):
    """Cloud mask for Sentinel-2"""
    qa = img.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).neq(0).Or(qa.bitwiseAnd(1 << 11).neq(0)).Not()
    return img.updateMask(cloud_mask).copyProperties(img, ['system:time_start'])

def calculate_dcbvi(img):
    """Calculate DCBVI for Sentinel-2"""
    # Sentinel-2 bands (scaled to reflectance)
    R410 = img.select('B1').multiply(1e-4)  # Coastal aerosol (443nm - closest to 410nm)
    R718 = img.select('B5').multiply(1e-4)  # Red Edge 1 (705nm - closest to 718nm)
    
    # Calculate DCBVI = R410 - R718
    dcbvi = R410.subtract(R718).rename('DCBVI')
    
    return img.addBands(dcbvi).copyProperties(img, ['system:time_start'])

def extract_dcbvi_mean(img, roi):
    """Extract mean DCBVI for each image"""
    mean_dcbvi = img.select('DCBVI').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=10,
        maxPixels=1e10,
        bestEffort=True
    ).get('DCBVI')
    
    date = img.date().format('YYYY-MM-dd')
    return ee.Feature(None, {
        'date': date,
        'mean_dcbvi': mean_dcbvi,
        'timestamp': img.date().millis()
    })

def get_dcbvi_geotiff_url(s2_collection, roi):
    """Get latest DCBVI GeoTIFF URL from GEE"""
    try:
        # Get the most recent image
        latest_image = s2_collection.sort('system:time_start', False).first()
        
        export_image = latest_image.select('DCBVI').clip(roi)
        
        # Get min/max values for scaling
        minMax = export_image.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=roi,
            scale=10,
            maxPixels=1e10,
            bestEffort=True
        )
        
        dcbvi_min = minMax.get('DCBVI_min')
        dcbvi_max = minMax.get('DCBVI_max')
        
        # Color palette for DCBVI (red=stressed, yellow=mild, green=healthy)
        color_palette = ['ff0000', 'ff8800', 'ffff00', '88ff00', '00ff00']
        
        # Create visualization
        rgb_image = export_image.visualize(
            min=dcbvi_min,
            max=dcbvi_max,
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
        return None

async def detect_pest(latitude: float, longitude: float) -> Dict[str, Any]:
    """Detect pest stress using DCBVI from Sentinel-2 data
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        
    Returns:
        Dictionary containing:
        - detected_pest: Boolean indicating if pest stress is detected
        - confidence_score: Confidence level (0-1)
        - dcbvi_value: Current DCBVI value
        - image_url: Latest DCBVI visualization URL
    """
    try:
        # Initialize Earth Engine
        if not initialize_earth_engine():
            raise Exception("Failed to initialize Earth Engine")
        
        # Create ROI from lat/lon (same as bloom detection)
        point = ee.Geometry.Point([longitude, latitude])
        roi = point.buffer(500).bounds()
        
        # Get recent data (last 30 days)
        end_date = ee.Date(date.today().isoformat())
        start_date = end_date.advance(-30, 'day')
        
        # Load Sentinel-2 data
        s2_collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(roi)
                        .filterDate(start_date, end_date)
                        .map(mask_clouds)
                        .map(calculate_dcbvi)
                        .map(lambda img: img.clip(roi)))
        
        num_images = s2_collection.size().getInfo()
        if num_images == 0:
            raise Exception("No Sentinel-2 data available for the location")

        # Filter out cloudy images
        s2_collection = filter_cloudy_images(s2_collection, cloud_threshold=75)
        
        # Extract time series
        time_series_fc = s2_collection.map(lambda img: extract_dcbvi_mean(img, roi)).filter(ee.Filter.notNull(['mean_dcbvi']))
        time_series_data = time_series_fc.getInfo()
        
        if not time_series_data['features']:
            raise Exception("No valid DCBVI data extracted")
        
        # Get most recent DCBVI value
        most_recent_dcbvi = None
        most_recent_date = None
        
        for feature in time_series_data['features']:
            props = feature['properties']
            dcbvi_val = props['mean_dcbvi']
            timestamp = props['timestamp']
            
            if dcbvi_val and (most_recent_date is None or timestamp > most_recent_date):
                most_recent_dcbvi = dcbvi_val
                most_recent_date = timestamp
        
        if most_recent_dcbvi is None:
            raise Exception("Could not extract recent DCBVI value")
        
        # Analyze pest stress based on DCBVI value
        # Higher DCBVI values indicate healthier plants
        if most_recent_dcbvi >= 0.1:
            detected_pest = False
            confidence_score = 0.9
        elif most_recent_dcbvi >= 0.0:
            detected_pest = True
            confidence_score = 0.6
        else:
            detected_pest = True
            confidence_score = 0.8
        
        # Get DCBVI GeoTIFF URL
        image_url = get_dcbvi_geotiff_url(s2_collection, roi)
        
        return {
            "detected_pest": detected_pest,
            "confidence_score": round(confidence_score, 2),
            "image_url": image_url,
        }
        
    except Exception as e:
        return {
            "detected_pest": False,
            "confidence_score": 0.0,
            "image_url": None,
        }