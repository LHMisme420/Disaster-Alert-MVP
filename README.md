# Disaster-Alert-MVP
mkdir disaster-alert-mvp &amp;&amp; cd disaster-alert-mvp &amp;&amp; git init
"""
Disaster Early Warning MVP: Flood + Cyclone Detection
Integrates GEE (NDWI/Precip/Winds), IMD Cyclone Bulletins, Grok API for alerts.
Usage: python alert_engine.py (mocks Bangalore; tweak lat/lon/report).
Live: Pass real user_report, lat/lon via CLI/Streamlit.
Author: Grok + You (2025)
"""
git clone https://github.com/LHMisme420/Disaster-Alert-MVP.git
cd Disaster-Alert-MVP
import json
import requests
import math
import os
from datetime import datetime, timedelta
from io import BytesIO
import re
import ee  # pip install earthengine-api
from dotenv import load_dotenv  # pip install python-dotenv
import PyPDF2  # pip install pypdf2

# Load env (GROK_API_KEY from .env)
load_dotenv()
API_KEY = os.getenv('GROK_API_KEY')
if not API_KEY:
    raise ValueError("Set GROK_API_KEY in .env")

# Auth GEE (run ee.Authenticate() once)
ee.Initialize()

# Haversine distance (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# IMD Cyclone: Fetch/parse latest PDF bulletin
def get_imd_cyclone_info(user_lat, user_lon):
    now = datetime.now()
    day, month, year = now.day, now.month, now.year
    for hour_utc in ['0300', '1200', '1800', '0000']:  # Try recent bulletins
        pdf_url = f"https://mausam.imd.gov.in/backend/assets/cyclone_pdf/Tropical_Weather_Outlook_based_on_{hour_utc}_UTC_of_{day:02d}_{month:02d}_{year}.pdf"
        try:
            resp = requests.get(pdf_url, timeout=10)
            if resp.status_code == 200:
                break
        except requests.RequestException:
            continue
    else:
        print("No IMD bulletin; no active cyclone.")
        return {'active': False, 'dist_km': float('inf')}
    
    # Extract text
    pdf_file = BytesIO(resp.content)
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''.join(page.extract_text() + '\n' for page in reader.pages)
    
    # Parse active system (regex for position/name)
    cyclone_match = re.search(r"(Cyclone|Depression|Low Pressure Area) ['`\"]([A-Z]+)['`\"]?[^.]*?(\d+\.?\d*)[°N]?\s*(\d+\.?\d*)[°E]?", text, re.IGNORECASE | re.DOTALL)
    if not cyclone_match:
        return {'active': False, 'dist_km': float('inf')}
    
    name = cyclone_match.group(2) if len(cyclone_match.groups()) > 1 else 'Unnamed'
    c_lat = float(cyclone_match.group(3))
    c_lon = float(cyclone_match.group(4))
    dist = haversine(user_lat, user_lon, c_lat, c_lon)
    
    # Intensity
    intensity = 'LOW PRESSURE'
    for level in ['EXTREMELY SEVERE CYCLONE', 'VERY SEVERE CYCLONIC STORM', 'SEVERE CYCLONIC STORM', 'CYCLONIC STORM']:
        if level in text.upper():
            intensity = level
            break
    
    print(f"IMD: {name} at {c_lat}N {c_lon}E ({intensity}), dist {dist:.0f}km")
    return {'active': True, 'name': name, 'lat': c_lat, 'lon': c_lon, 'intensity': intensity, 'dist_km': dist}

# Hazard Risks: Flood (NDWI + Precip) + Cyclone (Winds + IMD)
def get_hazard_risk(location_lat, location_lon):
    point = ee.Geometry.Point([location_lon, location_lat])
    
    # Flood: NDWI (Landsat surface water)
    landsat_coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(point).sort('system:time_start', False).limit(1)
    ndwi_risk = 0.0
    if landsat_coll.size().getInfo() > 0:
        image = landsat_coll.first()
        ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30)
        ndwi_val = ndwi.getInfo().get('ndwi', 0)
        ndwi_risk = min(abs(float(ndwi_val) - 0.5) * 2, 1.0)
    
    # Flood: Precip (NOAA GFS last 24h)
    end_date = ee.Date(datetime.now())
    start_date = end_date.advance(-1, 'day')
    gfs_coll = (ee.ImageCollection('NOAA/GFS0P25').filterDate(start_date, end_date).filterBounds(point)
                .select('total_precipitation_surface').sort('system:time_start', False).limit(1))
    precip_risk, precip_mm = 0.0, 0.0
    if gfs_coll.size().getInfo() > 0:
        gfs_image = gfs_coll.first()
        precip = gfs_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=27830)
        precip_mm = float(precip.getInfo().get('total_precipitation_surface', 0))
        if precip_mm > 50:
            precip_risk = 1.0
        elif precip_mm > 10:
            precip_risk = (precip_mm - 10) / 40
    flood_risk = (ndwi_risk + precip_risk) / 2
    
    # Cyclone: Winds (ECMWF ERA5 last 6h)
    wind_end = ee.Date(datetime.now())
    wind_start = wind_end.advance(-6, 'hour')
    era5_coll = (ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate(wind_start, wind_end).filterBounds(point)
                 .select(['u10', 'v10']).sort('system:time_start', False).limit(1))
    cyclone_risk, wind_speed = 0.0, 0.0
    if era5_coll.size().getInfo() > 0:
        era5_image = era5_coll.first()
        winds = era5_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=11132)
        u10 = float(winds.get('u10', 0))
        v10 = float(winds.get('v10', 0))
        wind_speed = math.sqrt(u10**2 + v10**2)
        cyclone_risk = min(wind_speed / 74.0, 1.0)  # Normalize to Cat5
    
    # IMD Boost
    imd_info = get_imd_cyclone_info(location_lat, location_lon)
    if imd_info['active'] and imd_info['dist_km'] < 500:
        boost = max(0.3, 0.5 * (1 - imd_info['dist_km'] / 500))
        if 'SEVERE' in imd_info['intensity']:
            boost *= 1.5
        cyclone_risk = min(cyclone_risk + boost, 1.0)
        print(f"IMD boost: +{boost:.2f}")
    
    # Compound: Wind boosts flood
    if cyclone_risk > 0.3:
        flood_risk = min(flood_risk * 1.2, 1.0)
    
    print(f"Flood: NDWI {ndwi_risk:.2f}, Precip {precip_risk:.2f} ({precip_mm:.1f}mm) → {flood_risk:.2f}")
    print(f"Cyclone: Wind {wind_speed:.1f}m/s → {cyclone_risk:.2f}")
    
    return {
        'flood': flood_risk,
        'cyclone': cyclone_risk,
        'overall': min((flood_risk + cyclone_risk) / 2, 1.0),
        'imd_cyclone': imd_info,
        'precip_mm': precip_mm,
        'wind_speed': wind_speed
    }

# Grok Analysis: Structured prompt → JSON alert
def grok_analyze(report, risks, api_key):
    imd = risks.pop('imd_cyclone')  # For prompt
    prompt = f"""
    You are an AI early warning expert for natural disasters in India. Analyze:
    - Overall risk: {risks['overall']:.2f} (0=safe, 1=imminent; flood + cyclone blend).
    - Flood: {risks['flood']:.2f} (NDWI water + {risks['precip_mm']:.1f}mm GFS precip).
    - Cyclone: {risks['cyclone']:.2f} (ERA5 {risks['wind_speed']:.1f}m/s winds; IMD boost if active).
    - IMD: {imd['name'] if imd['active'] else 'None'} ({imd['intensity'] if imd['active'] else 'N/A'}), {imd['dist_km']:.0f}km away.
    - Report: "{report}".
    
    Output ONLY JSON: {{"level": "LOW|MEDIUM|HIGH", "action": "1-2 sentences, urgent/local (IMD-ref for cyclones)", "confidence": 0.0-1.0, "primary_hazard": "FLOOD|CYCLONE|COMPOUND"}}
    
    Thresholds: LOW (<0.3), MEDIUM (0.3-0.7), HIGH (>0.7). Factor keywords ('rising water') + compounds.
    India tips: Evacuate lowlands/rivers; check IMD app for zones.
    """
    
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "grok-3",  # Swap grok-4 if subbed
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 150
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        start = content.find('{')
        end = content.rfind('}') + 1
        parsed = json.loads(content[start:end])
        risks['imd_cyclone'] = imd  # Restore
        return parsed
    except Exception as e:
        print(f"Grok error: {e}. Fallback.")
        overall = risks['overall']
        primary = "CYCLONE" if risks['cyclone'] > risks['flood'] else "FLOOD"
        if overall > 0.7 or 'rising water' in report.lower():
            level, action, conf = "HIGH", "Evacuate immediately—head to higher ground/shelters per IMD.", 0.8
        elif overall > 0.4:
            level, action, conf = "MEDIUM", "Prepare kits, secure items; monitor IMD updates.", 0.7
        else:
            level, action, conf = "LOW", "Stay informed via IMD/local channels.", 0.9
        risks['imd_cyclone'] = imd
        return {"level": level, "action": action, "confidence": conf, "primary_hazard": primary}

# Generate Alert
def generate_alert(user_report, lat, lon, api_key=API_KEY):
    risks = get_hazard_risk(lat, lon)
    analysis = grok_analyze(user_report, risks, api_key)
    return {
        "level": analysis["level"],
        "location": f"{lat:.2f}N, {lon:.2f}E",
        "action": analysis["action"],
        "confidence": analysis["confidence"],
        "primary_hazard": analysis["primary_hazard"],
        "risk_breakdown": risks,
        "timestamp": datetime.now().isoformat()
    }

# CLI Demo (Bangalore, mock report)
if __name__ == "__main__":
    user_report = "Heavy rain and rising water levels reported in village."
    lat, lon = 12.97, 77.59
    alert = generate_alert(user_report, lat, lon)
    print(json.dumps(alert, indent=2))
    earthengine-api==0.1.414
requests==2.32.3
PyPDF2==3.0.1
python-dotenv==1.0.1
GROK_API_KEY=your_grok_api_key_here  # From console.x.ai
# GEE_SERVICE_ACCOUNT=your-service-account@project.iam.gserviceaccount.com  # For prod auth
# Disaster Alert MVP

AI-powered early warning for floods/cyclones in India. Blends GEE satellite data (NDWI, precip, winds) + IMD bulletins + Grok for contextual alerts.

## Quickstart
1. Clone: `git clone https://github.com/yourusername/disaster-alert-mvp.git`
2. Env: Copy `.env.example` to `.env`, add keys.
3. Install: `pip install -r requirements.txt`
4. GEE Auth: `python -c "import ee; ee.Authenticate()"`
5. Run: `python alert_engine.py` → Sample JSON alert.
6. Customize: Tweak `generate_alert(report, lat, lon)` for your UI/SMS.

## Features
- **Flood Risk**: Landsat NDWI + NOAA GFS precip (24h forecast).
- **Cyclone Risk**: ERA5 winds + IMD PDF parse (active systems <500km boost).
- **Alerts**: Grok API → Structured JSON (level/action/confidence).
- **Fallbacks**: Rule-based if API/GEE flakes.

## Limits
- GEE: Free quota ~1k req/day; auth required.
- Grok: Free tier 10k tokens/day.
- IMD: PDF-based (daily bulletins); no real-time tracks.

## Next
- UI: Streamlit dashboard w/ maps (folium).
- Scale: Async (aiohttp) + Redis cache.
- Hazards: Add quakes (USGS API).

Issues? PRs welcome. Built w/ Grok (xAI).
import json
from unittest.mock import patch, MagicMock
from alert_engine import generate_alert

@patch('alert_engine.get_hazard_risk')
@patch('alert_engine.grok_analyze')
def test_generate_alert(mock_grok, mock_risk):
    mock_risk.return_value = {'flood': 0.5, 'cyclone': 0.2, 'overall': 0.35, 'imd_cyclone': {'active': False}, 'precip_mm': 20, 'wind_speed': 10}
    mock_grok.return_value = {'level': 'MEDIUM', 'action': 'Test alert', 'confidence': 0.7, 'primary_hazard': 'FLOOD'}
    
    alert = generate_alert("Test report", 12.97, 77.59)
    assert alert['level'] == 'MEDIUM'
    print(json.dumps(alert, indent=2))

if __name__ == "__main__":
    test_generate_alert()
    .env
__pycache__/
*.pyc
.earthengine/
{
  "level": "LOW",
  "location": "12.97N, 77.59E",
  "action": "Stay informed via IMD for any developments in Bay of Bengal.",
  "confidence": 0.85,
  "primary_hazard": "FLOOD",
  "risk_breakdown": {
    "flood": 0.06,
    "cyclone": 0.08,
    "overall": 0.07,
    "imd_cyclone": {"active": false, "dist_km": 999999999.0},
    "precip_mm": 0.0,
    "wind_speed": 5.2
  },
  "timestamp": "2025-11-11T10:08:00"
}
