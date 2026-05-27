import requests
import streamlit as st

def recommend_hospitals():
    # =========================
    # GET LOCATION
    # =========================
    latitude = st.session_state.get("latitude")
    longitude = st.session_state.get("longitude")

    if latitude is None or longitude is None:
        st.error("Location not available")
        return

    # =========================
    # GOOGLE API
    # =========================
    api_key = st.secrets["GOOGLE_API_KEY"]
    url = "https://places.googleapis.com/v1/places:searchNearby"
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.rating"
    }

    data = {
        "includedTypes": ["hospital"],
        "maxResultCount": 5,
        "rankPreference": "DISTANCE",
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "radius": 5000
            }
        }
    }

    response = requests.post(
        url,
        headers=headers,
        json=data
    )

    # =========================
    # RESPONSE
    # =========================
    if response.status_code == 200:
        results = response.json().get("places", [])

        if len(results) == 0:
            st.warning("No nearby hospitals found.")
        else:
            st.subheader("Nearby Hospitals")
            for i, place in enumerate(results, start=1):
                name = place.get("displayName", {}).get("text", "Unknown Hospital")
                address = place.get("formattedAddress", "No address")
                rating = place.get("rating", "No rating")

                st.write(f"### {i}. {name}")
                st.write(f"Address: {address}")
                st.write(f"Rating: {rating}")
                st.divider()
    else:
        st.error("Failed to fetch hospitals")
        st.write(response.text)