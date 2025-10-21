#!/usr/bin/env python3
"""
Test script to verify Vercel deployment
"""
import requests
import json

def test_endpoints():
    base_url = "https://render-for-python-url.vercel.app"
    
    endpoints = [
        "/health",
        "/api/trade-ideas", 
        "/api/ai-trading-recommendations",
        "/api/ai-trade-assistance",
        "/analyze",
        "/predict", 
        "/backtest"
    ]
    
    print("🧪 Testing Vercel Deployment")
    print("=" * 50)
    
    for endpoint in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            print(f"\n🔍 Testing: {endpoint}")
            
            if endpoint in ["/api/ai-trade-assistance"]:
                # POST request
                response = requests.post(url, json={"symbol": "AAPL"})
            else:
                # GET request
                response = requests.get(url)
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   ✅ Success: {json.dumps(data, indent=2)[:200]}...")
                except:
                    print(f"   ✅ Success: {response.text[:200]}...")
            else:
                print(f"   ❌ Error: {response.text[:200]}...")
                
        except Exception as e:
            print(f"   💥 Exception: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🏁 Testing Complete")

if __name__ == "__main__":
    test_endpoints()
