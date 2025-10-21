#!/usr/bin/env python3
"""
Test script to verify Vercel deployment endpoints
"""
import requests
import json

def test_endpoint(base_url, endpoint, method='GET', data=None):
    """Test a single endpoint"""
    url = f"{base_url}{endpoint}"
    print(f"\n🧪 Testing {method} {url}")
    
    try:
        if method == 'GET':
            response = requests.get(url, timeout=10)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=10)
        
        print(f"✅ Status: {response.status_code}")
        print(f"📄 Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return False

def main():
    # Replace with your actual Vercel URL
    base_url = input("Enter your Vercel deployment URL (e.g., https://your-app.vercel.app): ").strip()
    
    if not base_url:
        print("❌ No URL provided")
        return
    
    if not base_url.startswith('http'):
        base_url = f"https://{base_url}"
    
    print(f"🚀 Testing Vercel deployment at: {base_url}")
    
    # Test endpoints
    tests = [
        ('/health', 'GET', None),
        ('/analyze', 'POST', {'symbol': 'AAPL'}),
        ('/predict', 'POST', {'symbol': 'AAPL', 'timeframe': '1 week'}),
        ('/backtest', 'POST', {'user_id': 'test123', 'initial_capital': 10000})
    ]
    
    results = []
    for endpoint, method, data in tests:
        success = test_endpoint(base_url, endpoint, method, data)
        results.append((endpoint, success))
    
    # Summary
    print(f"\n📊 Test Results:")
    print("=" * 50)
    for endpoint, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {endpoint}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Vercel deployment is working!")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
