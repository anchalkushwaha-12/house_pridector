import unittest
import json
import os
import sys

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app

class TestHousePriceAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'running')

    def test_health_route(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['model_loaded'])

    def test_predict_success(self):
        payload = {
            "Area(sqft)": 2500,
            "Rooms": 3,
            "Bathrooms": 2,
            "Floors": 2,
            "Location": "Suburban",
            "YearBuilt": 2010,
            "Parking": 2,
            "Condition": "Good"
        }
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('predicted_price', data)
        self.assertGreater(data['predicted_price'], 0)
        print(f"Test Prediction Result: {data['formatted_price']} (${data['price_per_sqft']}/sqft)")

    def test_predict_missing_feature(self):
        payload = {
            "Area(sqft)": 2500,
            "Rooms": 3
        }
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
