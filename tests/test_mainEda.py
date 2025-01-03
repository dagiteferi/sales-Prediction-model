import unittest
import os
import sys
import pandas as pd

# Add the scripts folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from mainEda import EDA

class TestEDA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eda = EDA(train_path='../data/train.csv', test_path='../data/test.csv', store_path='../data/store.csv')
        cls.eda.load_data()

    def test_load_data(self):
        self.assertIsInstance(self.eda.train, pd.DataFrame)
        self.assertIsInstance(self.eda.test, pd.DataFrame)
        self.assertIsInstance(self.eda.store, pd.DataFrame)

    def test_seasonal_sales_trends(self):
        try:
            self.eda.seasonal_sales_trends()
        except Exception as e:
            self.fail(f"seasonal_sales_trends() raised {e} unexpectedly!")

    def test_correlation_analysis(self):
        try:
            self.eda.correlation_analysis(self.eda.train[['Sales', 'Customers']])
        except Exception as e:
            self.fail(f"correlation_analysis() raised {e} unexpectedly!")

    def test_label_encode_columns(self):
        try:
            correlation_matrix = self.eda.label_encode_columns()
            self.assertIsInstance(correlation_matrix, pd.DataFrame)
        except Exception as e:
            self.fail(f"label_encode_columns() raised {e} unexpectedly!")

    def test_promo_analysis(self):
        try:
            self.eda.promo_analysis()
        except Exception as e:
            self.fail(f"promo_analysis() raised {e} unexpectedly!")

    def test_promo2_analysis(self):
        try:
            self.eda.promo2_analysis()
        except Exception as e:
            self.fail(f"promo2_analysis() raised {e} unexpectedly!")

    def test_promo2_distribution_analysis(self):
        try:
            self.eda.promo2_distribution_analysis()
        except Exception as e:
            self.fail(f"promo2_distribution_analysis() raised {e} unexpectedly!")

    def test_assortment_sales_analysis(self):
        try:
            self.eda.assortment_sales_analysis()
        except Exception as e:
            self.fail(f"assortment_sales_analysis() raised {e} unexpectedly!")

    def test_assortment_effect_analysis(self):
        try:
            self.eda.assortment_effect_analysis()
        except Exception as e:
            self.fail(f"assortment_effect_analysis() raised {e} unexpectedly!")

    def test_assortment_competition_distance(self):
        try:
            self.eda.assortment_competition_distance()
        except Exception as e:
            self.fail(f"assortment_competition_distance() raised {e} unexpectedly!")

    def test_competition_effect_analysis(self):
        try:
            self.eda.competition_effect_analysis()
        except Exception as e:
            self.fail(f"competition_effect_analysis() raised {e} unexpectedly!")

    def test_city_center_analysis(self):
        try:
            self.eda.city_center_analysis()
        except Exception as e:
            self.fail(f"city_center_analysis() raised {e} unexpectedly!")

if __name__ == '__main__':
    unittest.main()
