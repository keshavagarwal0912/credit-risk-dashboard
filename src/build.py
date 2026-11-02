"""One command to rebuild everything: python -m src.build"""
import warnings

from src import anomaly, ingest, model

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("1/3 ingest + data quality"); ingest.build()
    print("2/3 default model"); model.train()
    print("3/3 anomaly detection"); anomaly.detect()
    print("done -> data/processed/, artifacts/")
