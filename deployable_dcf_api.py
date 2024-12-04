
from flask import Flask, request, jsonify, send_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import norm
import os

app = Flask(__name__)

# Function to fetch stock data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    historical = stock.history(period="5y")
    return info, historical

# Monte Carlo DCF function
def monte_carlo_dcf(cash_flows, growth_rate, discount_rate, iterations=1000):
    results = []
    for _ in range(iterations):
        simulated_rate = np.random.normal(growth_rate, 0.01)  # Add small randomness to growth rate
        simulated_discount = np.random.normal(discount_rate, 0.01)  # Add small randomness to discount rate
        dcf_value = sum(cf / (1 + simulated_discount)**i for i, cf in enumerate(cash_flows, 1))
        results.append(dcf_value)
    return results

# Visualization function
def create_visualization(results, output_file="simulation_plot.png"):
    sns.histplot(results, kde=True, stat="density")
    plt.title("Monte Carlo DCF Simulation Results")
    plt.xlabel("DCF Value")
    plt.ylabel("Density")
    plt.savefig(output_file)
    plt.close()

# API Endpoints
@app.route("/fetch-data", methods=["GET"])
def fetch_data():
    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400
    try:
        info, historical = fetch_stock_data(ticker)
        return jsonify({"info": info, "historical": historical.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/run-dcf", methods=["POST"])
def run_dcf():
    data = request.get_json()
    cash_flows = data.get("cash_flows")
    growth_rate = data.get("growth_rate")
    discount_rate = data.get("discount_rate")
    iterations = data.get("iterations", 1000)

    if not all([cash_flows, growth_rate, discount_rate]):
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        results = monte_carlo_dcf(cash_flows, growth_rate, discount_rate, iterations)
        return jsonify({"results": results, "mean_value": np.mean(results), "std_dev": np.std(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/visualize", methods=["POST"])
def visualize():
    data = request.get_json()
    results = data.get("results")
    if not results:
        return jsonify({"error": "Results data is required"}), 400

    try:
        output_file = "simulation_plot.png"
        create_visualization(results, output_file)
        return send_file(output_file, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
