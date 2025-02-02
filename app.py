from flask import Flask, request, jsonify
from old import optimize_model_parameters, generate_results, load_user_artists
from pathlib import Path
import csv
import os
import time

app = Flask(__name__)

if not os.path.exists('results'):
    os.makedirs('results')

# optimize the model parameters
@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        pack_size = int(request.args.get('pack_size', 25))
        iterations = int(request.args.get('iterations', 1000))
        
        start_time = time.time()
        user_artists = load_user_artists(Path("./dataset/user_artists.dat"))
        factors, regularization = optimize_model_parameters(user_artists, pack_size, iterations)
        end_time = time.time()
        
        with open('results/optimized_params.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['factors', 'regularization', 'time', 'pack_size', 'iterations'])
            writer.writerow([factors, regularization, end_time - start_time, pack_size, iterations])
        
        return jsonify({
            "message": "Optimization complete",
            "factors": factors,
            "regularization": regularization,
            "time": end_time - start_time,
            "pack_size": pack_size,
            "iterations": iterations
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# generate the recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        user_index = int(request.args.get('user_index', 2))
        recommend_limit = int(request.args.get('limit', 10))
        
        # throw an error if user_index is missing
        if user_index is None:
            return jsonify({"error": "user_index is required"}), 400
        
        # throw an error if recommend_limit is missing
        if recommend_limit is None:
            return jsonify({"error": "recommend_limit is required"}), 400
        
        # generate the results
        results = generate_results(user_index=user_index, recommend_limit=recommend_limit)
        

        return jsonify({
            "message": f"Recommendations generated for user {user_index}",
            "data": results
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# get the recommendations for a user
@app.route('/view-recommendations', methods=['GET'])
def recommendations():
    try:
        user_index = int(request.args.get('user_index', 2))

        # read file recommendations/recommendation_list_user_{user_index}.csv
        with open(f'results/user_{user_index}/recommendation_list.csv', 'r') as file:
            reader = csv.DictReader(file)
            recommendations = [
                {"artist": row["artist"], "score": float(row["score"])}
                for row in reader
            ]

        return jsonify({
            "message": f"Recommendations for user {user_index}",
            "data": recommendations
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)