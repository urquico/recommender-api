from flask import Flask, request, jsonify
from enums import Models
from models import igwo, igwo_results, pigwo, pigwo_results, igwo_evaluate, pigwo_evaluate, evaluate_v2
from igwo import analyze_user_data
import csv
import os
from flask_cors import CORS
from spotify import get_top_songs

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

if not os.path.exists('results'):
    os.makedirs('results')

# optimize the model parameters
@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        pack_size = int(request.args.get('pack_size', 25))
        iterations = int(request.args.get('iterations', 1000))
        model = str(request.args.get('model', 'igwo'))
        
        if model == Models.IGWO:
            factors, regularization, elapsed_time = igwo(pack_size, iterations)
        elif model == Models.PIGWO:
            factors, regularization, elapsed_time = pigwo(pack_size, iterations)
        else:
            return jsonify({"error": "Invalid model"}), 400
        
        return jsonify({
            "message": "Optimization complete",
            "data": {
                "factors": factors,
                "regularization": regularization,
                "elapsed_time": elapsed_time,
                "pack_size": pack_size,
                "iterations": iterations,
            },
            "model": model
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# generate the recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_index = int(request.args.get('user_index', 2))
        recommend_limit = int(request.args.get('limit', 10))
        model = str(request.args.get('model', 'igwo'))

        # throw an error if user_index is missing
        if user_index is None:
            return jsonify({"error": "user_index is required"}), 400
        
        # throw an error if recommend_limit is missing
        if recommend_limit is None:
            return jsonify({"error": "recommend_limit is required"}), 400
        
        # generate the results
        if model == Models.IGWO:
            results = igwo_results(user_index=user_index, recommend_limit=recommend_limit)
        elif model == Models.PIGWO:
            results = pigwo_results(user_index=user_index, recommend_limit=recommend_limit)
        else:
            return jsonify({"error": "Invalid model"}), 400
        
        return jsonify({
            "message": f"Recommendations generated for user {user_index}",
            "data": results,
            "model": model
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# get the recommendations for a user
@app.route('/view-recommendations', methods=['GET'])
def recommendations():
    try:
        user_index = int(request.args.get('user_index', 2))
        model = str(request.args.get('model', 'igwo'))
        
        # throw error if model is not IGWO or PIGWO
        if model not in [Models.IGWO, Models.PIGWO]:
            return jsonify({"error": "Invalid model"}), 400

        # read file recommendations/recommendation_list_user_{user_index}.csv
        with open(f'results/user_{user_index}/recommendation_list_{model}.csv', 'r') as file:
            reader = csv.DictReader(file)
            recommendations = [
                {
                    "artist": row["artist"], 
                    "score": float(row["score"]), 
                    "spotify": get_top_songs(row["artist"])
                }
                for row in reader
            ]

        return jsonify({
            "message": f"Recommendations for user {user_index}",
            "data": recommendations
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# view model optimized parameters
@app.route('/view-optimized-parameters', methods=['GET'])
def view_optimized_parameters():
    try:
        model = str(request.args.get('model', 'igwo'))
        
        if model not in [Models.IGWO, Models.PIGWO]:
            return jsonify({"error": "Invalid model"}), 400
        
        # read the optimized parameters from the file
        with open(f'results/optimized_params_{model}.csv', 'r') as file:
            reader = csv.DictReader(file)
            optimized_parameters = [
                {"factors": int(row["factors"]), "regularization": float(row["regularization"]), "time": float(row["time"]), "pack_size": int(row["pack_size"]), "iterations": int(row["iterations"]), "iteration_counter": int(row["iteration_counter"])}
                for row in reader
            ]

        return jsonify({
            "message": f"Optimized parameters for {model.upper()}",
            "data": optimized_parameters[0]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# evaluate model using evaluate_v2
@app.route('/evaluate-model-v2', methods=['POST'])
def evaluate_model_v2():
    try:
        evaluation = evaluate_v2()
        
        return jsonify({"message": "Model evaluated", "data": {"igwo": evaluation[0], "pigwo": evaluation[1]}}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# get the evaluation metrics
@app.route('/get-evaluation-metrics', methods=['GET'])
def get_evaluation_metrics():
    try:
        # get evaluation metrics from the evaluation_igwo.csv file
        with open(f'results/evaluation_{Models.IGWO}.csv', 'r') as file:
            reader = csv.DictReader(file)
            evaluation_igwo = [row for row in reader]
            
        # get evaluation metrics from the evaluation_pigwo.csv file
        with open(f'results/evaluation_{Models.PIGWO}.csv', 'r') as file:
            reader = csv.DictReader(file)
            evaluation_pigwo = [row for row in reader]
        
        data = [
            {
                "Metric": row_igwo["metric"],
                "IGWO": float(row_igwo["value"]),
                "E-IGWO": float(row_pigwo["value"]),
            }
            for row_igwo, row_pigwo in zip(evaluation_igwo, evaluation_pigwo)
        ]
        
        return jsonify({
            "message": "Evaluation metrics",
            "data": data
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# get the user details
@app.route('/all-users', methods=['GET'])
def all_users():
    try:
        # read all the users from the users folder
        users = []
        for i in range(2, 2101):
            with open(f'users/user_{i}.csv', 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    users.append(row)
                    
        return jsonify({
            "message": "All users",
            "data": users
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# get user top 10 artists
@app.route('/user-top-10-artists', methods=['GET'])
def user_top_10_artists():
    try:
        user_index = int(request.args.get('user_index', 2))
        
        top_10_artists = analyze_user_data(user_index)
        
        return jsonify({
            "message": f"Top 10 artists for user {user_index}",
            "data": top_10_artists
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        

if __name__ == '__main__':
    app.run(debug=True)