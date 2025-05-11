# app/api.py
from flask import Blueprint, request, jsonify
from app.recommender.content import content_based_filtering, collaborative_filtering
from app.recommender.regression import ensemble_predict

api = Blueprint("api", __name__)

def register_routes(app):
    app.register_blueprint(api, url_prefix="/api")

@api.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@api.route("/recommend", methods=["POST"])
def recommend_books():
    data = request.json or {}
    user_id = data.get("user_id")
    selected_books = data.get("selected_books", [])
    alpha = float(data.get("alpha", 0.5))

    # If user skipped selection, fallback to 5 classic defaults
    if not selected_books:
        selected_books = [
            "Pride and Prejudice",     # Classic Romance
            "Dracula",                 # Classic Horror
            "1984",                    # Dystopian Sci-Fi
            "To Kill a Mockingbird",   # Classic American Lit
            "The Hobbit"               # Fantasy
        ]

    if not user_id:
        return jsonify({"error": "Missing 'user_id'"}), 400
    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({"error": "'user_id' must be an integer"}), 400

    # Aggregate ensemble predictions from each selected book
    all_scores = {}
    for title in selected_books:
        scores = ensemble_predict(title, user_id, alpha)
        for book, score in scores.items():
            all_scores[book] = all_scores.get(book, 0) + score

    # Rank and return top 10
    top_books = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    recommendations = [{"title": title, "score": round(score, 4)} for title, score in top_books]

    return jsonify({
        "user_id": user_id,
        "input_books": selected_books,
        "recommendations": recommendations
    })

@api.route("/recommend/content", methods=["GET"])
def recommend_by_content():
    title = request.args.get("title")
    if not title:
        return jsonify({"error": "Missing 'title' query parameter"}), 400
    recommendations = content_based_filtering(title)
    return jsonify({"title": title, "recommendations": recommendations})

@api.route("/recommend/collaborative", methods=["GET"])
def recommend_by_collaborative():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing 'user_id' query parameter"}), 400
    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({"error": "'user_id' must be an integer"}), 400

    recommendations = collaborative_filtering(user_id)
    return jsonify({"user_id": user_id, "recommendations": recommendations})
@api.route("/recommend/ensemble", methods=["GET"])
def recommend_ensemble():
    title = request.args.get("title")
    user_id = request.args.get("user_id")
    alpha = request.args.get("alpha", 0.5)

    if not title or not user_id:
        return jsonify({"error": "Missing 'title' or 'user_id' parameter"}), 400
    try:
        user_id = int(user_id)
        alpha = float(alpha)
    except ValueError:
        return jsonify({"error": "'user_id' must be int and 'alpha' must be float"}), 400

    results = ensemble_predict(title, user_id, alpha)
    return jsonify({"title": title, "user_id": user_id, "results": results})