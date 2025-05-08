from flask import Flask, request, render_template_string
from pyngrok import ngrok
import pandas as pd
import threading

# Set up ngrok
ngrok.set_auth_token("2wp5vJWBKWioQYfjdHFYHhWrXWw_3NdLWgeuMLf92iEFQPNr9")

# Load processed dataset
books = pd.read_json("book_snippets.json")

# Dummy data: book recommendations by genre
recommendation_data = {
    'Horror': ['Dracula', 'Frankenstein', 'The Shining'],
    'Comedy': ['Good Omens', 'The Hitchhiker’s Guide to the Galaxy', 'Bossypants'],
    'Action': ['The Bourne Identity', 'Die Hard: Year One', 'The Hunger Games'],
    'Science Fiction': ['Dune', 'Ender’s Game', 'Neuromancer'],
    'Nonfiction': ['Sapiens', 'Educated', 'The Immortal Life of Henrietta Lacks']
}

# Flask setup
app = Flask(__name__)

# HTML template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Genre-Based Book Recommender</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; max-width: 600px; margin: auto; background-color: #f4f4f4; }
        label { display: block; margin-top: 15px; }
        .slider-value { font-weight: bold; margin-left: 10px; }
    </style>
</head>
<body>
    <h2>Search Book Title</h2>
    <form method="POST">
        <input type="text" name="search_title" placeholder="Search for a book title" style="width: 100%; padding: 8px; margin-bottom: 20px;">

        <h2>Rate Your Genre Preferences (1-5)</h2>
        {% for genre in genres %}
        <label>{{ genre }}:
            <input type="range" name="{{ genre }}" min="1" max="5" value="3" oninput="document.getElementById('{{ genre }}_val').innerText = this.value">
            <span id="{{ genre }}_val">3</span>
        </label>
        {% endfor %}
        <br>
        <input type="submit" value="Get Book Recommendations">
    </form>

    {% if search_title %}
    <p><strong>Search Title:</strong> {{ search_title }}</p>
    {% endif %}

    {% if recommendations %}
    <h3>Recommendations Based on Your Preferences:</h3>
    <ul>
        {% for rec in recommendations %}
        <li>{{ rec }}</li>
        {% endfor %}
    </ul>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    genres = list(recommendation_data.keys())
    recommendations = []
    search_title = ""
    if request.method == "POST":
        search_title = request.form.get("search_title", "")
        for genre in genres:
            try:
                rating = int(request.form.get(genre, 0))
                if rating >= 3:
                    recommendations.extend(recommendation_data[genre])
            except ValueError:
                pass
    return render_template_string(html_template, genres=genres, recommendations=recommendations, search_title=search_title)

# Launch Flask in background
threading.Thread(target=lambda: app.run()).start()

# Expose via ngrok
print("🔗 Open your app here:", ngrok.connect(5000))
