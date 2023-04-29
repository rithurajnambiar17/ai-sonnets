from flask import Flask, render_template, request
from programs.generate_text import generated_text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/generate/')
def generate():
    output = generated_text()
    return render_template('generate.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)