from flask import Flask, render_template, request
import os
# from programs.generate_text import generated_text
from programs.generate_sonnet import generate_sonnet

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/generate/')
def generate():
    # output = generated_text()
    if request.method == 'POST':
        seedText = request.form['seedText']
        output = generate_sonnet(seedText)
    else:
        output = ''
    return render_template('generate.html', output=output)

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0',port=int(os.environ.get('PORT', 8080))) 