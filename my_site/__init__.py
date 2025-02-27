from flask import Flask
from flask_wtf import CSRFProtect

from my_site.code.home import pageViews

def create_app():
    app = Flask(__name__)
    csrf = CSRFProtect(app)
    app.config['SECRET_KEY'] = 'ihatemyself'

    app.register_blueprint(pageViews, prefix='/')

    return app