from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()

def create_app():
    """Application factory function"""
    app = Flask(__name__)
    
    # Configuration
    app.config.from_mapping(
        SECRET_KEY='dev',
        SQLALCHEMY_DATABASE_URI='sqlite:///agriculture.db',
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER='app/static/uploads',
        ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg'}
    )

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)

    # Register blueprints
    register_blueprints(app)

    return app

def register_blueprints(app):
    """Register all blueprints"""
    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)