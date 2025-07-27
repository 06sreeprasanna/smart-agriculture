from . import db

class YieldData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    crop = db.Column(db.String(100), nullable=False)
    area = db.Column(db.Float, nullable=False)
    yield_amount = db.Column(db.Float, nullable=False)
    date = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f'<YieldData {self.crop}>'