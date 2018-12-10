

def predict(model, x, no_value=-1, yes_value=1, batch_size=128):
    predicted = model.predict_proba(x, batch_size=batch_size)
    return [yes_value if pred < .5 else no_value for pred in predicted]
