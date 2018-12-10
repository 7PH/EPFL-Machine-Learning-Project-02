from src.ben.model import get_model, tokenize
from src.loader.loader import load_data

data = load_data(sample=True)
tokenized = tokenize(data['train'], data['test'])
model = get_model(tokenized['nb_features'], tokenized['train_sequences_pad'])
model.fit(
    tokenized['train_sequences_pad'],
    data['train']['label'].values,
    validation_split=.1,
    epochs=1,
    batch_size=128,
    verbose=True
)
pred_train = model.predict_proba(tokenized['train_sequences_pad'], batch_size=128)

print("\nData summary")
print(data['train'].head(4))
print(data['test'].head(4))

print("\nTokens")
print(tokenized['train_sequences_pad'])

print("\nModel summary")
print(model.summary())
print("\nPrediction")
print(pred_train)
