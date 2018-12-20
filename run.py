from src.model.functions import get_model, tokenize, get_embedding_matrix
from src.load.functions import load_data
from src.predict.functions import predict

# Constants
EMBEDDING_DIM = 100
LSTM_OUT = 100

# Load
data = load_data(sample=False)

# Tokenize & Model
tokenized = tokenize(data['train'], data['test'])
embedding_matrix = get_embedding_matrix(tokenized['word_index'], EMBEDDING_DIM)
model = get_model(tokenized['nb_features'], tokenized['train'], embedding_matrix, LSTM_OUT, EMBEDDING_DIM)

model.summary()

# Train
history = model.fit(
    tokenized['train'],
    data['train']['label'],
    validation_split=0.25,
    epochs=1,
    batch_size=128,
    verbose=1
)

# Prediction on train set
pred = predict(model, tokenized['train'])
labels_ok = data['train']['label'].values * 2 - 1 == pred
accuracy = sum(labels_ok) / len(labels_ok)
print("Accuracy on train set: " + str(accuracy))

# Prediction on test set
pred = predict(model, tokenized['test'])

# Store
d = data['test'][['id']]
d['label'] = pred
csv_raw = d.to_csv(index=False, columns=['id', 'label'], header=['Id', 'Prediction'])
with open('dist/prediction.csv', 'w') as file:
    file.write(csv_raw)
