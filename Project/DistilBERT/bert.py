import pandas as pd
import torch
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

def Bert():
  #notebook_login()
  
  filepath_train = r"/home/student/Desktop/train.csv"
  train = pd.read_csv(filepath_train, encoding='latin-1') 
  
  filepath_test = r"/home/student/Desktop/test.csv"
  test = pd.read_csv(filepath_test, encoding='latin-1') 

  X_train = train["Email Text"].tolist()
  y_train = train["Email Type"].map({'Safe Email': 0, 'Phishing Email': 1}).tolist()
  
  X_test = test["Email Text"].tolist()
  y_test = test["Email Type"].map({'Safe Email': 0, 'Phishing Email': 1}).tolist() 

  tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

  X_train = [str(text) for text in X_train]
  X_test = [str(text) for text in X_test]

  train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512)
  test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=512)

  with open('train_encodings.pkl', 'wb') as f:
    pickle.dump(train_encodings, f)

  with open('test_encodings.pkl', 'wb') as f:
      pickle.dump(test_encodings, f)

  # with open('train_edistilbert-phishing-detectionncodings.pkl', 'rb') as f:
  #     train_encodings = pickle.load(f)

  # with open('val_encodings.pkl', 'rb') as f:
  #     val_encodings = pickle.load(f)

  class EmailDataset(torch.utils.data.Dataset):
      def __init__(self, encodings, labels):
          self.encodings = encodings
          self.labels = labels

      def __getitem__(self, idx):
          item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
          item['labels'] = torch.tensor(self.labels[idx])
          return item

      def __len__(self):
          return len(self.labels)


  train_dataset = EmailDataset(train_encodings, y_train)
  val_dataset = EmailDataset(test_encodings, y_test)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)
  model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)

  training_args = TrainingArguments(
      output_dir='./results',
      num_train_epochs=3,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=64,
      warmup_steps=500,
      weight_decay=0.01,
      logging_dir='./logs',
      logging_steps=10,
      evaluation_strategy="epoch"
  )

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset
  )

  trainer.train()

  results = trainer.evaluate()
  
  predictions, labels, _ = trainer.predict(val_dataset)
  
  preds = predictions.argmax(axis=1)
  
  cm = confusion_matrix(labels, preds)
  print(cm)
  #model.push_to_hub('jbe1/distilbert-phishing-detection', commit_message="First upload")
  #tokenizer.push_to_hub('jbe1/distilbert-phishing-detection')
  
  model.save_pretrained('./distilbert-phishing-detection')
  tokenizer.save_pretrained('./distilbert-phishing-detection')

Bert()
