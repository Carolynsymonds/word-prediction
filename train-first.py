import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
from nltk.tokenize import word_tokenize
import nltk
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def text_to_indices(sentence, vocab):

  numerical_sentence = []

  for token in sentence:
    if token in vocab:
      numerical_sentence.append(vocab[token])
    else:
      numerical_sentence.append(vocab['<unk>'])

  return numerical_sentence

# Function to calculate accuracy
def calculate_accuracy(model, dataloader1, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to compute gradients
        for batch_x, batch_y in dataloader1:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Get model predictions
            outputs = model(batch_x)

            # Get the predicted word indices
            _, predicted = torch.max(outputs, dim=1)

            # Compare with actual labels
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total * 100
    return accuracy

class CustomDataset(Dataset):

  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

# prediction

def prediction(model, vocab, text):

  # tokenize
  tokenized_text = word_tokenize(text.lower())

  # text -> numerical indices
  numerical_text = text_to_indices(tokenized_text, vocab)

  # padding
  padded_text = torch.tensor([0] * (61 - len(numerical_text)) + numerical_text, dtype=torch.long).unsqueeze(0)

  # send to model
  output = model(padded_text)

  # predicted index
  value, index = torch.max(output, dim=1)

  # merge with text
  return text + " " + list(vocab.keys())[index]

class LSTMModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, 100)
    self.lstm = nn.LSTM(100, 150, batch_first=True, dropout=0.2)
    self.fc = nn.Linear(150, vocab_size)

  def forward(self, x):
    embedded = self.embedding(x)
    intermediate_hidden_states, (final_hidden_state, final_cell_state) = self.lstm(embedded)
    output = self.fc(final_hidden_state.squeeze(0))
    return output

def main():
    document = """About the Program
    What is the course fee for  Data Science Mentorship Program (DSMP 2023)
    The course follows a monthly subscription model where you have to make monthly payments of Rs 799/month.
    What is the total duration of the course?
    The total duration of the course is 7 months. So the total course fee becomes 799*7 = Rs 5600(approx.)
    What is the syllabus of the mentorship program?
    We will be covering the following modules:
    Python Fundamentals
    Python libraries for Data Science
    Data Analysis
    SQL for Data Science
    Maths for Machine Learning
    ML Algorithms
    Practical ML
    MLOPs
    Case studies
    You can check the detailed syllabus here - https://learnwith.campusx.in/courses/CampusX-Data-Science-Mentorship-Program-637339afe4b0615a1bbed390
    Will Deep Learning and NLP be a part of this program?
    No, NLP and Deep Learning both are not a part of this program’s curriculum.
    What if I miss a live session? Will I get a recording of the session?
    Yes all our sessions are recorded, so even if you miss a session you can go back and watch the recording.
    Where can I find the class schedule?
    Checkout this google sheet to see month by month time table of the course - https://docs.google.com/spreadsheets/d/16OoTax_A6ORAeCg4emgexhqqPv3noQPYKU7RJ6ArOzk/edit?usp=sharing.
    What is the time duration of all the live sessions?
    Roughly, all the sessions last 2 hours.
    What is the language spoken by the instructor during the sessions?
    Hinglish
    How will I be informed about the upcoming class?
    You will get a mail from our side before every paid session once you become a paid user.
    Can I do this course if I am from a non-tech background?
    Yes, absolutely.
    I am late, can I join the program in the middle?
    Absolutely, you can join the program anytime.
    If I join/pay in the middle, will I be able to see all the past lectures?
    Yes, once you make the payment you will be able to see all the past content in your dashboard.
    Where do I have to submit the task?
    You don’t have to submit the task. We will provide you with the solutions, you have to self evaluate the task yourself.
    Will we do case studies in the program?
    Yes.
    Where can we contact you?
    You can mail us at nitish.campusx@gmail.com
    Payment/Registration related questions
    Where do we have to make our payments? Your YouTube channel or website?
    You have to make all your monthly payments on our website. Here is the link for our website - https://learnwith.campusx.in/
    Can we pay the entire amount of Rs 5600 all at once?
    Unfortunately no, the program follows a monthly subscription model.
    What is the validity of monthly subscription? Suppose if I pay on 15th Jan, then do I have to pay again on 1st Feb or 15th Feb
    15th Feb. The validity period is 30 days from the day you make the payment. So essentially you can join anytime you don’t have to wait for a month to end.
    What if I don’t like the course after making the payment. What is the refund policy?
    You get a 7 days refund period from the day you have made the payment.
    I am living outside India and I am not able to make the payment on the website, what should I do?
    You have to contact us by sending a mail at nitish.campusx@gmail.com
    Post registration queries
    Till when can I view the paid videos on the website?
    This one is tricky, so read carefully. You can watch the videos till your subscription is valid. Suppose you have purchased subscription on 21st Jan, you will be able to watch all the past paid sessions in the period of 21st Jan to 20th Feb. But after 21st Feb you will have to purchase the subscription again.
    But once the course is over and you have paid us Rs 5600(or 7 installments of Rs 799) you will be able to watch the paid sessions till Aug 2024.
    Why lifetime validity is not provided?
    Because of the low course fee.
    Where can I reach out in case of a doubt after the session?
    You will have to fill a google form provided in your dashboard and our team will contact you for a 1 on 1 doubt clearance session
    If I join the program late, can I still ask past week doubts?
    Yes, just select past week doubt in the doubt clearance google form.
    I am living outside India and I am not able to make the payment on the website, what should I do?
    You have to contact us by sending a mail at nitish.campusx@gmai.com
    Certificate and Placement Assistance related queries
    What is the criteria to get the certificate?
    There are 2 criterias:
    You have to pay the entire fee of Rs 5600
    You have to attempt all the course assessments.
    I am joining late. How can I pay payment of the earlier months?
    You will get a link to pay fee of earlier months in your dashboard once you pay for the current month.
    I have read that Placement assistance is a part of this program. What comes under Placement assistance?
    This is to clarify that Placement assistance does not mean Placement guarantee. So we dont guarantee you any jobs or for that matter even interview calls. So if you are planning to join this course just for placements, I am afraid you will be disappointed. Here is what comes under placement assistance
    Portfolio Building sessions
    Soft skill sessions
    Sessions with industry mentors
    Discussion on Job hunting strategies
    """


    tokens = word_tokenize(document.lower())

    # build vocab
    vocab = {'<unk>':0}

    for token in Counter(tokens).keys():
      if token not in vocab:
        vocab[token] = len(vocab)

    input_sentences = document.split('\n')

    input_numerical_sentences = []

    for sentence in input_sentences:
        input_numerical_sentences.append(text_to_indices(word_tokenize(sentence.lower()), vocab))

    training_sequence = []
    for sentence in input_numerical_sentences:

        for i in range(1, len(sentence)):
            training_sequence.append(sentence[:i + 1])

    len_list = []

    for sequence in training_sequence:
        len_list.append(len(sequence))

    max(len_list)

    padded_training_sequence = []
    for sequence in training_sequence:
        padded_training_sequence.append([0] * (max(len_list) - len(sequence)) + sequence)

    padded_training_sequence = torch.tensor(padded_training_sequence, dtype=torch.long)

    X = padded_training_sequence[:, :-1]
    y = padded_training_sequence[:, -1]

    dataset = CustomDataset(X, y)

    model = LSTMModel(len(vocab))

    model.to(device)

    epochs = 50
    learning_rate = 0.001

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define sizes
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size  # Ensures no leftover data
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # training loop

    for epoch in range(epochs):
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            output = model(batch_x)

            loss = criterion(output, batch_y)

            loss.backward()

            optimizer.step()

            total_loss = total_loss + loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)

                val_output = model(val_x)
                loss = criterion(val_output, val_y)
                val_loss += loss.item()

                _, predicted = torch.max(val_output, dim=1)
                correct += (predicted == val_y).sum().item()
                total += val_y.size(0)

        val_accuracy = correct / total * 100
        print(f"Epoch: {epoch + 1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    prediction(model, vocab, "The course follows a monthly")

    num_tokens = 10
    input_text = "hi how are"

    for i in range(num_tokens):
        output_text = prediction(model, vocab, input_text)
        print(output_text)
        input_text = output_text
        time.sleep(0.5)

    # Compute accuracy
    accuracy = calculate_accuracy(model, test_loader, device)
    print(f"Model Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()

    # basic: Model Accuracy: 23.16%
    #2 added