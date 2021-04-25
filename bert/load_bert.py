from transformers import BertForQuestionAnswering

def load_model():
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    return model

def main():
    model = load_model()
    model.save_pretrained('squad-bert/')

if __name__ == "__main__":
    main()